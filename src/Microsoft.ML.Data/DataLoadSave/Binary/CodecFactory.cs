// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data.IO
{
    internal sealed partial class CodecFactory
    {
        // REVIEW: In future, this scheme might probably use loadable classes with
        // assembly attributes instead of having the mapping from load name to reader hard coded.
        // Or maybe not. That may depend on how much flexibility we really need from this.
        private readonly Dictionary<string, GetCodecFromStreamDelegate> _loadNameToCodecCreator;
        // The non-vector non-generic types can have a very simple codec mapping.
        private readonly Dictionary<DataKind, IValueCodec> _simpleCodecTypeMap;
        // A shared object pool of memory buffers. Objects returned to the memory stream pool
        // should be cleared and have position set to 0. Use the ReturnMemoryStream helper method.
        private readonly MemoryStreamPool _memPool;

        // This is the encoding used for strings and textspans.
        private readonly Encoding _encoding;

        private readonly IHost _host;

        private delegate bool GetCodecFromStreamDelegate(Stream definitionStream, out IValueCodec codec);

        private delegate bool GetCodecFromTypeDelegate(ColumnType type, out IValueCodec codec);

        public CodecFactory(IHostEnvironment env, MemoryStreamPool memPool = null)
        {
            Contracts.AssertValue(env, "env");
            Contracts.AssertValueOrNull(memPool);

            _host = env.Register("CodecFactory");

            _memPool = memPool ?? new MemoryStreamPool();
            _encoding = Encoding.UTF8;

            _loadNameToCodecCreator = new Dictionary<string, GetCodecFromStreamDelegate>();
            _simpleCodecTypeMap = new Dictionary<DataKind, IValueCodec>();
            // Register the current codecs.
            RegisterSimpleCodec(new UnsafeTypeCodec<sbyte>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<byte>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<Int16>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<ushort>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<int>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<uint>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<Int64>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<ulong>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<Single>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<Double>(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<DvTimeSpan>(this));
            RegisterSimpleCodec(new DvTextCodec(this));
            RegisterSimpleCodec(new BoolCodec(this));
            RegisterSimpleCodec(new DateTimeCodec(this));
            RegisterSimpleCodec(new DateTimeZoneCodec(this));
            RegisterSimpleCodec(new UnsafeTypeCodec<UInt128>(this));

            // Register the old boolean reading codec.
            var oldBool = new OldBoolCodec(this);
            RegisterOtherCodec(oldBool.LoadName, oldBool.GetCodec);

            RegisterOtherCodec("VBuffer", GetVBufferCodec);
            RegisterOtherCodec("Key", GetKeyCodec);
        }

        private BinaryWriter OpenBinaryWriter(Stream stream)
        {
            return new BinaryWriter(stream, _encoding, leaveOpen: true);
        }

        private BinaryReader OpenBinaryReader(Stream stream)
        {
            return new BinaryReader(stream, _encoding, leaveOpen: true);
        }

        private void RegisterSimpleCodec<T>(SimpleCodec<T> codec)
        {
            Contracts.Assert(!_loadNameToCodecCreator.ContainsKey(codec.LoadName));
            Contracts.Assert(!_simpleCodecTypeMap.ContainsKey(codec.Type.RawKind));
            _loadNameToCodecCreator.Add(codec.LoadName, codec.GetCodec);
            _simpleCodecTypeMap.Add(codec.Type.RawKind, codec);
        }

        private void RegisterOtherCodec(string name, GetCodecFromStreamDelegate fn)
        {
            Contracts.Assert(!_loadNameToCodecCreator.ContainsKey(name));
            _loadNameToCodecCreator.Add(name, fn);
        }

        public bool TryGetCodec(ColumnType type, out IValueCodec codec)
        {
            // Handle the primier types specially.
            if (type.IsKey)
                return GetKeyCodec(type, out codec);
            if (type.IsVector)
                return GetVBufferCodec(type, out codec);
            return _simpleCodecTypeMap.TryGetValue(type.RawKind, out codec);
        }

        /// <summary>
        /// Given a codec, write a type description to a stream, from which this codec can be
        /// reconstructed later. This returns the number of bytes written, so that, if this
        /// were a seekable stream, the positions would differ by this amount before and after
        /// a call to this method.
        /// </summary>
        public int WriteCodec(Stream definitionStream, IValueCodec codec)
        {
            // *** Codec type description ***
            // string: codec loadname
            // LEB128 int: Byte size of the parameterization
            // byte[]: The indicated parameterization

            using (BinaryWriter writer = OpenBinaryWriter(definitionStream))
            {
                string loadName = codec.LoadName;
                writer.Write(loadName);
                int bytes = _encoding.GetByteCount(loadName);
                bytes = checked(bytes + Utils.Leb128IntLength((uint)bytes));
                MemoryStream mem = _memPool.Get();
                int output = codec.WriteParameterization(mem);
                Contracts.Check(mem.Length == output, "codec description length did not match stream length");
                Contracts.Check(mem.Length <= int.MaxValue); // Is this even possible in the current implementation of MemoryStream?
                writer.WriteLeb128Int((ulong)mem.Length);
                bytes = checked(bytes + Utils.Leb128IntLength((uint)mem.Length) + output);
                mem.Position = 0;
                mem.CopyTo(definitionStream);
                _memPool.Return(ref mem);
                return bytes;
            }
        }

        /// <summary>
        /// Attempts to define a codec, given a stream positioned at the start of a serialized
        /// codec type definition.
        /// </summary>
        /// <param name="definitionStream">The input stream, which whether this returns true or false
        /// will be left at the end of the codec type definition</param>
        /// <param name="codec">A codec castable to a generic <c>IValueCodec{T}</c> where
        /// <c>typeof(T)==codec.Type.RawType</c></param>
        /// <returns>Whether the codec type definition was understood. If true the codec has defined
        /// value, and should be usable. If false, the name of the codec was unrecognized. Note that
        /// malformed definitions are detected, this will throw instead of returning either true or
        /// false.</returns>
        public bool TryReadCodec(Stream definitionStream, out IValueCodec codec)
        {
            Contracts.AssertValue(definitionStream, "definitionStream");

            using (IChannel ch = _host.Start("TryGetCodec"))
            using (BinaryReader reader = new BinaryReader(definitionStream, Encoding.UTF8, true))
            {
                string signature = reader.ReadString();
                Contracts.CheckDecode(!string.IsNullOrEmpty(signature), "Non-empty signature string expected");
                ulong ulen = reader.ReadLeb128Int();
                Contracts.CheckDecode(ulen <= long.MaxValue, "Codec type definition read from stream too large");
                long len = (long)ulen;
                GetCodecFromStreamDelegate del;
                if (!_loadNameToCodecCreator.TryGetValue(signature, out del))
                {
                    codec = default(IValueCodec);
                    if (len == 0)
                        return false;
                    // Move the stream past the end of the definition.
                    if (definitionStream.CanSeek)
                    {
                        long remaining = definitionStream.Length - definitionStream.Position;
                        if (remaining < len)
                            throw ch.ExceptDecode("Codec type definition supposedly has {0} bytes, but end-of-stream reached after {1} bytes", len, remaining);
                        definitionStream.Seek(len, SeekOrigin.Current);
                    }
                    else
                    {
                        for (long i = 0; i < len; ++i)
                        {
                            if (definitionStream.ReadByte() == -1)
                                throw ch.ExceptDecode("Codec type definition supposedly has {0} bytes, but end-of-stream reached after {1} bytes", len, i);
                        }
                    }
                    ch.Warning("Did not recognize value codec signature '{0}'", signature);
                    ch.Done();
                    return false;
                }
                // Opportunistically validate in the case of a seekable stream.
                long pos = definitionStream.CanSeek ? definitionStream.Position : -1;
                bool retval = del(definitionStream, out codec);
                if (definitionStream.CanSeek && definitionStream.Position - pos != len)
                    throw ch.ExceptDecode("Codec type definition supposedly has {0} bytes, but the handler consumed {1}", len, definitionStream.Position - pos);
                ch.Done();
                return retval;
            }
        }
    }
}
