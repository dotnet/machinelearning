// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Model
{
    /// <summary>
    /// This is a convenience context object for saving models to a repository, for
    /// implementors of <see cref="ICanSaveModel"/>. It is not mandated but designed to reduce the
    /// amount of boiler plate code. It can also be used when saving to a single stream,
    /// for implementors of <see cref="ICanSaveInBinaryFormat"/>.
    /// </summary>
    public sealed partial class ModelSaveContext : IDisposable
    {
        /// <summary>
        /// When in repository mode, this is the repository we're writing to. It is null when
        /// in single-stream mode.
        /// </summary>
        [BestFriend]
        internal readonly RepositoryWriter Repository;

        /// <summary>
        /// When in repository mode, this is the directory we're reading from. Null means the root
        /// of the repository. It is always null in single-stream mode.
        /// </summary>
        [BestFriend]
        internal readonly string Directory;

        /// <summary>
        /// The main stream writer.
        /// </summary>
        [BestFriend]
        internal readonly BinaryWriter Writer;

        /// <summary>
        /// The strings that will be saved in the main stream's string table.
        /// </summary>
        [BestFriend]
        internal readonly NormStr.Pool Strings;

        /// <summary>
        /// The main stream's model header.
        /// </summary>
        [BestFriend]
        internal ModelHeader Header;

        /// <summary>
        /// The min file position of the main stream.
        /// </summary>
        [BestFriend]
        internal readonly long FpMin;

        /// <summary>
        /// The wrapped entry.
        /// </summary>
        private readonly Repository.Entry _ent;

        /// <summary>
        /// Exception context provided by Repository (can be null).
        /// </summary>
        private readonly IExceptionContext _ectx;

        /// <summary>
        /// The assembly name where the loader resides.
        /// </summary>
        private string _loaderAssemblyName;

        /// <summary>
        /// Returns whether this context is in repository mode (true) or single-stream mode (false).
        /// </summary>
        [BestFriend]
        internal bool InRepository => Repository != null;

        /// <summary>
        /// Create a <see cref="ModelSaveContext"/> supporting saving to a repository, for implementors of <see cref="ICanSaveModel"/>.
        /// </summary>
        internal ModelSaveContext(RepositoryWriter rep, string dir, string name)
        {
            Contracts.CheckValue(rep, nameof(rep));
            Repository = rep;
            _ectx = rep.ExceptionContext;

            _ectx.CheckValueOrNull(dir);
            _ectx.CheckNonEmpty(name, nameof(name));

            Directory = dir;
            Strings = new NormStr.Pool();

            _ent = rep.CreateEntry(dir, name);
            try
            {
                Writer = new BinaryWriter(_ent.Stream, Encoding.UTF8, leaveOpen: true);
                try
                {
                    ModelHeader.BeginWrite(Writer, out FpMin, out Header);
                }
                catch
                {
                    Writer.Dispose();
                    throw;
                }
            }
            catch
            {
                _ent.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Create a <see cref="ModelSaveContext"/> supporting saving to a single-stream, for implementors of <see cref="ICanSaveInBinaryFormat"/>.
        /// </summary>
        internal ModelSaveContext(BinaryWriter writer, IExceptionContext ectx = null)
        {
            Contracts.AssertValueOrNull(ectx);
            _ectx = ectx;
            _ectx.CheckValue(writer, nameof(writer));

            Repository = null;
            Directory = null;
            _ent = null;

            Strings = new NormStr.Pool();
            Writer = writer;
            ModelHeader.BeginWrite(Writer, out FpMin, out Header);
        }

        [BestFriend]
        internal void CheckAtModel()
        {
            _ectx.Check(Writer.BaseStream.Position == FpMin + Header.FpModel);
        }

        /// <summary>
        /// Set the version information in the main stream's header. This should be called before
        /// <see cref="Done"/> is called.
        /// </summary>
        /// <param name="ver"></param>
        [BestFriend]
        internal void SetVersionInfo(VersionInfo ver)
        {
            ModelHeader.SetVersionInfo(ref Header, ver);
            _loaderAssemblyName = ver.LoaderAssemblyName;
        }

        [BestFriend]
        internal void SaveTextStream(string name, Action<TextWriter> action)
        {
            _ectx.Check(InRepository, "Can't save a text stream when writing to a single stream");
            _ectx.CheckNonEmpty(name, nameof(name));
            _ectx.CheckValue(action, nameof(action));

            // I verified in the CLR source that the default buffer size is 1024. It's unfortunate
            // that to set leaveOpen to true, we have to specify the buffer size....
            using (var ent = Repository.CreateEntry(Directory, name))
            using (var writer = Utils.OpenWriter(ent.Stream))
            {
                action(writer);
            }
        }

        [BestFriend]
        internal void SaveBinaryStream(string name, Action<BinaryWriter> action)
        {
            _ectx.Check(InRepository, "Can't save a text stream when writing to a single stream");
            _ectx.CheckNonEmpty(name, nameof(name));
            _ectx.CheckValue(action, nameof(action));

            // I verified in the CLR source that the default buffer size is 1024. It's unfortunate
            // that to set leaveOpen to true, we have to specify the buffer size....
            using (var ent = Repository.CreateEntry(Directory, name))
            using (var writer = new BinaryWriter(ent.Stream, Encoding.UTF8, leaveOpen: true))
            {
                action(writer);
            }
        }

        /// <summary>
        /// Puts a string into the context pool, and writes the integer code of the string ID
        /// to the write stream. If str is null, this writes -1 and doesn't add it to the pool.
        /// </summary>
        [BestFriend]
        internal void SaveStringOrNull(string str)
        {
            if (str == null)
                Writer.Write(-1);
            else
                Writer.Write(Strings.Add(str).Id);
        }

        /// <summary>
        /// Puts a string into the context pool, and writes the integer code of the string ID
        /// to the write stream. Checks that str is not null.
        /// </summary>
        [BestFriend]
        internal void SaveString(string str)
        {
            _ectx.CheckValue(str, nameof(str));
            Writer.Write(Strings.Add(str).Id);
        }

        [BestFriend]
        internal void SaveString(ReadOnlyMemory<char> str)
        {
            Writer.Write(Strings.Add(str).Id);
        }

        /// <summary>
        /// Puts a string into the context pool, and writes the integer code of the string ID
        /// to the write stream.
        /// </summary>
        [BestFriend]
        internal void SaveNonEmptyString(string str)
        {
            _ectx.CheckParam(!string.IsNullOrEmpty(str), nameof(str));
            Writer.Write(Strings.Add(str).Id);
        }

        [BestFriend]
        internal void SaveNonEmptyString(ReadOnlyMemory<char> str)
        {
            Writer.Write(Strings.Add(str).Id);
        }

        /// <summary>
        /// Commit the save operation. This completes writing of the main stream. When in repository
        /// mode, it disposes <see cref="Writer"/> (but not <see cref="Repository"/>).
        /// </summary>
        [BestFriend]
        internal void Done()
        {
            _ectx.Check(Header.ModelSignature != 0, "ModelSignature not specified!");
            ModelHeader.EndWrite(Writer, FpMin, ref Header, Strings, _loaderAssemblyName);
            Dispose();
        }

        /// <summary>
        /// When in repository mode, this disposes the Writer (but not the repository).
        /// </summary>
        public void Dispose()
        {
            _ectx.Assert((_ent == null) == !InRepository);

            // When in single stream mode, we don't own the Writer.
            if (InRepository)
            {
                Writer.Dispose();
                _ent.Dispose();
            }
        }
    }
}