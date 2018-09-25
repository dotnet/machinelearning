// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Model
{
    [StructLayout(LayoutKind.Explicit, Size = ModelHeader.Size)]
    public struct ModelHeader
    {
        /// <summary>
        /// This spells 'ML MODEL' with zero replacing space (assuming little endian).
        /// </summary>
        public const ulong SignatureValue = 0x4C45444F4D004C4DUL;
        public const ulong TailSignatureValue = 0x4D4C004D4F44454CUL;

        private const uint VerAssemblyNameSupported = 0x00010002;

        // These are private since they change over time. If we make them public we risk
        // another assembly containing a "copy" of their value when the other assembly
        // was compiled, which might not match the code that can load this.
        //private const uint VerWrittenCur = 0x00010001; // Initial
        private const uint VerWrittenCur = 0x00010002; // Added AssemblyName
        private const uint VerReadableCur = 0x00010002;
        private const uint VerWeCanReadBack = 0x00010001;

        [FieldOffset(0x00)]
        public ulong Signature;
        [FieldOffset(0x08)]
        public uint VerWritten;
        [FieldOffset(0x0C)]
        public uint VerReadable;

        // Location and size (in bytes) of the model block. Note that it is legal for CbModel to be zero.
        [FieldOffset(0x10)]
        public long FpModel;
        [FieldOffset(0x18)]
        public long CbModel;

        // Location and size (in bytes) of the string table block. If there are no strings, these are both zero.
        // If there are n strings then CbStringTable is n * sizeof(long), so is divisible by sizeof(long).
        // Each long is the offset from header.FpStringChars of the "lim" of the characters for that string.
        // The "min" is the "lim" for the previous string. The 0th string's "min" is offset zero.
        [FieldOffset(0x20)]
        public long FpStringTable;
        [FieldOffset(0x28)]
        public long CbStringTable;

        // Location and size (in bytes) of the string characters, without any prefix or termination. The characters
        // unicode (UTF-16).
        [FieldOffset(0x30)]
        public long FpStringChars;
        [FieldOffset(0x38)]
        public long CbStringChars;

        // ModelSignature specifies the format of the model block. These values are assigned by
        // the code that writes the model block.
        [FieldOffset(0x40)]
        public ulong ModelSignature;
        [FieldOffset(0x48)]
        public uint ModelVerWritten;
        [FieldOffset(0x4C)]
        public uint ModelVerReadable;

        // These encode up to two loader signature strings. These are up to 24 ascii characters each.
        [FieldOffset(0x50)]
        public ulong LoaderSignature0;
        [FieldOffset(0x58)]
        public ulong LoaderSignature1;
        [FieldOffset(0x60)]
        public ulong LoaderSignature2;
        [FieldOffset(0x68)]
        public ulong LoaderSignatureAlt0;
        [FieldOffset(0x70)]
        public ulong LoaderSignatureAlt1;
        [FieldOffset(0x78)]
        public ulong LoaderSignatureAlt2;

        // Location of the "tail" signature, which is simply the TailSignatureValue.
        [FieldOffset(0x80)]
        public long FpTail;
        [FieldOffset(0x88)]
        public long FpLim;

        // Location of the fully qualified assembly name string (in UTF-16).
        // Note that it is legal for both to be zero.
        [FieldOffset(0x90)]
        public long FpAssemblyName;
        [FieldOffset(0x98)]
        public uint CbAssemblyName;

        public const int Size = 0x0100;

        // Utilities for writing.

        /// <summary>
        /// Initialize the header and writer for writing. The value of fpMin and header
        /// should be passed to the other utility methods here.
        /// </summary>
        public static void BeginWrite(BinaryWriter writer, out long fpMin, out ModelHeader header)
        {
            Contracts.Assert(Marshal.SizeOf(typeof(ModelHeader)) == Size);
            Contracts.CheckValue(writer, nameof(writer));

            fpMin = writer.FpCur();
            header = default(ModelHeader);
            header.Signature = SignatureValue;
            header.VerWritten = VerWrittenCur;
            header.VerReadable = VerReadableCur;
            header.FpModel = ModelHeader.Size;

            // Write a blank header - the correct information is written by WriteHeaderAndTail.
            byte[] headerBytes = new byte[ModelHeader.Size];
            writer.Write(headerBytes);
            Contracts.CheckIO(writer.FpCur() == fpMin + ModelHeader.Size);
        }

        /// <summary>
        /// The current writer position should be the end of the model blob. Records the model size, writes the string table,
        /// completes and writes the header, and writes the tail.
        /// </summary>
        public static void EndWrite(BinaryWriter writer, long fpMin, ref ModelHeader header, NormStr.Pool pool = null, string loaderAssemblyName = null)
        {
            Contracts.CheckValue(writer, nameof(writer));
            Contracts.CheckParam(fpMin >= 0, nameof(fpMin));
            Contracts.CheckValueOrNull(pool);

            // Record the model size.
            EndModelCore(writer, fpMin, ref header);

            Contracts.Check(header.FpStringTable == 0);
            Contracts.Check(header.CbStringTable == 0);
            Contracts.Check(header.FpStringChars == 0);
            Contracts.Check(header.CbStringChars == 0);

            // Write the strings.
            if (pool != null && pool.Count > 0)
            {
                header.FpStringTable = writer.FpCur() - fpMin;
                long offset = 0;
                int cv = 0;
                // REVIEW: Implement an indexer on pool!
                foreach (var ns in pool)
                {
                    Contracts.Assert(ns.Id == cv);
                    offset += ns.Value.Length * sizeof(char);
                    writer.Write(offset);
                    cv++;
                }
                Contracts.Assert(cv == pool.Count);
                header.CbStringTable = pool.Count * sizeof(long);
                header.FpStringChars = writer.FpCur() - fpMin;
                Contracts.Assert(header.FpStringChars == header.FpStringTable + header.CbStringTable);
                foreach (var ns in pool)
                {
                    foreach (var ch in ns.Value.Span)
                        writer.Write((short)ch);
                }
                header.CbStringChars = writer.FpCur() - header.FpStringChars - fpMin;
                Contracts.Assert(offset == header.CbStringChars);
            }

            WriteLoaderAssemblyName(writer, fpMin, ref header, loaderAssemblyName);

            WriteHeaderAndTailCore(writer, fpMin, ref header);
        }

        private static void WriteLoaderAssemblyName(BinaryWriter writer, long fpMin, ref ModelHeader header, string loaderAssemblyName)
        {
            if (!string.IsNullOrEmpty(loaderAssemblyName))
            {
                header.FpAssemblyName = writer.FpCur() - fpMin;
                header.CbAssemblyName = (uint)loaderAssemblyName.Length * sizeof(char);

                foreach (var ch in loaderAssemblyName)
                    writer.Write((short)ch);
            }
            else
            {
                header.FpAssemblyName = 0;
                header.CbAssemblyName = 0;
            }
        }

        /// <summary>
        /// The current writer position should be where the tail belongs. Writes the header and tail.
        /// Typically this isn't called directly unless you are doing custom string table serialization.
        /// In that case you should have called EndModelCore before writing the string table information.
        /// </summary>
        public static void WriteHeaderAndTailCore(BinaryWriter writer, long fpMin, ref ModelHeader header)
        {
            Contracts.CheckValue(writer, nameof(writer));
            Contracts.CheckParam(fpMin >= 0, nameof(fpMin));

            header.FpTail = writer.FpCur() - fpMin;
            writer.Write(TailSignatureValue);
            header.FpLim = writer.FpCur() - fpMin;

            Exception ex;
            bool res = TryValidate(ref header, header.FpLim, out ex);
            // If this fails, we didn't construct the header correctly. This is both a bug and
            // something we want to protect against at runtime, hence both assert and check.
            Contracts.Assert(res);
            Contracts.Check(res);

            // Write the header, then seek back to the end.
            writer.Seek(fpMin);
            byte[] headerBytes = new byte[ModelHeader.Size];
            MarshalToBytes(ref header, headerBytes);
            writer.Write(headerBytes);
            Contracts.Assert(writer.FpCur() == fpMin + ModelHeader.Size);
            writer.Seek(header.FpLim + fpMin);
        }

        /// <summary>
        /// The current writer position should be the end of the model blob. Records the size of the model blob.
        /// Typically this isn't called directly unless you are doing custom string table serialization.
        /// </summary>
        public static void EndModelCore(BinaryWriter writer, long fpMin, ref ModelHeader header)
        {
            Contracts.Check(header.FpModel == ModelHeader.Size);
            Contracts.Check(header.CbModel == 0);

            long fpCur = writer.FpCur();
            Contracts.Check(fpCur - fpMin >= header.FpModel);

            // Record the size of the model.
            header.CbModel = fpCur - header.FpModel - fpMin;
        }

        /// <summary>
        /// Sets the version information the header.
        /// </summary>
        public static void SetVersionInfo(ref ModelHeader header, VersionInfo ver)
        {
            header.ModelSignature = ver.ModelSignature;
            header.ModelVerWritten = ver.VerWrittenCur;
            header.ModelVerReadable = ver.VerReadableCur;
            SetLoaderSig(ref header, ver.LoaderSignature);
            SetLoaderSigAlt(ref header, ver.LoaderSignatureAlt);
        }

        /// <summary>
        /// Record the given loader sig in the header. If sig is null, clears the loader sig.
        /// </summary>
        public static void SetLoaderSig(ref ModelHeader header, string sig)
        {
            header.LoaderSignature0 = 0;
            header.LoaderSignature1 = 0;
            header.LoaderSignature2 = 0;

            if (sig == null)
                return;

            Contracts.Check(sig.Length <= 24);
            for (int ich = 0; ich < sig.Length; ich++)
            {
                char ch = sig[ich];
                Contracts.Check(ch <= 0xFF);
                if (ich < 8)
                    header.LoaderSignature0 |= (ulong)ch << (ich * 8);
                else if (ich < 16)
                    header.LoaderSignature1 |= (ulong)ch << ((ich - 8) * 8);
                else if (ich < 24)
                    header.LoaderSignature2 |= (ulong)ch << ((ich - 16) * 8);
            }
        }

        /// <summary>
        /// Record the given alternate loader sig in the header. If sig is null, clears the alternate loader sig.
        /// </summary>
        public static void SetLoaderSigAlt(ref ModelHeader header, string sig)
        {
            header.LoaderSignatureAlt0 = 0;
            header.LoaderSignatureAlt1 = 0;
            header.LoaderSignatureAlt2 = 0;

            if (sig == null)
                return;

            Contracts.Check(sig.Length <= 24);
            for (int ich = 0; ich < sig.Length; ich++)
            {
                char ch = sig[ich];
                Contracts.Check(ch <= 0xFF);
                if (ich < 8)
                    header.LoaderSignatureAlt0 |= (ulong)ch << (ich * 8);
                else if (ich < 16)
                    header.LoaderSignatureAlt1 |= (ulong)ch << ((ich - 8) * 8);
                else if (ich < 24)
                    header.LoaderSignatureAlt2 |= (ulong)ch << ((ich - 16) * 8);
            }
        }

        /// <summary>
        /// Low level method for copying bytes from a header structure into a byte array.
        /// </summary>
        public static void MarshalToBytes(ref ModelHeader header, byte[] bytes)
        {
            Contracts.Check(Utils.Size(bytes) >= Size);
            unsafe
            {
                fixed (ModelHeader* pheader = &header)
                    Marshal.Copy((IntPtr)pheader, bytes, 0, Size);
            }
        }

        // Utilities for reading.

        /// <summary>
        /// Read the model header, strings, etc from reader. Also validates the header (throws if bad).
        /// Leaves the reader position at the beginning of the model blob.
        /// </summary>
        public static void BeginRead(out long fpMin, out ModelHeader header, out string[] strings, out string loaderAssemblyName, BinaryReader reader)
        {
            fpMin = reader.FpCur();

            byte[] headerBytes = reader.ReadBytes(ModelHeader.Size);
            Contracts.CheckDecode(headerBytes.Length == ModelHeader.Size);
            ModelHeader.MarshalFromBytes(out header, headerBytes);

            Exception ex;
            if (!ModelHeader.TryValidate(ref header, reader, fpMin, out strings, out loaderAssemblyName, out ex))
                throw ex;

            reader.Seek(header.FpModel + fpMin);
        }

        /// <summary>
        /// Finish reading. Checks that the current reader position is the end of the model blob.
        /// Seeks to the end of the entire model file (after the tail).
        /// </summary>
        public static void EndRead(long fpMin, ref ModelHeader header, BinaryReader reader)
        {
            Contracts.CheckDecode(header.FpModel + header.CbModel == reader.FpCur() - fpMin);
            reader.Seek(header.FpLim + fpMin);
        }

        /// <summary>
        /// Performs standard version validation.
        /// </summary>
        public static void CheckVersionInfo(ref ModelHeader header, VersionInfo ver)
        {
            Contracts.CheckDecode(header.ModelSignature == ver.ModelSignature, "Unknown file type");
            Contracts.CheckDecode(header.ModelVerReadable <= header.ModelVerWritten, "Corrupt file header");
            if (header.ModelVerReadable > ver.VerWrittenCur)
                throw Contracts.ExceptDecode("Cause: TLC {0} cannont read component '{1}' of the model, because the model is too new.\n" +
                                "Suggestion: Make sure the model is trained with TLC {0} or older.\n" +
                                "Debug details: Maximum expected version {2}, got {3}.",
                                typeof(VersionInfo).Assembly.GetName().Version, ver.LoaderSignature, header.ModelVerReadable, ver.VerWrittenCur);
            if (header.ModelVerWritten < ver.VerWeCanReadBack)
            {
                // Breaking backwards compatibility is something we should avoid if at all possible. If
                // this message is observed, it may be a bug.
                throw Contracts.ExceptDecode("Cause: TLC {0} cannot read component '{1}' of the model, because the model is too old.\n" +
                                  "Suggestion: Make sure the model is trained with TLC {0}.\n" +
                                  "Debug details: Minimum expected version {2}, got {3}.",
                                  typeof(VersionInfo).Assembly.GetName().Version, ver.LoaderSignature, header.ModelVerReadable, ver.VerWrittenCur);
            }
        }

        /// <summary>
        /// Low level method for copying bytes from a byte array to a header structure.
        /// </summary>
        public static void MarshalFromBytes(out ModelHeader header, byte[] bytes)
        {
            Contracts.Check(Utils.Size(bytes) >= Size);
            unsafe
            {
                fixed (ModelHeader* pheader = &header)
                    Marshal.Copy(bytes, 0, (IntPtr)pheader, Size);
            }
        }

        /// <summary>
        /// Checks the basic validity of the header, assuming the stream is at least the given size.
        /// Returns false (and the out exception) on failure.
        /// </summary>
        public static bool TryValidate(ref ModelHeader header, long size, out Exception ex)
        {
            Contracts.Check(size >= 0);

            try
            {
                Contracts.CheckDecode(header.Signature == SignatureValue, "Wrong file type");
                Contracts.CheckDecode(header.VerReadable <= header.VerWritten, "Corrupt file header");
                Contracts.CheckDecode(header.VerReadable <= VerWrittenCur, "File is too new");
                Contracts.CheckDecode(header.VerWritten >= VerWeCanReadBack, "File is too old");

                // Currently the model always comes immediately after the header.
                Contracts.CheckDecode(header.FpModel == Size);
                Contracts.CheckDecode(header.FpModel + header.CbModel >= header.FpModel);

                if (header.FpStringTable == 0)
                {
                    // No strings.
                    Contracts.CheckDecode(header.CbStringTable == 0);
                    Contracts.CheckDecode(header.FpStringChars == 0);
                    Contracts.CheckDecode(header.CbStringChars == 0);
                    if (header.VerWritten < VerAssemblyNameSupported || header.FpAssemblyName == 0)
                    {
                        Contracts.CheckDecode(header.FpTail == header.FpModel + header.CbModel);
                    }
                }
                else
                {
                    // Currently the string table always comes immediately after the model block.
                    Contracts.CheckDecode(header.FpStringTable == header.FpModel + header.CbModel);
                    Contracts.CheckDecode(header.CbStringTable % sizeof(long) == 0);
                    Contracts.CheckDecode(header.CbStringTable / sizeof(long) < int.MaxValue);
                    Contracts.CheckDecode(header.FpStringTable + header.CbStringTable > header.FpStringTable);
                    Contracts.CheckDecode(header.FpStringChars == header.FpStringTable + header.CbStringTable);
                    Contracts.CheckDecode(header.CbStringChars % sizeof(char) == 0);
                    Contracts.CheckDecode(header.FpStringChars + header.CbStringChars >= header.FpStringChars);
                    if (header.VerWritten < VerAssemblyNameSupported || header.FpAssemblyName == 0)
                    {
                        Contracts.CheckDecode(header.FpTail == header.FpStringChars + header.CbStringChars);
                    }
                }

                if (header.VerWritten >= VerAssemblyNameSupported)
                {
                    if (header.FpAssemblyName == 0)
                    {
                        Contracts.CheckDecode(header.CbAssemblyName == 0);
                    }
                    else
                    {
                        // the assembly name always immediately after the string table, if there is one
                        if (header.FpStringTable == 0)
                        {
                            Contracts.CheckDecode(header.FpAssemblyName == header.FpModel + header.CbModel);
                        }
                        else
                        {
                            Contracts.CheckDecode(header.FpAssemblyName == header.FpStringChars + header.CbStringChars);
                        }
                        Contracts.CheckDecode(header.CbAssemblyName % sizeof(char) == 0);
                        Contracts.CheckDecode(header.FpTail == header.FpAssemblyName + header.CbAssemblyName);
                    }
                }

                Contracts.CheckDecode(header.FpLim == header.FpTail + sizeof(ulong));
                Contracts.CheckDecode(size == 0 || size >= header.FpLim);

                ex = null;
                return true;
            }
            catch (Exception e)
            {
                ex = e;
                return false;
            }
        }

        /// <summary>
        /// Checks the validity of the header, reads the string table, etc.
        /// </summary>
        public static bool TryValidate(ref ModelHeader header, BinaryReader reader, long fpMin, out string[] strings, out string loaderAssemblyName, out Exception ex)
        {
            Contracts.CheckValue(reader, nameof(reader));
            Contracts.Check(fpMin >= 0);

            if (!TryValidate(ref header, reader.BaseStream.Length - fpMin, out ex))
            {
                strings = null;
                loaderAssemblyName = null;
                return false;
            }

            try
            {
                long fpOrig = reader.FpCur();

                StringBuilder sb = null;
                if (header.FpStringTable == 0)
                {
                    // No strings.
                    strings = null;
                }
                else
                {
                    reader.Seek(header.FpStringTable + fpMin);
                    Contracts.Assert(reader.FpCur() == header.FpStringTable + fpMin);

                    long cstr = header.CbStringTable / sizeof(long);
                    Contracts.Assert(cstr < int.MaxValue);
                    long[] offsets = reader.ReadLongArray((int)cstr);
                    Contracts.Assert(header.FpStringChars == reader.FpCur() - fpMin);
                    Contracts.CheckDecode(offsets[cstr - 1] == header.CbStringChars);

                    strings = new string[cstr];
                    long offset = 0;
                    sb = new StringBuilder();
                    for (int i = 0; i < offsets.Length; i++)
                    {
                        Contracts.CheckDecode(header.FpStringChars + offset == reader.FpCur() - fpMin);

                        long offsetPrev = offset;
                        offset = offsets[i];
                        Contracts.CheckDecode(offsetPrev <= offset & offset <= header.CbStringChars);
                        Contracts.CheckDecode(offset % sizeof(char) == 0);
                        long cch = (offset - offsetPrev) / sizeof(char);
                        Contracts.CheckDecode(cch < int.MaxValue);

                        sb.Clear();
                        for (long ich = 0; ich < cch; ich++)
                            sb.Append((char)reader.ReadUInt16());
                        strings[i] = sb.ToString();
                    }
                    Contracts.CheckDecode(offset == header.CbStringChars);
                    Contracts.CheckDecode(header.FpStringChars + header.CbStringChars == reader.FpCur() - fpMin);
                }

                if (header.VerWritten >= VerAssemblyNameSupported && header.FpAssemblyName != 0)
                {
                    reader.Seek(header.FpAssemblyName + fpMin);
                    int assemblyNameLength = (int)header.CbAssemblyName / sizeof(char);

                    sb = sb != null ? sb.Clear() : new StringBuilder(assemblyNameLength);

                    for (long ich = 0; ich < assemblyNameLength; ich++)
                        sb.Append((char)reader.ReadUInt16());

                    loaderAssemblyName = sb.ToString();
                }
                else
                {
                    loaderAssemblyName = null;
                }

                Contracts.CheckDecode(header.FpTail == reader.FpCur() - fpMin);

                ulong tail = reader.ReadUInt64();
                Contracts.CheckDecode(tail == TailSignatureValue, "Corrupt model file tail");

                ex = null;

                reader.Seek(fpOrig);
                return true;
            }
            catch (Exception e)
            {
                strings = null;
                loaderAssemblyName = null;
                ex = e;
                return false;
            }
        }

        /// <summary>
        /// Extract and return the loader sig from the header, trimming trailing zeros.
        /// </summary>
        public static string GetLoaderSig(ref ModelHeader header)
        {
            char[] chars = new char[3 * sizeof(ulong)];

            for (int ich = 0; ich < chars.Length; ich++)
            {
                char ch;
                if (ich < 8)
                    ch = (char)((header.LoaderSignature0 >> (ich * 8)) & 0xFF);
                else if (ich < 16)
                    ch = (char)((header.LoaderSignature1 >> ((ich - 8) * 8)) & 0xFF);
                else
                    ch = (char)((header.LoaderSignature2 >> ((ich - 16) * 8)) & 0xFF);
                chars[ich] = ch;
            }

            int cch = 24;
            while (cch > 0 && chars[cch - 1] == 0)
                cch--;
            return new string(chars, 0, cch);
        }

        /// <summary>
        /// Extract and return the alternate loader sig from the header, trimming trailing zeros.
        /// </summary>
        public static string GetLoaderSigAlt(ref ModelHeader header)
        {
            char[] chars = new char[3 * sizeof(ulong)];

            for (int ich = 0; ich < chars.Length; ich++)
            {
                char ch;
                if (ich < 8)
                    ch = (char)((header.LoaderSignatureAlt0 >> (ich * 8)) & 0xFF);
                else if (ich < 16)
                    ch = (char)((header.LoaderSignatureAlt1 >> ((ich - 8) * 8)) & 0xFF);
                else
                    ch = (char)((header.LoaderSignatureAlt2 >> ((ich - 16) * 8)) & 0xFF);
                chars[ich] = ch;
            }

            int cch = 24;
            while (cch > 0 && chars[cch - 1] == 0)
                cch--;
            return new string(chars, 0, cch);
        }
    }

    /// <summary>
    /// This is used to simplify version checking boiler-plate code. It is an optional
    /// utility type.
    /// </summary>
    public readonly struct VersionInfo
    {
        public readonly ulong ModelSignature;
        public readonly uint VerWrittenCur;
        public readonly uint VerReadableCur;
        public readonly uint VerWeCanReadBack;
        public readonly string LoaderAssemblyName;
        public readonly string LoaderSignature;
        public readonly string LoaderSignatureAlt;

        /// <summary>
        /// Construct version info with a string value for modelSignature. The string must be 8 characters
        /// all less than 0x100. Spaces are mapped to zero. This assumes little-endian.
        /// </summary>
        public VersionInfo(string modelSignature, uint verWrittenCur, uint verReadableCur, uint verWeCanReadBack,
            string loaderAssemblyName, string loaderSignature = null, string loaderSignatureAlt = null)
        {
            Contracts.Check(Utils.Size(modelSignature) == 8, "Model signature must be eight characters");
            ModelSignature = 0;
            for (int ich = 0; ich < modelSignature.Length; ich++)
            {
                char ch = modelSignature[ich];
                Contracts.Check(ch <= 0xFF);
                // Map space to zero.
                if (ch != ' ')
                    ModelSignature |= (ulong)ch << (ich * 8);
            }

            VerWrittenCur = verWrittenCur;
            VerReadableCur = verReadableCur;
            VerWeCanReadBack = verWeCanReadBack;
            LoaderAssemblyName = loaderAssemblyName;
            LoaderSignature = loaderSignature;
            LoaderSignatureAlt = loaderSignatureAlt;
        }
    }
}
