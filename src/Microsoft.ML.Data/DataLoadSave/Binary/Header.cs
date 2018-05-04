// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.Data.IO
{
    [StructLayout(LayoutKind.Explicit, Size = HeaderSize)]
    public struct Header
    {
        /// <summary>
        /// The fixed header size. This should not be changed even in future versions of the format.
        /// </summary>
        public const int HeaderSize = 256;

        /// <summary>
        /// The header must start with this signature. This number will
        /// appear as the eight-byte sequence "CML\0DVB\0" if encoded in
        /// little-endian. (CML DVB is meant to suggest CloudML DataView binary).
        /// </summary>
        public const ulong SignatureValue = 0x00425644004C4D43;

        /// <summary>
        /// The file must end with this value. Is is simply the
        /// byte-order-reversed version of the head signature.
        /// </summary>
        public const ulong TailSignatureValue = 0x434D4C0044564200;

        /// <summary>
        /// The current version of the format this software can write.
        /// </summary>
        //public const ulong WriterVersion = 0x0001000100010001; // This first version of the format, not publically released.
        //public const ulong WriterVersion = 0x0001000100010002; // Codec changes.
        //public const ulong WriterVersion = 0x0001000100010003; // Slot names.
        //public const ulong WriterVersion = 0x0001000100010004; // Column metadata.
        public const ulong WriterVersion = 0x0001000100010005; // "NA" DvText support.
        public const ulong CanBeReadByVersion = 0x0001000100010005;

        internal static string VersionToString(ulong v)
        {
            return string.Format("{0}.{1}.{2}.{3}",
                (v >> 48) & 0xffff, (v >> 32) & 0xffff,
                (v >> 16) & 0xffff, v & 0xffff);
        }

        /// <summary>
        /// The magic number of this file.
        /// </summary>
        [FieldOffset(0)]
        public ulong Signature;

        /// <summary>
        /// Indicates the version of the data file.
        /// </summary>
        [FieldOffset(8)]
        public ulong Version;

        /// <summary>
        /// Indicates the minimum reader version that can interpret this file, possibly
        /// with some data loss.
        /// </summary>
        [FieldOffset(16)]
        public ulong CompatibleVersion;

        /// <summary>
        /// The offset to the table of contents structure where the column type definitions
        /// are stored.
        /// </summary>
        [FieldOffset(24)]
        public long TableOfContentsOffset;

        /// <summary>
        /// The eight-byte tail signature starts at this offset. So, the entire dataset
        /// stream should be considered to have eight plus this value bytes.
        /// </summary>
        [FieldOffset(32)]
        public long TailOffset;

        /// <summary>
        /// The number of rows in this data file.
        /// </summary>
        [FieldOffset(40)]
        public long RowCount;

        /// <summary>
        /// The number of columns in this data file.
        /// </summary>
        [FieldOffset(48)]
        public int ColumnCount;

        // Lots of padding (up to size 256)....
    }
}
