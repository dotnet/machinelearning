// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Data.IO
{
    /// <summary>
    /// This structure is utilized by both the binary loader and binary saver to hold
    /// information on the location of blocks written to an .IDV binary file.
    /// </summary>
    internal readonly struct BlockLookup
    {
        /// <summary>The offset of the block into the file.</summary>
        public readonly long BlockOffset;
        /// <summary>The byte length of the block on disk.</summary>
        public readonly int BlockLength;
        /// <summary>The byte length of the block if decompressed.</summary>
        public readonly int DecompressedBlockLength;

        public BlockLookup(long blockOffset, int blockLength, int decompressedBlockLength)
        {
            BlockOffset = blockOffset;
            BlockLength = blockLength;
            DecompressedBlockLength = decompressedBlockLength;
        }
    }
}
