// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static class MemUtils
    {
        // The signature of this method is intentionally identical to
        // .Net 4.6's Buffer.MemoryCopy.
        // REVIEW: Remove once we're on a version of .NET which includes
        // Buffer.MemoryCopy.
        public unsafe static void MemoryCopy(void* source, void* destination, long destinationSizeInBytes, long sourceBytesToCopy)
        {
            // MemCpy has undefined behavior when handed overlapping source and
            // destination buffers.
            // Do not pass it overlapping source and destination buffers.
            Contracts.Check((byte*)destination + sourceBytesToCopy <= source || destination >= (byte*)source + sourceBytesToCopy);
            Contracts.Check(destinationSizeInBytes >= sourceBytesToCopy);
#if CORECLR
            System.Buffer.MemoryCopy(source, destination, destinationSizeInBytes, sourceBytesToCopy);
#else
            Thunk.MemCpy(destination, source, sourceBytesToCopy);
#endif
        }
    }
}
