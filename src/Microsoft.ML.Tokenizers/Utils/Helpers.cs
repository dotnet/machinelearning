// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.IO.Compression;
using System.Buffers;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    internal static partial class Helpers
    {
        internal static void ArrayPoolGrow<T>(ref T[] arrayPoolArray, int requiredCapacity)
        {
            T[] tmp = ArrayPool<T>.Shared.Rent(Math.Max(arrayPoolArray.Length * 2, requiredCapacity));
            arrayPoolArray.CopyTo(tmp.AsSpan());
            ArrayPool<T>.Shared.Return(arrayPoolArray);
            arrayPoolArray = tmp;
        }

        internal static async Task<Stream> OpenEmbeddedCompressedStreamAsync(string resourceName, CancellationToken cancellationToken = default(CancellationToken))
        {
            using Stream compressedStream = typeof(Tokenizer).Assembly.GetManifestResourceStream(resourceName)!;
            using DeflateStream deflateStream = new DeflateStream(compressedStream, CompressionMode.Decompress);
            MemoryStream memoryStream = new MemoryStream();
            await deflateStream.CopyToAsync(memoryStream, bufferSize: 81920, cancellationToken).ConfigureAwait(false);
            memoryStream.Position = 0;
            return memoryStream;
        }
    }
}
