// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// This class holds an in-memory sample of the text file, and serves as an <see cref="IMultiStreamSource"/> proxy to it.
    /// </summary>
    public sealed class TextFileSample : IMultiStreamSource
    {
        // REVIEW: consider including multiple files via IMultiStreamSource.

        // REVIEW: right now, it expects 0x0A being the trailing character of line break. 
        // Consider a more general implementation.

        private const int BufferSizeMb = 4;
        private const int FirstChunkSizeMb = 1;
        private const int LinesPerChunk = 20;
        private const Double OversamplingRate = 1.1;

        private readonly byte[] _buffer;
        private readonly long? _fullFileSize;
        private readonly long? _approximateRowCount;

        private TextFileSample(byte[] buffer, long? fullFileSize, long? lineCount)
        {
            _buffer = buffer;
            _fullFileSize = fullFileSize;
            _approximateRowCount = lineCount;
        }

        public int Count
        {
            get { return 1; }
        }

        // Full file size, if known, otherwise, null.
        public long? FullFileSize
        {
            get { return _fullFileSize; }
        }

        public int SampleSize
        {
            get { return _buffer.Length; }
        }

        public string GetPathOrNull(int index)
        {
            Contracts.Check(index == 0, "Index must be 0");
            return null;
        }

        public Stream Open(int index)
        {
            Contracts.Check(index == 0, "Index must be 0");
            return new MemoryStream(_buffer);
        }

        public TextReader OpenTextReader(int index)
        {
            return new StreamReader(Open(index));
        }

        public long? ApproximateRowCount => _approximateRowCount;

        /// <summary>
        /// Create a <see cref="TextFileSample"/> by reading multiple chunks from the file (or other source) and 
        /// then stitching them together. The algorithm is as follows:
        /// 0. If the source is not seekable, revert to <see cref="CreateFromHead"/>.
        /// 1. If the file length is less than 2 * <see cref="BufferSizeMb"/>, revert to <see cref="CreateFromHead"/>.
        /// 2. Read first <see cref="FirstChunkSizeMb"/> MB chunk. Determine average line length in the chunk.
        /// 3. Determine how large one chunk should be, and how many chunks there should be, to end up 
        /// with <see cref="BufferSizeMb"/> * <see cref="OversamplingRate"/> MB worth of lines.
        /// 4. Determine seek locations and read the chunks.
        /// 5. Stitch and return a <see cref="TextFileSample"/>.
        /// </summary>
        public static TextFileSample CreateFromFullFile(IHostEnvironment env, string path)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckNonEmpty(path, nameof(path));

            using (var fs = StreamUtils.OpenInStream(path))
            {
                if (!fs.CanSeek)
                    return CreateFromHead(path);
                var fileSize = fs.Length;

                if (fileSize <= 2 * BufferSizeMb * (1 << 20))
                    return CreateFromHead(path);

                var firstChunk = new byte[FirstChunkSizeMb * (1 << 20)];
                int count = fs.Read(firstChunk, 0, firstChunk.Length);
                Contracts.Assert(count == firstChunk.Length);
                if (!IsEncodingOkForSampling(firstChunk))
                    return CreateFromHead(path);
                // REVIEW: CreateFromHead still truncates the file before the last 0x0A byte. For multi-byte encoding, 
                // this might cause an unfinished string to be present in the buffer. Right now this is considered an acceptable
                // price to pay for parse-free processing.

                var lineCount = firstChunk.Count(x => x == '\n');
                if (lineCount == 0)
                    throw Contracts.Except("Counldn't identify line breaks. File is not text?");

                long approximateRowCount = (long)(lineCount * fileSize * 1.0 / firstChunk.Length);
                var firstNewline = Array.FindIndex(firstChunk, x => x == '\n');

                // First line may be header, so we exclude it. The remaining lineCount-1 line breaks are 
                // splitting the text into lineCount lines, and the last line is actually half-size.
                Double averageLineLength = 2.0 * (firstChunk.Length - firstNewline) / (lineCount * 2 - 1);
                averageLineLength = Math.Max(averageLineLength, 3);

                int usefulChunkSize = (int)(averageLineLength * LinesPerChunk);
                int chunkSize = (int)(usefulChunkSize + averageLineLength); // assuming that 1 line worth will be trimmed out

                int chunkCount = (int)Math.Ceiling((BufferSizeMb * OversamplingRate - FirstChunkSizeMb) * (1 << 20) / usefulChunkSize);
                int maxChunkCount = (int)Math.Floor((double)(fileSize - firstChunk.Length) / chunkSize);
                chunkCount = Math.Min(chunkCount, maxChunkCount);

                var chunks = new List<byte[]>();
                chunks.Add(firstChunk);

                // determine the start of each remaining chunk
                long fileSizeRemaining = fileSize - firstChunk.Length - ((long)chunkSize) * chunkCount;
                Contracts.Assert(fileSizeRemaining > 0);

                var rnd = env.Register("TextFileSample").Rand;
                var chunkStartIndices = Enumerable.Range(0, chunkCount)
                    .Select(x => rnd.NextDouble() * fileSizeRemaining)
                    .OrderBy(x => x)
                    .Select((spot, i) => (long)(spot + firstChunk.Length + i * chunkSize))
                    .ToArray();

                foreach (var chunkStartIndex in chunkStartIndices)
                {
                    fs.Seek(chunkStartIndex, SeekOrigin.Begin);
                    byte[] chunk = new byte[chunkSize];
                    int readCount = fs.Read(chunk, 0, chunkSize);
                    Contracts.Assert(readCount > 0);
                    Array.Resize(ref chunk, chunkSize);
                    chunks.Add(chunk);
                }

                return new TextFileSample(StitchChunks(false, chunks.ToArray()), fileSize, approximateRowCount);
            }
        }

        /// <summary>
        /// Create a <see cref="TextFileSample"/> by reading one chunk from the beginning.
        /// </summary>
        public static TextFileSample CreateFromHead(string path)
        {
            Contracts.CheckNonEmpty(path, nameof(path));
            using (var stream = StreamUtils.OpenInStream(path))
            {
                var buf = new byte[BufferSizeMb * (1 << 20)];
                int readCount = stream.Read(buf, 0, buf.Length);
                Array.Resize(ref buf, readCount);
                long? multiplier = stream.CanSeek ? (int?)(stream.Length / buf.Length) : null;
                return new TextFileSample(StitchChunks(readCount == stream.Length, buf),
                    stream.CanSeek ? (long?)stream.Length : null,
                    multiplier.HasValue ? buf.Count(x => x == '\n') * multiplier : null);
            }
        }

        /// <summary>
        /// Given an array of chunks of the text file, of which the first chunk is the head,
        /// this method trims incomplete lines from the beginning and end of each chunk 
        /// (except that it doesn't trim the beginning of the first chunk and end of last chunk if we read whole file),
        /// then joins the rest together to form a final byte buffer and returns a <see cref="TextFileSample"/> 
        /// wrapped around it.
        /// </summary>
        /// <param name="wholeFile">did we read whole file</param>
        /// <param name="chunks">chunks of data</param>
        /// <returns></returns>
        private static byte[] StitchChunks(bool wholeFile, params byte[][] chunks)
        {
            Contracts.AssertValue(chunks);
            Contracts.Assert(chunks.All(x => x != null));

            using (var resultStream = new MemoryStream(BufferSizeMb * (1 << 20)))
            {
                for (int i = 0; i < chunks.Length; i++)
                {
                    int iMin = (i == 0) ? 0 : Array.FindIndex(chunks[i], x => x == '\n') + 1;
                    int iLim = (wholeFile && i == chunks.Length - 1)
                        ? chunks[i].Length
                        : Array.FindLastIndex(chunks[i], x => x == '\n') + 1;

                    if (iLim == 0)
                    {
                        //entire buffer is one string, skip
                        continue;
                    }

                    resultStream.Write(chunks[i], iMin, iLim - iMin);
                }

                var resultBuffer = resultStream.ToArray();
                if (Utils.Size(resultBuffer) == 0)
                    throw Contracts.Except("File is not text, or couldn't detect line breaks");

                return resultBuffer;
            }
        }

        /// <summary>
        /// Detect whether we can auto-detect EOL characters without parsing. 
        /// If we do, we can cheaply sample from different file locations and trim the partial strings.
        /// The encodings that pass the test are UTF8 and all single-byte encodings.
        /// </summary>
        private static bool IsEncodingOkForSampling(byte[] buffer)
        {
            // First check if a BOM/signature exists (sourced from http://www.unicode.org/faq/utf_bom.html#bom4)
            if (buffer.Length >= 4 && buffer[0] == 0x00 && buffer[1] == 0x00 && buffer[2] == 0xFE && buffer[3] == 0xFF)
            {
                // UTF-32, big-endian 
                return false;
            }
            if (buffer.Length >= 4 && buffer[0] == 0xFF && buffer[1] == 0xFE && buffer[2] == 0x00 && buffer[3] == 0x00)
            {
                // UTF-32, little-endian
                return false;
            }
            if (buffer.Length >= 2 && buffer[0] == 0xFE && buffer[1] == 0xFF)
            {
                // UTF-16, big-endian
                return false;
            }
            if (buffer.Length >= 2 && buffer[0] == 0xFF && buffer[1] == 0xFE)
            {
                // UTF-16, little-endian
                return false;
            }
            if (buffer.Length >= 3 && buffer[0] == 0xEF && buffer[1] == 0xBB && buffer[2] == 0xBF)
            {
                // UTF-8
                return true;
            }
            if (buffer.Length >= 3 && buffer[0] == 0x2b && buffer[1] == 0x2f && buffer[2] == 0x76)
            {
                // UTF-7
                return true;
            }

            // No BOM/signature was found, so now we need to 'sniff' the file to see if can manually discover the encoding. 
            int sniffLim = Math.Min(1000, buffer.Length);

            // Some text files are encoded in UTF8, but have no BOM/signature. Hence the below manually checks for a UTF8 pattern. This code is based off
            // the top answer at: http://stackoverflow.com/questions/6555015/check-for-invalid-utf8 .
            int i = 0;
            bool utf8 = false;
            while (i < sniffLim - 4)
            {
                if (buffer[i] <= 0x7F)
                {
                    i += 1;
                    continue;
                }
                if (buffer[i] >= 0xC2 && buffer[i] <= 0xDF && buffer[i + 1] >= 0x80 && buffer[i + 1] < 0xC0)
                {
                    i += 2;
                    utf8 = true;
                    continue;
                }
                if (buffer[i] >= 0xE0 && buffer[i] <= 0xF0 && buffer[i + 1] >= 0x80 && buffer[i + 1] < 0xC0 &&
                    buffer[i + 2] >= 0x80 && buffer[i + 2] < 0xC0)
                {
                    i += 3;
                    utf8 = true;
                    continue;
                }
                if (buffer[i] >= 0xF0 && buffer[i] <= 0xF4 && buffer[i + 1] >= 0x80 && buffer[i + 1] < 0xC0 &&
                    buffer[i + 2] >= 0x80 && buffer[i + 2] < 0xC0 && buffer[i + 3] >= 0x80 && buffer[i + 3] < 0xC0)
                {
                    i += 4;
                    utf8 = true;
                    continue;
                }
                utf8 = false;
                break;
            }
            if (utf8)
                return true;

            if (buffer.Take(sniffLim).Any(x => x == 0))
            {
                // likely a UTF-16 or UTF-32 wuthout a BOM.
                return false;
            }

            // If all else failed, the file is likely in a local 1-byte encoding.
            return true;
        }
    }
}
