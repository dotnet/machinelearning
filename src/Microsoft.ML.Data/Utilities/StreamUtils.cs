// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Internal.Utilities
{
    // REVIEW: Implement properly on CoreCLR.
    [BestFriend]
    internal static class StreamUtils
    {
        public static Stream OpenInStream(string fileName)
        {
#if !CORECLR
            return Microsoft.ML.Internal.IO.ZStreamIn.Open(fileName);
#else
            return new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
#endif
        }

        public static StreamWriter CreateWriter(string fileName)
        {
#if !CORECLR
            return Internal.IO.ZStreamWriter.Open(fileName);
#else
            // REVIEW: Should specify the encoding.
            return new StreamWriter(new FileStream(fileName, FileMode.Create, FileAccess.Write));
#endif
        }

        public static string[] ExpandWildCards(string pattern)
        {
#if !CORECLR
            return Internal.IO.IOUtil.ExpandWildcards(pattern);
#else
            return Expand(pattern);
#endif
        }

#if CORECLR
        private static readonly char[] _wildChars = new char[] { '*', '?' };
        private static readonly char[] _wildPlusChars = new char[] { '*', '?', '+' };

        /// <summary>
        /// Expand an extended wildcard pattern into a set of file paths.
        /// </summary>
        /// <param name="pattern">the pattern to expand</param>
        /// <returns>the set of file paths matching the pattern</returns>
        /// <remarks>
        /// The wildcard pattern accepts the standard "*" and "?" placeholders.
        /// "..." also refers to a recursive search over subdirectories.
        /// "+" can also be used to make a union of several filenames or patterns.
        /// Names of files that do not exist will be excluded.
        /// </remarks>
        private static string[] Expand(string pattern)
        {
            if (pattern == null || (pattern.IndexOfAny(_wildPlusChars) < 0 && pattern.IndexOf("...") < 0))
            {
                if (File.Exists(pattern))
                    return new string[] { pattern };
                else
                    return new string[0];
            }
            var patterns = pattern.Split('+');
            var matchList = new List<string>();
            foreach (var currentPattern in patterns)
            {
                // hard-code in special types??
                if (currentPattern.Length == 0)
                    continue;
                var patLower = currentPattern.ToLower();
                if (currentPattern.IndexOfAny(_wildChars) >= 0 || currentPattern.IndexOf("...") >= 0)
                {
                    // compressed extensions are not automatically used! ***
                    int recursiveIndex = currentPattern.IndexOf("...");
                    if (recursiveIndex >= 0)
                    {
                        var left = currentPattern.Substring(0, recursiveIndex);
                        var right = currentPattern.Substring(recursiveIndex + 3);
                        right = right.TrimStart('\\', '/');
                        if (right.Length == 0)
                            right = "*";
                        var path = left;
                        var pathEmpty = (path == null || path.Length == 0);
                        if (pathEmpty)
                            path = ".";
                        var dirsLeft = new Stack<string>();
                        dirsLeft.Push(path);
                        while (dirsLeft.Count != 0)
                        {
                            var dir = dirsLeft.Pop();
                            // watch for lack of access:
                            try
                            {
                                // this is actually incorrect, for 3-char extensions: ***
                                var files = Directory.GetFiles(dir, right).OrderBy(f => f).ToArray();
                                if (pathEmpty)
                                {
                                    for (int i = 0; i < files.Length; i++)
                                    {
                                        if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                            files[i] = files[i].Substring(2);
                                    }
                                }
                                matchList.AddRange(files);
                                var subs = Directory.GetDirectories(dir).OrderBy(f => f).ToArray();
                                for (var i = subs.Length - 1; i >= 0; i--)
                                    dirsLeft.Push(subs[i]);
                            }
                            catch
                            {
                                // ignore
                            }
                        }
                    }
                    else
                    {
                        try
                        {
                            var path = Path.GetDirectoryName(currentPattern);
                            var pathEmpty = !(currentPattern.StartsWith("./") || currentPattern.StartsWith(".\\"));
                            if (path == null || path.Length == 0)
                                path = ".";
                            // watch for lack of access:
                            try
                            {
                                var files = Directory.GetFiles(path, Path.GetFileName(currentPattern)).OrderBy(f => f).ToArray();
                                if (pathEmpty)
                                {
                                    for (int i = 0; i < files.Length; i++)
                                    {
                                        if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                            files[i] = files[i].Substring(2);
                                    }
                                }
                                matchList.AddRange(files);
                            }
                            catch
                            {
                                // ignore
                            }
                        }
                        catch
                        {
                            // ignore bad path?
                        }
                    }
                }
                else
                {
                    // what to do?? Filter to only those that exist?? ***
                    if (!File.Exists(currentPattern))
                        continue;
                    matchList.Add(currentPattern);
                }
            }

            // remove duplicates, very inefficiently - but it is simple, preserves
            // the order, uses no additional memory, and is case-insensitive...:
            for (var i = 0; i < matchList.Count - 1; i++)
            {
                for (var j = i + 1; j < matchList.Count; j++)
                {
                    if (string.Compare(matchList[i], matchList[j], true) == 0)
                    {
                        matchList.RemoveAt(j);
                        j--;
                    }
                }
            }
            return matchList.ToArray();
        }
#endif

        /// <summary>Validates arguments provided to reading and writing methods on <see cref="Stream"/>.</summary>
        /// <param name="buffer">The array "buffer" argument passed to the reading or writing method.</param>
        /// <param name="offset">The integer "offset" argument passed to the reading or writing method.</param>
        /// <param name="count">The integer "count" argument passed to the reading or writing method.</param>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> was null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// <paramref name="offset"/> was outside the bounds of <paramref name="buffer"/>, or
        /// <paramref name="count"/> was negative, or the range specified by the combination of
        /// <paramref name="offset"/> and <paramref name="count"/> exceed the length of <paramref name="buffer"/>.
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateBufferArguments(byte[] buffer, int offset, int count)
        {
            if (buffer is null)
            {
                throw new ArgumentNullException(nameof(buffer));
            }

            if (offset < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(offset), "Offset must be non-negative.");
            }

            if ((uint)count > buffer.Length - offset)
            {
                throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative and count must not exceed buffer.Length - offset.");
            }
        }

        // No argument checking is done here. It is up to the caller.
        private static int ReadAtLeastCore(Stream stream, byte[] buffer, int offset, int minimumBytes, bool throwOnEndOfStream)
        {
            Debug.Assert(minimumBytes <= buffer.Length);
            int count = minimumBytes;
            int totalRead = 0;
            while (totalRead < minimumBytes)
            {
                int read = stream.Read(buffer, offset, count);
                offset += read;
                count -= read;
                if (read == 0)
                {
                    if (throwOnEndOfStream)
                    {
                        throw new EndOfStreamException("Unable to read beyond the end of the stream.");
                    }

                    return totalRead;
                }

                totalRead += read;
            }

            return totalRead;
        }

        /// <summary>
        /// Reads <paramref name="count"/> number of bytes from the current stream and advances the position within the stream.
        /// </summary>
        /// <param name="stream"></param>
        /// <param name="buffer">
        /// An array of bytes. When this method returns, the buffer contains the specified byte array with the values
        /// between <paramref name="offset"/> and (<paramref name="offset"/> + <paramref name="count"/> - 1) replaced
        /// by the bytes read from the current stream.
        /// </param>
        /// <param name="offset">The byte offset in <paramref name="buffer"/> at which to begin storing the data read from the current stream.</param>
        /// <param name="count">The number of bytes to be read from the current stream.</param>
        /// <exception cref="ArgumentNullException"><paramref name="buffer"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// <paramref name="offset"/> is outside the bounds of <paramref name="buffer"/>.
        /// -or-
        /// <paramref name="count"/> is negative.
        /// -or-
        /// The range specified by the combination of <paramref name="offset"/> and <paramref name="count"/> exceeds the
        /// length of <paramref name="buffer"/>.
        /// </exception>
        /// <exception cref="EndOfStreamException">
        /// The end of the stream is reached before reading <paramref name="count"/> number of bytes.
        /// </exception>
        /// <remarks>
        /// When <paramref name="count"/> is 0 (zero), this read operation will be completed without waiting for available data in the stream.
        /// </remarks>
        public static void ReadExactly(this Stream stream, byte[] buffer, int offset, int count)
        {
            ValidateBufferArguments(buffer, offset, count);
            _ = ReadAtLeastCore(stream, buffer, offset, count, throwOnEndOfStream: true);
        }
    }
}
