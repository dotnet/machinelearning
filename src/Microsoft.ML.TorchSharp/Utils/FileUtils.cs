// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TorchSharp.Utils
{
    internal static class FileUtils
    {
        private const int WriteBatchSize = 1024 * 1024;     // this size seems good on performance
        private static readonly Type[] _validTypes = new Type[]
        {
            typeof(short),
            typeof(ushort),
            typeof(int),
            typeof(uint),
            typeof(long),
            typeof(ulong),
            typeof(float),
            typeof(double),
        };

        public static string LoadFromFileOrDownloadFromWeb(string path, string fileName, Uri url, IChannel ch)
        {
            Contracts.AssertNonWhiteSpace(fileName, "Filename can't be empty");

            var contents = "";
            var filePath = Path.Combine(path, fileName);
            if (!File.Exists(filePath))
            {
                try
                {
                    using var webClient = new WebClient();
                    contents = webClient.DownloadString(url);

                }
                catch (WebException e)
                {
                    throw new WebException($"File {fileName} not found and cannot be downloaded from {url}.\n" +
                                           $"Error message: {e.Message}", e);
                }

                try
                {
                    File.WriteAllText(filePath, contents);
                    ch.Info($"File {fileName} successfully downloaded from {url} and saved to {path}.");
                }
                catch (Exception e)
                {
                    ch.Warning($"{DateTime.Now} - WARNING: File {fileName} successfully downloaded from {url}, " +
                                      $"but error occurs when saving file {fileName} into {path}.\n" +
                                      $"Error message: {e.Message}");
                }
            }
            else
            {
                try
                {
                    contents = File.ReadAllText(filePath);
                }
                catch (Exception e)
                {
                    throw new IOException($"Problems met when reading {filePath}.\n" +
                                          $"Error message: {e.Message}", e);
                }
            }

            return contents;
        }

        /// <summary>
        /// Load a continuous segment of bytes from stream and parse them into a number array.
        /// NOTE: this function is only for little-endian storage!
        /// </summary>
        /// <typeparam name="T">should be a numeric type</typeparam>
        /// <param name="stream">the stream to read from its current position</param>
        /// <param name="numElements">expected number of parsed numbers</param>
        /// <param name="tSize">number of bytes occupied by the specified type</param>
        /// <exception cref="NotSupportedException">When the generic type T is not a valid numeric type.</exception>
        /// <exception cref="ArgumentException"/>
        /// <exception cref="InvalidDataException">When the contents in the stream don't match the need.</exception>
        public static IEnumerable<T> LoadNumberArrayFromStream<T>(Stream stream, int numElements, int tSize)
        {
            if (stream == null || !stream.CanRead)
            {
                throw new ArgumentException($"Stream should be non-null and its stream.CanRead property should be true.");
            }
            if (!_validTypes.Contains(typeof(T)))
            {
                throw new NotSupportedException($"Type {typeof(T)} not supported in data loading.");
            }

            var numBytesConsumed = numElements * tSize;
            var byteBuffer = new byte[numBytesConsumed];
            var numBytesRead = stream.Read(byteBuffer, 0, numBytesConsumed);
            if (numBytesConsumed != numBytesRead)
            {
                throw new InvalidDataException(
                    $"The number of bytes read from stream is less than expected. Please check the data files.");
            }

            var targetBuffer = new T[numBytesConsumed / tSize];
            Buffer.BlockCopy(byteBuffer, 0, targetBuffer, 0, numBytesConsumed);
            return targetBuffer;
        }
    }
}
