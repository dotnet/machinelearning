// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Security.Cryptography;
using System.Text;
namespace Microsoft.DotNet.Cli.Telemetry
{
    internal static class Sha256Hasher
    {
        /// <summary>
        /// The hashed mac address needs to be the same hashed value as produced by the other distinct sources given the same input. (e.g. VsCode)
        /// </summary>
        public static string Hash(string text)
        {
            var sha256 = SHA256.Create();
            return HashInFormat(sha256, text);
        }

        public static string HashWithNormalizedCasing(string text)
        {
            return Hash(text.ToUpperInvariant());
        }

        private static string HashInFormat(SHA256 sha256, string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            byte[] hash = sha256.ComputeHash(bytes);
            StringBuilder hashString = new StringBuilder();
            foreach (byte x in hash)
            {
                hashString.AppendFormat("{0:x2}", x);
            }
            return hashString.ToString();
        }
    }
}