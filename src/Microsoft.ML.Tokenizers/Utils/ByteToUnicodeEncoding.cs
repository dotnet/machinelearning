// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Map between utf-8 byte to unicode with avoiding mapping to whitespace/control characters.
    /// </summary>
    internal sealed class ByteToUnicodeEncoding
    {
        public static ByteToUnicodeEncoding Instance { get; } = new ByteToUnicodeEncoding();

        public ByteToUnicodeEncoding()
        {
            var byteToUnicodeMapping = Enumerable.Range('!', '~' - '!' + 1)
                .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
                .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
                .ToDictionary(b => (char)b, b => (char)b);

            const int numChars = 256;
            var n = 0;
            foreach (var b in Enumerable.Range(0, numChars))
            {
                if (!byteToUnicodeMapping.ContainsKey((char)b))
                {
                    byteToUnicodeMapping.Add((char)b, (char)(numChars + n));
                    ++n;
                }
            }

            ByteToUnicode = byteToUnicodeMapping;
            UnicodeToByte = ByteToUnicode.ToDictionary(kv => kv.Value, kv => kv.Key);

            int count = numChars + n;

            CharToString = new string[count];
            for (char c = (char)0; c < (char)count; c++)
            {
                CharToString[c] = c.ToString();
            }
        }

        public IReadOnlyDictionary<char, char> ByteToUnicode { get; }
        public IReadOnlyDictionary<char, char> UnicodeToByte { get; }
        public string[] CharToString { get; }
    }
}
