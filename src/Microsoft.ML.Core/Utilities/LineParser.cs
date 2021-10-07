// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Internal.Utilities
{
    [BestFriend]
    internal static class LineParser
    {
        public static (bool isSuccess, string key, float[] values) ParseKeyThenNumbers(string line, bool invariantCulture)
        {
            if (string.IsNullOrWhiteSpace(line))
                return (false, null, null);

            ReadOnlySpan<char> trimmedLine = line.AsSpan().TrimEnd(); // TrimEnd creates a Span, no allocations

            int firstSeparatorIndex = trimmedLine.IndexOfAny(' ', '\t'); // the first word is the key, we just skip it
            ReadOnlySpan<char> valuesToParse = trimmedLine.Slice(start: firstSeparatorIndex + 1);

            float[] values = AllocateFixedSizeArrayToStoreParsedValues(valuesToParse);

            int toParseStartIndex = 0;
            int valueIndex = 0;
            for (int i = 0; i <= valuesToParse.Length; i++)
            {
                if (i == valuesToParse.Length || valuesToParse[i] == ' ' || valuesToParse[i] == '\t')
                {
                    if (invariantCulture)
                    {
                        if (DoubleParser.TryParse(valuesToParse.Slice(toParseStartIndex, i - toParseStartIndex), out float parsed))
                            values[valueIndex++] = parsed;
                        else
                            return (false, null, null);
                    }
                    else
                    {
                        if (float.TryParse(valuesToParse.Slice(toParseStartIndex, i - toParseStartIndex).ToString(), out float parsed))
                            values[valueIndex++] = parsed;
                        else
                            return (false, null, null);
                    }

                    toParseStartIndex = i + 1;
                }
            }

            return (true, trimmedLine.Slice(0, firstSeparatorIndex).ToString(), values);
        }

        /// <summary>
        /// we count the number of values first to allocate a single array with of proper size
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float[] AllocateFixedSizeArrayToStoreParsedValues(ReadOnlySpan<char> valuesToParse)
        {
            int valuesCount = 0;

            for (int i = 0; i < valuesToParse.Length; i++)
                if (valuesToParse[i] == ' ' || valuesToParse[i] == '\t')
                    valuesCount++;

            return new float[valuesCount + 1]; // + 1 because the line is trimmed and there is no whitespace at the end
        }
    }
}
