using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal static class LineParser
    {
        [BestFriend] // required because the return type is a value tuple..
        internal static (bool isSuccess, string key, float[] values) ParseKeyThenNumbers(string line)
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
                    if (DoubleParser.TryParse(valuesToParse.Slice(toParseStartIndex, i - toParseStartIndex), out float parsed))
                        values[valueIndex++] = parsed;
                    else
                        return (false, null, null);

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