using System;
using System.Globalization;
using System.Linq;

namespace Microsoft.ML.Transforms.Text
{
    public static class LineParser
    {
        private static readonly char[] _delimiters = { ' ', '\t' };

#if NETSTANDARD2_0
        public static (bool isSuccess, string key, float[] values) ParseKeyThenNumbers(string line)
        {
            if (string.IsNullOrWhiteSpace(line))
                return (false, null, null);

            string[] words = line.TrimEnd().Split(_delimiters);

            NumberFormatInfo info = NumberFormatInfo.CurrentInfo; // moved otuside the loop to save  8% of the time
            float[] values = new float[words.Length - 1];
            for (int i = 1; i < words.Length; i++)
            {
                if (float.TryParse(words[i], NumberStyles.Float | NumberStyles.AllowThousands, info, out float parsed))
                    values[i - 1] = parsed;
                else
                    return (false, null, null);
            }

            return (true, words[0], values);
        }
#else
        public static (bool isSuccess, string key, float[] values) ParseKeyThenNumbers(string line)
        {
            if (string.IsNullOrWhiteSpace(line))
                return (false, null, null);

            ReadOnlySpan<char> trimmedLine = line.AsSpan().TrimEnd(); // TrimEnd creates a Span, no allocations

            int firstSeparatorIndex = trimmedLine.IndexOfAny(' ', '\t'); // the first word is the key, we just skip it
            ReadOnlySpan<char> valuesToParse = trimmedLine.Slice(start: firstSeparatorIndex + 1);

            int valuesCount = 0; // we count the number of values first to allocate a single array with of proper size
            for (int i = 0; i < valuesToParse.Length; i++)
                if (valuesToParse[i] == ' ' || valuesToParse[i] == '\t')
                    valuesCount++;

            float[] values = new float[valuesCount + 1]; // + 1 because the line is trimmed and there is no whitespace at the end
            int textStart = 0;
            int valueIndex = 0;
            NumberFormatInfo info = NumberFormatInfo.CurrentInfo; // moved otuside the loop to save  8% of the time
            for (int i = 0; i <= valuesToParse.Length; i++)
            {
                if (i == valuesToParse.Length || valuesToParse[i] == ' ' || valuesToParse[i] == '\t')
                {
                    var toParse = valuesToParse.Slice(textStart, i - textStart);

                    if (float.TryParse(toParse, NumberStyles.Float | NumberStyles.AllowThousands, info, out float parsed))
                        values[valueIndex++] = parsed;
                    else
                        return (false, null, null);

                    textStart = i + 1;
                }
            }

            return (true, new string(trimmedLine.Slice(0, firstSeparatorIndex)), values);
        }
#endif
    }
}