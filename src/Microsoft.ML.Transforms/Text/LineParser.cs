using System;
using System.Linq;

namespace Microsoft.ML.Transforms.Text
{
    public static class LineParser
    {
        public static (bool isSuccess, string key, float[] values) ParseKeyThenNumbers(string line)
        {
            char[] delimiters = { ' ', '\t' };
            string[] words = line.TrimEnd().Split(delimiters);
            string key = words[0];
            float[] values = words.Skip(1).Select(x => float.TryParse(x, out var tmp) ? tmp : Single.NaN).ToArray();
            if (!values.Contains(Single.NaN))
                return (true, key, values);

            return (false, null, null);
        }
    }
}