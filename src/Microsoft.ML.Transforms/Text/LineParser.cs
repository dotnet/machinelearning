using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Transforms.Text
{
    public static class LineParser
    {
        public static (bool isSuccess, string key, float[] values) ParseKeyThenNumbers(string line)
        {
            string key = null;
            float[] values = null;

            return (true, key, values);
        }
    }
}
