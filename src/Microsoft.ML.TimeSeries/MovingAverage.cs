using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public class MovingAverage
    {
        /// <summary>
        /// calculate the moving average of a given series.
        /// </summary>
        /// <param name="s">the input series</param>
        /// <param name="length">the length of the moving average window</param>
        public static List<double> MA(IReadOnlyList<double> s, int length)
        {
            //ExtendedDiagnostics.EnsureArgumentNotNull(s, nameof(s));
            if (s.Count <= 1 || length <= 1 || length >= s.Count)
                throw new Exception("the input information for moving average is invalid!");
            //var result1 = MaFast(s, length);
            //var result2 = MaOld(s, length);

            //Trace.Assert(result1.Count == result2.Count);

            //for (int i=0; i<result1.Count; ++i)
            //{
            //    Trace.Assert(result1[i] == result2[i]);
            //}

            return MaFast(s, length);
        }

        public static List<double> MaOld(IReadOnlyList<double> s, int length)
        {
            //ExtendedDiagnostics.EnsureArgumentNotNull(s, nameof(s));

            List<double> series = new List<double>(s);
            int left = length / 2;
            int right = length - left - 1;
            List<double> result = new List<double>();
            for (int i = left; i < series.Count - right; i++)
            {
                int startIndex = i - left;
                int endIndex = i + right;
                double sum = 0;
                for (int j = startIndex; j <= endIndex; j++)
                {
                    sum += series[j];
                }
                result.Add(sum / length);
            }
            return result;
        }

        public static List<double> MaFast(IReadOnlyList<double> s, int length)
        {
            List<double> results = new List<double>(s.Count);
            double partialSum = 0;
            for (int i = 0; i < length; ++i)
            {
                partialSum += s[i];
            }

            for (int i = length; i < s.Count; ++i)
            {
                results.Add(partialSum / length);
                partialSum = partialSum - s[i - length] + s[i];
            }
            results.Add(partialSum / length);

            return results;
        }
    }
}
