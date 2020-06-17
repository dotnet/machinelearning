using System.Collections.Generic;

namespace Microsoft.ML.TimeSeries
{
    internal class MathUtility
    {
        /// <summary>
        /// use quick-sort like method to obtain the median value.
        /// the complexity in expectation is O(n), which is faster than using quickSort.
        /// </summary>
        /// <param name="values">the input list of values. note that this list will be modified after calling this method</param>
        /// <returns>returns the median value</returns>
        public static double QuickMedian(List<double> values)
        {
            if (values == null || values.Count == 0)
                return double.NaN;

            // here the third parameter is start from 1. so we need to plus 1 to compliant.
            return QuickSelect(values, values.Count / 2 + 1);
        }

        /// <summary>
        /// use quick-sort like method to obtain the median value.
        /// the complexity in expectation is O(n), which is faster than using quickSort.
        /// </summary>
        /// <param name="values">The list of values</param>
        /// <param name="k">The k smallest value in the list</param>
        public static double QuickSelect(IReadOnlyList<double> values, int k)
        {
            var nums = values;
            double[] left = new double[values.Count];
            double[] right = new double[values.Count];
            int numsCount = nums.Count;

            while (true)
            {
                if (numsCount == 1)
                    return nums[0];

                int idx = FindMedianIndex(nums, 0, numsCount - 1);
                double key = nums[idx];

                int leftIdx = 0;
                int rightIdx = 0;
                for (int i = 0; i < numsCount; i++)
                {
                    if (i == idx)
                        continue;

                    if (nums[i] < key)
                        left[leftIdx++] = nums[i];
                    else
                        right[rightIdx++] = nums[i];
                }

                if (leftIdx == k - 1)
                    return key;

                if (leftIdx >= k)
                {
                    nums = left;
                    numsCount = leftIdx;
                }
                else
                {
                    nums = right;
                    k = k - leftIdx - 1;
                    numsCount = rightIdx;
                }
            }
        }

        public static int FindMedianIndex(IReadOnlyList<double> values, int start, int end)
        {
            // use the middle value among first/middle/end as the guard value, to make sure the average performance good.
            // according to unit test, this fix will improve the average performance 10%. and works normally when input list is ordered.
            double first = values[start];
            double last = values[end];
            int midIndex = (start + end) / 2;
            int medianIndex = -1;
            double middleValue = values[midIndex];
            if (first < last)
            {
                if (middleValue > last)
                {
                    // last is the middle value
                    medianIndex = end;
                }
                else if (middleValue > first)
                {
                    // middleValue is the middle value
                    medianIndex = midIndex;
                }
                else
                {
                    // first is the middle value
                    medianIndex = start;
                }
            }
            else
            {
                if (middleValue > first)
                {
                    // first is the middle value
                    medianIndex = start;
                }
                else if (middleValue < last)
                {
                    // last is the middle value
                    medianIndex = end;
                }
                else
                {
                    // middleValue is the middle value
                    medianIndex = midIndex;
                }
            }
            return medianIndex;
        }
    }
}
