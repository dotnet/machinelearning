// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public static class Algorithms
    {
        /// <summary>
        /// Returns the index of the first array position that is larger than or equal to val
        /// </summary>
        /// <typeparam name="T">an IComparable type</typeparam>
        /// <param name="array">a sorted array of values</param>
        /// <param name="val">the value to search for</param>
        public static int FindFirstGE<T>(T[] array, T val) where T : IComparable
        {
            if (val.CompareTo(array[0]) <= 0)
                return 0;
            if (val.CompareTo(array[array.Length - 1]) > 0)
                //return array.Length - 1;
                throw Contracts.Except("value is greater than the greatest number in the array");

            // Considering anything from lower to upper, inclusive
            int lower = 0;
            int upper = array.Length - 1;
            int mid;
            int comparison;
            T midValue;
            while (true)
            {
                mid = (upper + lower) / 2;
                midValue = array[mid];

                comparison = val.CompareTo(midValue);

                if (comparison < 0)
                    upper = mid;
                else if (comparison > 0)
                    lower = mid;
                else
                    return mid;

                if (upper - lower == 1)
                    return upper;
            }
        }

        /// <summary>
        /// Returns the index of the last array position that is less than or equal to val
        /// </summary>
        /// <typeparam name="T">an IComparable type</typeparam>
        /// <param name="array">a sorted array of values</param>
        /// <param name="val">the value to search for</param>
        public static int FindLastLE<T>(T[] array, T val) where T : IComparable
        {
            if (val.CompareTo(array[0]) < 0)
                throw Contracts.Except("value is less than the first array element");
            if (val.CompareTo(array[array.Length - 1]) >= 0)
                return array.Length - 1;

            // Considering anything from lower to upper, inclusive
            int lower = 0;
            int upper = array.Length - 1;
            int mid;
            int comparison;
            T midValue;
            while (true)
            {
                mid = (upper + lower) / 2;
                midValue = array[mid];

                comparison = val.CompareTo(midValue);

                if (comparison < 0)
                    upper = mid;
                else if (comparison > 0)
                    lower = mid;
                else
                    return mid;

                if (upper - lower == 1)
                    return lower;
            }
        }

        /// <summary>
        /// Finds the largest k entries in an array (with offset and length)
        /// </summary>
        /// <typeparam name="T">An IComparible type</typeparam>
        /// <param name="array">The array being searched</param>
        /// <param name="offset">An offset into the array</param>
        /// <param name="length">The length of the search</param>
        /// <param name="topK">The values of the top K</param>
        /// <param name="topKPositions">The positions of the top K</param>
        /// <returns>The number of entries set in topK and topKPositions (length could be less than K)</returns>
        public static int TopK<T>(T[] array, int offset, int length, T[] topK, int[] topKPositions) where T : IComparable
        {
            int k = topK.Length < length ? topK.Length : length;
            for (int i = 0; i < k; ++i)
            {
                topK[i] = array[i];
                topKPositions[i] = i;
            }

            int argmin = -1;
            T min = Min(topK, out argmin);

            for (int i = k; i < length; ++i)
            {
                if (min.CompareTo(array[i]) > 0)
                {
                    topK[argmin] = array[i];
                    topKPositions[argmin] = i;
                    min = Min(topK, out argmin);
                }
            }

            return k;
        }

        /// <summary>
        /// Fidns the minimum and the argmin in an array of values
        /// </summary>
        public static T Min<T>(T[] array, out int argmin) where T : IComparable
        {
            T min = array[0];
            argmin = 0;
            for (int i = 1; i < array.Length; ++i)
            {
                if (min.CompareTo(array[i]) > 0)
                {
                    min = array[i];
                    argmin = i;
                }
            }
            return min;
        }

        /// <summary>
        /// Takes an arbitrary array of sorted uniqued IComparables and returns a sorted uniqued merge
        /// </summary>
        /// <typeparam name="T">An IComparable </typeparam>
        /// <param name="arrays">An array of sorted uniqued arrays</param>
        /// <returns>A sorted and uniqued merge</returns>
        public static T[] MergeSortedUniqued<T>(T[][] arrays) where T : IComparable
        {
            int maxLength = arrays.Sum(x => x.Length);
            T[] working = new T[maxLength];
            int[] begins = new int[arrays.Length + 1];
            int begin = 0;
            for (int i = 0; i < arrays.Length; ++i)
            {
                begins[i] = begin;
                Array.Copy(arrays[i], 0, working, begin, arrays[i].Length);
                begin += arrays[i].Length;
            }
            begins[arrays.Length] = begin;
            T[] tmp = new T[maxLength];
            int length = MergeSortedUniqued(begins, 0, arrays.Length, working, tmp);

            T[] output = new T[length];
            Array.Copy(working, output, length);
            return output;
        }

        private static int MergeSortedUniqued<T>(int[] begins, int fromArray, int toArray, T[] working, T[] tmp) where T : IComparable
        {
            if (toArray - fromArray == 1)
            {
                return begins[toArray] - begins[fromArray];
            }
            else if (toArray - fromArray == 2)
            {
                int length = MergeSortedUniqued(working, begins[fromArray], begins[fromArray + 1], working, begins[fromArray + 1], begins[toArray], tmp, begins[fromArray]);
                Array.Copy(tmp, begins[fromArray], working, begins[fromArray], length);
                return length;
            }
            else
            {
                int midArray = fromArray + (toArray - fromArray) / 2;
                int length1 = MergeSortedUniqued(begins, fromArray, midArray, working, tmp);
                int length2 = MergeSortedUniqued(begins, midArray, toArray, working, tmp);
                int length = MergeSortedUniqued(working, begins[fromArray], begins[fromArray] + length1, working, begins[midArray], begins[midArray] + length2, tmp, begins[fromArray]);
                Array.Copy(tmp, begins[fromArray], working, begins[fromArray], length);
                return length;
            }
        }

        public static int MergeSortedUniqued<T>(T[] input1, int begin1, int end1, T[] input2, int begin2, int end2, T[] output, int beginOutput) where T : IComparable
        {
            int beginOutputCopy = beginOutput;

            while (begin1 < end1 && begin2 < end2)
            {
                int compare = input1[begin1].CompareTo(input2[begin2]);
                if (compare < 0)
                    output[beginOutput++] = input1[begin1++];
                else if (compare > 0)
                    output[beginOutput++] = input2[begin2++];
                else
                {
                    output[beginOutput++] = input1[begin1++];
                    begin2++;
                }
            }

            while (begin1 < end1)
            {
                output[beginOutput++] = input1[begin1++];
            }

            while (begin2 < end2)
            {
                output[beginOutput++] = input2[begin2++];
            }

            return beginOutput - beginOutputCopy;
        }
    }
}
