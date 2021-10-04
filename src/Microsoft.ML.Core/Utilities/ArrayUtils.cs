// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;

namespace Microsoft.ML.Internal.Utilities
{
    internal static partial class ArrayUtils
    {
        // Maximum size of one-dimensional array.
        // See: https://msdn.microsoft.com/en-us/library/hh285054(v=vs.110).aspx
        public const int ArrayMaxSize = 0X7FEFFFFF;

        public static int Size<T>(T[] x)
        {
            return x == null ? 0 : x.Length;
        }

        /// <summary>
        /// Akin to <c>FindIndexSorted</c>, except stores the found index in the output
        /// <c>index</c> parameter, and returns whether that index is a valid index
        /// pointing to a value equal to the input parameter <c>value</c>.
        /// </summary>
        public static bool TryFindIndexSorted(int[] input, int min, int lim, int value, out int index)
        {
            index = FindIndexSorted(input, min, lim, value);
            return index < lim && input[index] == value;
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(int[] input, int min, int lim, int value)
        {
            return FindIndexSorted(input.AsSpan(), min, lim, value);
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(ReadOnlySpan<int> input, int min, int lim, int value)
        {
            Debug.Assert(0 <= min && min <= lim && lim <= input.Length);

            int minCur = min;
            int limCur = lim;
            while (minCur < limCur)
            {
                int mid = (int)(((uint)minCur + (uint)limCur) / 2);
                Debug.Assert(minCur <= mid && mid < limCur);

                if (input[mid] >= value)
                    limCur = mid;
                else
                    minCur = mid + 1;

                Debug.Assert(min <= minCur && minCur <= limCur && limCur <= lim);
                Debug.Assert(minCur == min || input[minCur - 1] < value);
                Debug.Assert(limCur == lim || input[limCur] >= value);
            }
            Debug.Assert(min <= minCur && minCur == limCur && limCur <= lim);
            Debug.Assert(minCur == min || input[minCur - 1] < value);
            Debug.Assert(limCur == lim || input[limCur] >= value);

            return minCur;
        }

        public static int EnsureSize<T>(ref T[] array, int min, int max, bool keepOld, out bool resized)
        {
            if (min > max)
                throw new ArgumentOutOfRangeException(nameof(max), "min must not exceed max");

            // This code adapted from the private method EnsureCapacity code of List<T>.
            int size = ArrayUtils.Size(array);
            if (size >= min)
            {
                resized = false;
                return size;
            }

            int newSize = size == 0 ? 4 : size * 2;
            // This constant taken from the internal code of system\array.cs of mscorlib.
            if ((uint)newSize > max)
                newSize = max;
            if (newSize < min)
                newSize = min;
            if (keepOld && size > 0)
                Array.Resize(ref array, newSize);
            else
                array = new T[newSize];

            resized = true;
            return newSize;
        }
    }
}
