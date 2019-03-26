// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

/*============================================================
* Purpose: class to sort Spans
*
* Taken from https://github.com/dotnet/coreclr/blob/480defd204b58fae05b692937295c6533673d3a2/src/System.Private.CoreLib/shared/System/Collections/Generic/ArraySortHelper.cs#L871-L1112
* and changed to support Span instead of arrays.
*
* Code changes from coreclr:
*  1. Name changed from GenericArraySortHelper => GenericSpanSortHelper
*  2. Changed Array usages to Span
*  3. Change Sort method to static
*  4. Changed single-line, multi-variable declarations to be multi-line.
*  5. Contracts.Assert => Contracts.Assert
*
*This can be removed once https://github.com/dotnet/corefx/issues/15329 is fixed.
===========================================================*/

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Numeric
{
    internal static class IntrospectiveSortUtilities
    {
        // This is the threshold where Introspective sort switches to Insertion sort.
        // Empirically, 16 seems to speed up most cases without slowing down others, at least for integers.
        // Large value types may benefit from a smaller number.
        internal const int IntrosortSizeThreshold = 16;

        internal static int FloorLog2PlusOne(int n)
        {
            int result = 0;
            while (n >= 1)
            {
                result++;
                n = n / 2;
            }
            return result;
        }
    }

    internal partial class GenericSpanSortHelper<TKey>
        where TKey : IComparable<TKey>
    {
        public static void Sort<TValue>(Span<TKey> keys, Span<TValue> values, int index, int length)
        {
            Contracts.Assert(keys != null, "Check the arguments in the caller!");
            Contracts.Assert(index >= 0 && length >= 0 && (keys.Length - index >= length), "Check the arguments in the caller!");

            IntrospectiveSort(keys, values, index, length);
        }

        public static void Sort(Span<TKey> keys, int index, int length)
        {
            Sort(keys, keys, index, length);
        }

        private static void SwapIfGreaterWithItems<TValue>(Span<TKey> keys, Span<TValue> values, int a, int b)
        {
            if (a != b)
            {
                if (keys[a] != null && keys[a].CompareTo(keys[b]) > 0)
                {
                    TKey key = keys[a];
                    TValue value = values[a];
                    keys[a] = keys[b];
                    values[a] = values[b];
                    keys[b] = key;
                    values[b] = value;
                }
            }
        }

        private static void Swap<TValue>(Span<TKey> keys, Span<TValue> values, int i, int j)
        {
            if (i != j)
            {
                TKey k = keys[i];
                TValue v = values[i];
                keys[i] = keys[j];
                values[i] = values[j];
                keys[j] = k;
                values[j] = v;
            }
        }

        internal static void IntrospectiveSort<TValue>(Span<TKey> keys, Span<TValue> values, int left, int length)
        {
            Contracts.Assert(keys != null);
            Contracts.Assert(values != null);
            Contracts.Assert(left >= 0);
            Contracts.Assert(length >= 0);
            Contracts.Assert(length <= keys.Length);
            Contracts.Assert(length + left <= keys.Length);
            Contracts.Assert(length + left <= values.Length);

            if (length < 2)
                return;

            IntroSort(keys, values, left, length + left - 1, 2 * IntrospectiveSortUtilities.FloorLog2PlusOne(length));
        }

        private static void IntroSort<TValue>(Span<TKey> keys, Span<TValue> values, int lo, int hi, int depthLimit)
        {
            Contracts.Assert(keys != null);
            Contracts.Assert(values != null);
            Contracts.Assert(lo >= 0);
            Contracts.Assert(hi < keys.Length);

            while (hi > lo)
            {
                int partitionSize = hi - lo + 1;
                if (partitionSize <= IntrospectiveSortUtilities.IntrosortSizeThreshold)
                {
                    if (partitionSize == 1)
                    {
                        return;
                    }
                    if (partitionSize == 2)
                    {
                        SwapIfGreaterWithItems(keys, values, lo, hi);
                        return;
                    }
                    if (partitionSize == 3)
                    {
                        SwapIfGreaterWithItems(keys, values, lo, hi - 1);
                        SwapIfGreaterWithItems(keys, values, lo, hi);
                        SwapIfGreaterWithItems(keys, values, hi - 1, hi);
                        return;
                    }

                    InsertionSort(keys, values, lo, hi);
                    return;
                }

                if (depthLimit == 0)
                {
                    Heapsort(keys, values, lo, hi);
                    return;
                }
                depthLimit--;

                int p = PickPivotAndPartition(keys, values, lo, hi);
                // Note we've already partitioned around the pivot and do not have to move the pivot again.
                IntroSort(keys, values, p + 1, hi, depthLimit);
                hi = p - 1;
            }
        }

        private static int PickPivotAndPartition<TValue>(Span<TKey> keys, Span<TValue> values, int lo, int hi)
        {
            Contracts.Assert(keys != null);
            Contracts.Assert(values != null);
            Contracts.Assert(lo >= 0);
            Contracts.Assert(hi > lo);
            Contracts.Assert(hi < keys.Length);

            // Compute median-of-three.  But also partition them, since we've done the comparison.
            int middle = lo + ((hi - lo) / 2);

            // Sort lo, mid and hi appropriately, then pick mid as the pivot.
            SwapIfGreaterWithItems(keys, values, lo, middle);  // swap the low with the mid point
            SwapIfGreaterWithItems(keys, values, lo, hi);   // swap the low with the high
            SwapIfGreaterWithItems(keys, values, middle, hi); // swap the middle with the high

            TKey pivot = keys[middle];
            Swap(keys, values, middle, hi - 1);
            int left = lo;
            int right = hi - 1;  // We already partitioned lo and hi and put the pivot in hi - 1.  And we pre-increment & decrement below.

            while (left < right)
            {
                if (pivot == null)
                {
                    while (left < (hi - 1) && keys[++left] == null) ;
                    while (right > lo && keys[--right] != null) ;
                }
                else
                {
                    while (pivot.CompareTo(keys[++left]) > 0) ;
                    while (pivot.CompareTo(keys[--right]) < 0) ;
                }

                if (left >= right)
                    break;

                Swap(keys, values, left, right);
            }

            // Put pivot in the right location.
            Swap(keys, values, left, (hi - 1));
            return left;
        }

        private static void Heapsort<TValue>(Span<TKey> keys, Span<TValue> values, int lo, int hi)
        {
            Contracts.Assert(keys != null);
            Contracts.Assert(values != null);
            Contracts.Assert(lo >= 0);
            Contracts.Assert(hi > lo);
            Contracts.Assert(hi < keys.Length);

            int n = hi - lo + 1;
            for (int i = n / 2; i >= 1; i = i - 1)
            {
                DownHeap(keys, values, i, n, lo);
            }
            for (int i = n; i > 1; i = i - 1)
            {
                Swap(keys, values, lo, lo + i - 1);
                DownHeap(keys, values, 1, i - 1, lo);
            }
        }

        private static void DownHeap<TValue>(Span<TKey> keys, Span<TValue> values, int i, int n, int lo)
        {
            Contracts.Assert(keys != null);
            Contracts.Assert(lo >= 0);
            Contracts.Assert(lo < keys.Length);

            TKey d = keys[lo + i - 1];
            TValue dValue = values[lo + i - 1];
            int child;
            while (i <= n / 2)
            {
                child = 2 * i;
                if (child < n && (keys[lo + child - 1] == null || keys[lo + child - 1].CompareTo(keys[lo + child]) < 0))
                {
                    child++;
                }
                if (keys[lo + child - 1] == null || keys[lo + child - 1].CompareTo(d) < 0)
                    break;
                keys[lo + i - 1] = keys[lo + child - 1];
                values[lo + i - 1] = values[lo + child - 1];
                i = child;
            }
            keys[lo + i - 1] = d;
            values[lo + i - 1] = dValue;
        }

        private static void InsertionSort<TValue>(Span<TKey> keys, Span<TValue> values, int lo, int hi)
        {
            Contracts.Assert(keys != null);
            Contracts.Assert(values != null);
            Contracts.Assert(lo >= 0);
            Contracts.Assert(hi >= lo);
            Contracts.Assert(hi <= keys.Length);

            int i;
            int j;
            TKey t;
            TValue tValue;
            for (i = lo; i < hi; i++)
            {
                j = i;
                t = keys[i + 1];
                tValue = values[i + 1];
                while (j >= lo && (t == null || t.CompareTo(keys[j]) < 0))
                {
                    keys[j + 1] = keys[j];
                    values[j + 1] = values[j];
                    j--;
                }
                keys[j + 1] = t;
                values[j + 1] = tValue;
            }
        }
    }
}
