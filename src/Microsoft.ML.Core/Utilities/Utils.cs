// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Microsoft.ML.Runtime;
namespace Microsoft.ML.Internal.Utilities
{

    [BestFriend]
    internal static partial class Utils
    {
        // Maximum size of one-dimensional array.
        // See: https://msdn.microsoft.com/en-us/library/hh285054(v=vs.110).aspx
        public const int ArrayMaxSize = 0X7FEFFFFF;

        public static bool StartsWithInvariantCultureIgnoreCase(this string str, string startsWith)
        {
            return str.StartsWith(startsWith, StringComparison.InvariantCultureIgnoreCase);
        }

        public static bool StartsWithInvariantCulture(this string str, string startsWith)
        {
            return str.StartsWith(startsWith, StringComparison.InvariantCulture);
        }

        public static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }

        public static void Reverse<T>(T[] a, int iMin, int iLim)
        {
            while (iMin < --iLim)
                Swap(ref a[iMin++], ref a[iLim]);
        }

        // Getting the size of a collection, when the collection may be null.

        public static int Size(string x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Length;
        }

        public static int Size(StringBuilder x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Length;
        }

        public static int Size(Array x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Length;
        }

        public static int Size<T>(T[] x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Length;
        }

        public static int Size<T>(List<T> x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Count;
        }

        public static int Size<T>(IList<T> x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Count;
        }

        public static int Size<T>(IReadOnlyList<T> x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Count;
        }

        public static int Size<T>(Stack<T> x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Count;
        }

        public static int Size<T>(HashSet<T> x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Count;
        }

        public static int Size<T>(SortedSet<T> x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Count;
        }

        public static int Size<TKey, TValue>(Dictionary<TKey, TValue> x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Count;
        }

        public static int Size(BitArray x)
        {
            Contracts.AssertValueOrNull(x);
            return x == null ? 0 : x.Length;
        }

        // Getting items from a collection when the collection may be null.

        public static bool TryGetValue<TKey, TValue>(Dictionary<TKey, TValue> map, TKey key, out TValue value)
        {
            Contracts.AssertValueOrNull(map);
            if (map == null)
            {
                value = default(TValue);
                return false;
            }
            return map.TryGetValue(key, out value);
        }

        public static T[] ToArray<T>(List<T> list)
        {
            Contracts.AssertValueOrNull(list);
            return list == null ? null : list.ToArray();
        }

        // Adding items to a collection when the collection may be null.

        public static void Add<T>(ref List<T> list, T item)
        {
            Contracts.AssertValueOrNull(list);
            if (list == null)
                list = new List<T>();
            list.Add(item);
        }

        public static bool Add<T>(ref HashSet<T> set, T item)
        {
            Contracts.AssertValueOrNull(set);
            if (set == null)
                set = new HashSet<T>();
            return set.Add(item);
        }

        public static void Add<TKey, TValue>(ref Dictionary<TKey, TValue> map, TKey key, TValue value)
        {
            Contracts.AssertValueOrNull(map);
            if (map == null)
                map = new Dictionary<TKey, TValue>();
            map.Add(key, value);
        }

        public static void Set<TKey, TValue>(ref Dictionary<TKey, TValue> map, TKey key, TValue value)
        {
            Contracts.AssertValueOrNull(map);
            if (map == null)
                map = new Dictionary<TKey, TValue>();
            map[key] = value;
        }

        public static void Push<T>(ref Stack<T> stack, T item)
        {
            Contracts.AssertValueOrNull(stack);
            if (stack == null)
                stack = new Stack<T>();
            stack.Push(item);
        }

        /// <summary>
        /// Copies the values from src to dst.
        /// </summary>
        /// <remarks>
        /// This can be removed once we have the APIs from https://github.com/dotnet/corefx/issues/33006.
        /// </remarks>
        public static void CopyTo<T>(this List<T> src, Span<T> dst, int? count = null)
        {
            Contracts.Assert(src != null);
            Contracts.Assert(!count.HasValue || (0 <= count && count <= src.Count));
            Contracts.Assert(src.Count <= dst.Length);

            count = count ?? src.Count;
            for (int i = 0; i < count; i++)
            {
                dst[i] = src[i];
            }
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(this IList<int> input, int value)
        {
            Contracts.AssertValue(input);
            return FindIndexSorted(input, 0, input.Count, value);
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(this IList<float> input, float value)
        {
            Contracts.AssertValue(input);
            return FindIndexSorted(input, 0, input.Count, value);
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(this Double[] input, Double value)
        {
            Contracts.AssertValue(input);
            return FindIndexSorted(input, 0, input.Length, value);
        }

        /// <summary>
        /// Akin to <c>FindIndexSorted</c>, except stores the found index in the output
        /// <c>index</c> parameter, and returns whether that index is a valid index
        /// pointing to a value equal to the input parameter <c>value</c>.
        /// </summary>
        public static bool TryFindIndexSorted(this int[] input, int min, int lim, int value, out int index)
        {
            index = input.FindIndexSorted(min, lim, value);
            return index < lim && input[index] == value;
        }

        /// <summary>
        /// Akin to <c>FindIndexSorted</c>, except stores the found index in the output
        /// <c>index</c> parameter, and returns whether that index is a valid index
        /// pointing to a value equal to the input parameter <c>value</c>.
        /// </summary>
        public static bool TryFindIndexSorted(ReadOnlySpan<int> input, int min, int lim, int value, out int index)
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
        public static int FindIndexSorted(this int[] input, int min, int lim, int value)
        {
            return FindIndexSorted(input.AsSpan(), min, lim, value);
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(this ReadOnlySpan<int> input, int min, int lim, int value)
        {
            Contracts.Assert(0 <= min & min <= lim & lim <= input.Length);

            int minCur = min;
            int limCur = lim;
            while (minCur < limCur)
            {
                int mid = (int)(((uint)minCur + (uint)limCur) / 2);
                Contracts.Assert(minCur <= mid & mid < limCur);

                if (input[mid] >= value)
                    limCur = mid;
                else
                    minCur = mid + 1;

                Contracts.Assert(min <= minCur & minCur <= limCur & limCur <= lim);
                Contracts.Assert(minCur == min || input[minCur - 1] < value);
                Contracts.Assert(limCur == lim || input[limCur] >= value);
            }
            Contracts.Assert(min <= minCur & minCur == limCur & limCur <= lim);
            Contracts.Assert(minCur == min || input[minCur - 1] < value);
            Contracts.Assert(limCur == lim || input[limCur] >= value);

            return minCur;
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(this IList<int> input, int min, int lim, int value)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= min & min <= lim & lim <= input.Count);

            int minCur = min;
            int limCur = lim;
            while (minCur < limCur)
            {
                int mid = (int)(((uint)minCur + (uint)limCur) / 2);
                Contracts.Assert(minCur <= mid & mid < limCur);

                if (input[mid] >= value)
                    limCur = mid;
                else
                    minCur = mid + 1;

                Contracts.Assert(min <= minCur & minCur <= limCur & limCur <= lim);
                Contracts.Assert(minCur == min || input[minCur - 1] < value);
                Contracts.Assert(limCur == lim || input[limCur] >= value);
            }
            Contracts.Assert(min <= minCur & minCur == limCur & limCur <= lim);
            Contracts.Assert(minCur == min || input[minCur - 1] < value);
            Contracts.Assert(limCur == lim || input[limCur] >= value);

            return minCur;
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(this IList<float> input, int min, int lim, float value)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= min & min <= lim & lim <= input.Count);
            Contracts.Assert(!float.IsNaN(value));

            int minCur = min;
            int limCur = lim;
            while (minCur < limCur)
            {
                int mid = (int)(((uint)minCur + (uint)limCur) / 2);
                Contracts.Assert(minCur <= mid & mid < limCur);
                Contracts.Assert(!float.IsNaN(input[mid]));

                if (input[mid] >= value)
                    limCur = mid;
                else
                    minCur = mid + 1;

                Contracts.Assert(min <= minCur & minCur <= limCur & limCur <= lim);
                Contracts.Assert(minCur == min || input[minCur - 1] < value);
                Contracts.Assert(limCur == lim || input[limCur] >= value);
            }
            Contracts.Assert(min <= minCur & minCur == limCur & limCur <= lim);
            Contracts.Assert(minCur == min || input[minCur - 1] < value);
            Contracts.Assert(limCur == lim || input[limCur] >= value);

            return minCur;
        }

        /// <summary>
        /// Assumes input is sorted and finds value using BinarySearch.
        /// If value is not found, returns the logical index of 'value' in the sorted list i.e index of the first element greater than value.
        /// In case of duplicates it returns the index of the first one.
        /// It guarantees that items before the returned index are &lt; value, while those at and after the returned index are &gt;= value.
        /// </summary>
        public static int FindIndexSorted(this Double[] input, int min, int lim, Double value)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= min & min <= lim & lim <= input.Length);
            Contracts.Assert(!Double.IsNaN(value));

            int minCur = min;
            int limCur = lim;
            while (minCur < limCur)
            {
                int mid = (int)(((uint)minCur + (uint)limCur) / 2);
                Contracts.Assert(minCur <= mid & mid < limCur);
                Contracts.Assert(!Double.IsNaN(input[mid]));

                if (input[mid] >= value)
                    limCur = mid;
                else
                    minCur = mid + 1;

                Contracts.Assert(min <= minCur & minCur <= limCur & limCur <= lim);
                Contracts.Assert(minCur == min || input[minCur - 1] < value);
                Contracts.Assert(limCur == lim || input[limCur] >= value);
            }
            Contracts.Assert(min <= minCur & minCur == limCur & limCur <= lim);
            Contracts.Assert(minCur == min || input[minCur - 1] < value);
            Contracts.Assert(limCur == lim || input[limCur] >= value);

            return minCur;
        }

        /// <summary>
        /// Finds the unique index for which func(input[i]) == false whenever i &lt; index and
        /// func(input[i]) == true whenever i &gt;= index.
        /// Callers should guarantee that there is such an index. Uses binary search.
        /// </summary>
        public static int FindIndexSorted<T>(this T[] input, int min, int lim, Func<T, bool> func)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= min & min <= lim & lim <= input.Length);

            int minCur = min;
            int limCur = lim;
            while (minCur < limCur)
            {
                int mid = (int)(((uint)minCur + (uint)limCur) / 2);
                Contracts.Assert(minCur <= mid & mid < limCur);

                if (func(input[mid]))
                    limCur = mid;
                else
                    minCur = mid + 1;

                Contracts.Assert(min <= minCur & minCur <= limCur & limCur <= lim);
                Contracts.Assert(minCur == min || !func(input[minCur - 1]));
                Contracts.Assert(limCur == lim || func(input[limCur]));
            }
            Contracts.Assert(min <= minCur & minCur == limCur & limCur <= lim);
            Contracts.Assert(minCur == min || !func(input[minCur - 1]));
            Contracts.Assert(limCur == lim || func(input[limCur]));

            return minCur;
        }

        /// <summary>
        /// Finds the unique index for which func(input[i], value) == false whenever i &lt; index and
        /// func(input[i], value) == true whenever i &gt;= index.
        /// Callers should guarantee that there is such an index. Uses binary search.
        /// </summary>
        public static int FindIndexSorted<T, TValue>(this T[] input, int min, int lim, Func<T, TValue, bool> func, TValue value)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= min & min <= lim & lim <= input.Length);

            int minCur = min;
            int limCur = lim;
            while (minCur < limCur)
            {
                int mid = (int)(((uint)minCur + (uint)limCur) / 2);
                Contracts.Assert(minCur <= mid & mid < limCur);

                if (func(input[mid], value))
                    limCur = mid;
                else
                    minCur = mid + 1;

                Contracts.Assert(min <= minCur & minCur <= limCur & limCur <= lim);
                Contracts.Assert(minCur == min || !func(input[minCur - 1], value));
                Contracts.Assert(limCur == lim || func(input[limCur], value));
            }
            Contracts.Assert(min <= minCur & minCur == limCur & limCur <= lim);
            Contracts.Assert(minCur == min || !func(input[minCur - 1], value));
            Contracts.Assert(limCur == lim || func(input[limCur], value));

            return minCur;
        }

        public static int[] GetIdentityPermutation(int size)
        {
            Contracts.Assert(size >= 0);

            var res = new int[size];
            for (int i = 0; i < size; i++)
                res[i] = i;
            return res;
        }

        public static void FillIdentity(Span<int> a, int lim)
        {
            Contracts.Assert(0 <= lim & lim <= a.Length);

            for (int i = 0; i < lim; ++i)
                a[i] = i;
        }

        // REVIEW: Maybe remove this in future?
        public static void InterlockedAdd(ref double target, double v)
        {
            double snapshotOfTargetBefore;
            double snapshotOfTargetDuring = target;
            double targetPlusV;

            do
            {
                snapshotOfTargetBefore = snapshotOfTargetDuring;
                targetPlusV = snapshotOfTargetBefore + v;

                snapshotOfTargetDuring = Interlocked.CompareExchange(ref target, targetPlusV, snapshotOfTargetBefore);

            } while (snapshotOfTargetDuring != snapshotOfTargetBefore);
        }

        public static int[] InvertPermutation(int[] perm)
        {
            Contracts.AssertValue(perm);

            var res = new int[perm.Length];
            for (int i = 0; i < perm.Length; i++)
            {
                int j = perm[i];
                Contracts.Assert(0 <= j & j < perm.Length);
                Contracts.Assert(res[j] == 0 & (j != perm[0] | i == 0));
                res[j] = i;
            }
            return res;
        }

        public static int[] GetRandomPermutation(Random rand, int size)
        {
            Contracts.AssertValue(rand);
            Contracts.Assert(size >= 0);

            var res = GetIdentityPermutation(size);
            Shuffle<int>(rand, res);
            return res;
        }

        public static bool AreEqual(float[] arr1, float[] arr2)
        {
            if (arr1 == arr2)
                return true;
            if (arr1 == null || arr2 == null)
                return false;
            if (arr1.Length != arr2.Length)
                return false;

            for (int i = 0; i < arr1.Length; i++)
            {
                if (arr1[i] != arr2[i])
                    return false;
            }
            return true;
        }

        public static bool AreEqual(double[] arr1, double[] arr2)
        {
            if (arr1 == arr2)
                return true;
            if (arr1 == null || arr2 == null)
                return false;
            if (arr1.Length != arr2.Length)
                return false;

            for (int i = 0; i < arr1.Length; i++)
            {
                if (arr1[i] != arr2[i])
                    return false;
            }
            return true;
        }

        public static void Shuffle<T>(Random rand, Span<T> rgv)
        {
            Contracts.AssertValue(rand);

            for (int iv = 0; iv < rgv.Length; iv++)
                Swap(ref rgv[iv], ref rgv[iv + rand.Next(rgv.Length - iv)]);
        }

        public static bool AreEqual(int[] arr1, int[] arr2)
        {
            if (arr1 == arr2)
                return true;
            if (arr1 == null || arr2 == null)
                return false;
            if (arr1.Length != arr2.Length)
                return false;

            for (int i = 0; i < arr1.Length; i++)
            {
                if (arr1[i] != arr2[i])
                    return false;
            }
            return true;
        }

        public static bool AreEqual(bool[] arr1, bool[] arr2)
        {
            if (arr1 == arr2)
                return true;
            if (arr1 == null || arr2 == null)
                return false;
            if (arr1.Length != arr2.Length)
                return false;

            for (int i = 0; i < arr1.Length; i++)
            {
                if (arr1[i] != arr2[i])
                    return false;
            }
            return true;
        }

        public static string ExtractLettersAndNumbers(string value)
        {
            return Regex.Replace(value, "[^A-Za-z0-9]", "");
        }

        /// <summary>
        /// Checks that an input IList is monotonically increasing.
        /// </summary>
        /// <param name="values">An array of values</param>
        /// <returns>True if the array is monotonically increasing (if each element is greater
        /// than or equal to previous elements); false otherwise. ILists containing NaN values
        /// are considered to be not monotonically increasing.</returns>
        public static bool IsMonotonicallyIncreasing(IList<float> values)
        {
            if (Utils.Size(values) <= 1)
                return true;

            var previousValue = values[0];
            var listLength = values.Count;
            for (int i = 1; i < listLength; i++)
            {
                var currentValue = values[i];
                // Inverted check for NaNs
                if (!(currentValue >= previousValue))
                    return false;

                previousValue = currentValue;
            }

            return true;
        }

        /// <summary>
        /// Checks that an input array is monotonically increasing.
        /// </summary>
        /// <param name="values">An array of values</param>
        /// <returns>True if the array is monotonically increasing (if each element is greater
        /// than or equal to previous elements); false otherwise.</returns>
        public static bool IsMonotonicallyIncreasing(IList<int> values)
        {
            if (Utils.Size(values) <= 1)
                return true;

            var previousValue = values[0];
            var listLength = values.Count;
            for (int i = 1; i < listLength; i++)
            {
                var currentValue = values[i];
                if (currentValue < previousValue)
                    return false;

                previousValue = currentValue;
            }

            return true;
        }

        /// <summary>
        /// Checks that an input array is monotonically increasing.
        /// </summary>
        /// <param name="values">An array of values</param>
        /// <returns>True if the array is monotonically increasing (if each element is greater
        /// than or equal to previous elements); false otherwise. Arrays containing NaN values
        /// are considered to be not monotonically increasing.</returns>
        public static bool IsMonotonicallyIncreasing(IList<double> values)
        {
            if (Utils.Size(values) <= 1)
                return true;

            var previousValue = values[0];
            var listLength = values.Count;
            for (int i = 1; i < listLength; i++)
            {
                var currentValue = values[i];
                // Inverted check for NaNs
                if (!(currentValue >= previousValue))
                    return false;

                previousValue = currentValue;
            }

            return true;
        }

        /// <summary>
        /// Returns whether an input integer vector is sorted and unique,
        /// and between an inclusive lower and exclusive upper bound for
        /// the first and last items, respectively.
        /// </summary>
        public static bool IsIncreasing(int min, ReadOnlySpan<int> values, int lim)
        {
            if (values.Length < 1)
                return true;

            var prev = values[0];
            if (prev < min)
                return false;
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] <= prev)
                    return false;
                prev = values[i];
            }
            return prev < lim;
        }

        /// <summary>
        /// Returns whether an input integer vector up to <paramref name="len"/>
        /// is sorted and unique, and between an inclusive lower and exclusive
        /// upper bound for the first and last items, respectively.
        /// </summary>
        public static bool IsIncreasing(int min, ReadOnlySpan<int> values, int len, int lim)
        {
            Contracts.Check(values.Length >= len);
            if (len < 1)
                return true;

            var prev = values[0];
            if (prev < min)
                return false;
            for (int i = 1; i < len; i++)
            {
                if (values[i] <= prev)
                    return false;
                prev = values[i];
            }
            return prev < lim;
        }

        /// <summary>
        /// Create an array of specified length, filled with a specified value
        /// </summary>
        public static T[] CreateArray<T>(int length, T value)
        {
            Contracts.Assert(length >= 0, "Length can't be negative");
            var result = new T[length];
            for (int i = 0; i < length; i++)
                result[i] = value;
            return result;
        }

        public static bool[] BuildArray(int length, IEnumerable<DataViewSchema.Column> columnsNeeded)
        {
            Contracts.CheckParam(length >= 0, nameof(length));

            var result = new bool[length];
            foreach (var col in columnsNeeded)
            {
                if (col.Index < result.Length)
                    result[col.Index] = true;
            }

            return result;
        }

        public static T[] BuildArray<T>(int length, Func<int, T> func)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckValue(func, nameof(func));

            var result = new T[length];
            for (int i = 0; i < result.Length; i++)
                result[i] = func(i);
            return result;
        }

        /// <summary>
        /// Given a predicate, over a range of values defined by a limit calculate
        /// first the values for which that predicate was true, and second an inverse
        /// map.
        /// </summary>
        /// <param name="schema">The input schema where the predicate can check if columns are active.</param>
        /// <param name="pred">The predicate to test for various value</param>
        /// <param name="map">An ascending array of values from 0 inclusive
        /// to <paramref name="schema.Count"/> exclusive, holding all values for which
        /// <paramref name="pred"/> is true</param>
        /// <param name="invMap">Forms an inverse mapping of <paramref name="map"/>,
        /// so that <c><paramref name="invMap"/>[<paramref name="map"/>[i]] == i</c>,
        /// and for other entries not appearing in <paramref name="map"/>,
        /// <c><paramref name="invMap"/>[i] == -1</c></param>
        public static void BuildSubsetMaps(DataViewSchema schema, Func<DataViewSchema.Column, bool> pred, out int[] map, out int[] invMap)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(schema.Count > 0, nameof(schema));
            Contracts.CheckValue(pred, nameof(pred));
            // REVIEW: Better names?
            List<int> mapList = new List<int>();
            invMap = new int[schema.Count];
            for (int c = 0; c < schema.Count; ++c)
            {
                if (!pred(schema[c]))
                {
                    invMap[c] = -1;
                    continue;
                }
                invMap[c] = mapList.Count;
                mapList.Add(c);
            }
            map = mapList.ToArray();
        }

        /// <summary>
        /// Given a predicate, over a range of values defined by a limit calculate
        /// first the values for which that predicate was true, and second an inverse
        /// map.
        /// </summary>
        /// <param name="lim">Indicates the exclusive upper bound on the tested values</param>
        /// <param name="pred">The predicate to test for various value</param>
        /// <param name="map">An ascending array of values from 0 inclusive
        /// to <paramref name="lim"/> exclusive, holding all values for which
        /// <paramref name="pred"/> is true</param>
        /// <param name="invMap">Forms an inverse mapping of <paramref name="map"/>,
        /// so that <c><paramref name="invMap"/>[<paramref name="map"/>[i]] == i</c>,
        /// and for other entries not appearing in <paramref name="map"/>,
        /// <c><paramref name="invMap"/>[i] == -1</c></param>
        public static void BuildSubsetMaps(int lim, Func<int, bool> pred, out int[] map, out int[] invMap)
        {
            Contracts.CheckParam(lim >= 0, nameof(lim));
            Contracts.CheckValue(pred, nameof(pred));
            // REVIEW: Better names?
            List<int> mapList = new List<int>();
            invMap = new int[lim];
            for (int c = 0; c < lim; ++c)
            {
                if (!pred(c))
                {
                    invMap[c] = -1;
                    continue;
                }
                invMap[c] = mapList.Count;
                mapList.Add(c);
            }
            map = mapList.ToArray();
        }

        /// <summary>
        /// Given the columns needed, over a range of values defined by a limit calculate
        /// first the values for which the column is present was true, and second an inverse
        /// map.
        /// </summary>
        /// <param name="lim">Indicates the exclusive upper bound on the tested values</param>
        /// <param name="columnsNeeded">The set of columns the calling component operates on.</param>
        /// <param name="map">An ascending array of values from 0 inclusive
        /// to <paramref name="lim"/> exclusive, holding all values for which
        /// <paramref name="columnsNeeded"/> are present.
        /// (The respective index appears in the <paramref name="columnsNeeded"/> collection).</param>
        /// <param name="invMap">Forms an inverse mapping of <paramref name="map"/>,
        /// so that <c><paramref name="invMap"/>[<paramref name="map"/>[i]] == i</c>,
        /// and for other entries not appearing in <paramref name="map"/>,
        /// <c><paramref name="invMap"/>[i] == -1</c></param>
        public static void BuildSubsetMaps(int lim, IEnumerable<DataViewSchema.Column> columnsNeeded, out int[] map, out int[] invMap)
        {
            Contracts.CheckParam(lim >= 0, nameof(lim));
            Contracts.CheckValue(columnsNeeded, nameof(columnsNeeded));

            // REVIEW: Better names?
            List<int> mapList = new List<int>();
            invMap = invMap = Enumerable.Repeat(-1, lim).ToArray<int>();

            foreach (var col in columnsNeeded)
            {
                Contracts.Check(col.Index < lim);
                invMap[col.Index] = mapList.Count;
                mapList.Add(col.Index);
            }

            map = mapList.ToArray();
        }

        public static T[] Concat<T>(T[] a, T[] b)
        {
            if (a == null)
                return b;
            if (b == null)
                return a;
            if (a.Length == 0)
                return b;
            if (b.Length == 0)
                return a;
            var res = new T[a.Length + b.Length];
            Array.Copy(a, res, a.Length);
            Array.Copy(b, 0, res, a.Length, b.Length);
            return res;
        }

        public static T[] Concat<T>(params T[][] arrays)
        {
            // Total size.
            int size = 0;
            // First non-null.
            T[] nn = null;
            // First non-empty.
            T[] ne = null;
            foreach (var a in arrays)
            {
                if (a == null)
                    continue;
                checked { size += a.Length; }
                if (nn == null)
                    nn = a;
                if (ne == null && size > 0)
                    ne = a;
            }
            Contracts.Assert(nn != null || size == 0);
            Contracts.Assert((ne == null) == (size == 0));

            // If the size is zero, return the first non-null.
            if (size == 0)
                return nn;
            // If there is only one non-empty, return it.
            if (size == ne.Length)
                return ne;

            var res = new T[size];
            int ivDst = 0;
            foreach (var a in arrays)
            {
                int cv = Utils.Size(a);
                if (cv == 0)
                    continue;
                Array.Copy(a, 0, res, ivDst, cv);
                ivDst += cv;
            }
            Contracts.Assert(ivDst == size);
            return res;
        }

        /// <summary>
        /// Resizes the array if necessary, to ensure that it has at least <paramref name="min"/> elements.
        /// </summary>
        /// <param name="array">The array to resize. Can be null.</param>
        /// <param name="min">The minimum number of items the new array must have.</param>
        /// <param name="keepOld">True means that the old array is preserved, if possible (Array.Resize is called). False
        /// means that a new array will be allocated.
        /// </param>
        /// <returns>The new size, that is no less than <paramref name="min"/>.</returns>
        public static int EnsureSize<T>(ref T[] array, int min, bool keepOld = true)
        {
            return EnsureSize(ref array, min, Utils.ArrayMaxSize, keepOld);
        }

        /// <summary>
        /// Resizes the array if necessary, to ensure that it has at least <paramref name="min"/> and at most <paramref name="max"/> elements.
        /// </summary>
        /// <param name="array">The array to resize. Can be null.</param>
        /// <param name="min">The minimum number of items the new array must have.</param>
        /// <param name="max">The maximum number of items the new array can have.</param>
        /// <param name="keepOld">True means that the old array is preserved, if possible (Array.Resize is called). False
        /// means that a new array will be allocated.
        /// </param>
        /// <returns>The new size, that is no less than <paramref name="min"/> and no more that <paramref name="max"/>.</returns>
        public static int EnsureSize<T>(ref T[] array, int min, int max, bool keepOld = true)
            => EnsureSize(ref array, min, max, keepOld, out bool _);

        public static int EnsureSize<T>(ref T[] array, int min, int max, bool keepOld, out bool resized)
        {
            Contracts.CheckParam(min <= max, nameof(max), "min must not exceed max");
            // This code adapted from the private method EnsureCapacity code of List<T>.
            int size = Utils.Size(array);
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

        /// <summary>
        /// Returns the number of set bits in a bit array.
        /// </summary>
        public static int GetCardinality(BitArray bitArray)
        {
            Contracts.CheckValue(bitArray, nameof(bitArray));
            int cnt = 0;
            foreach (bool b in bitArray)
            {
                if (b)
                    cnt++;
            }
            return cnt;
        }

        private static MethodInfo MarshalInvokeCheckAndCreate<TRet>(Type genArg, Delegate func)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, func);
            if (meth.ReturnType != typeof(TRet))
                throw Contracts.ExceptParam(nameof(func), "Cannot be generic on return type");
            return meth;
        }

        // REVIEW: n-argument versions? The multi-column re-application problem?
        // Think about how to address these.

        /// <summary>
        /// Given a generic method with a single type parameter, re-create the generic method on a new type,
        /// then reinvoke the method and return the result. A common pattern throughout the code base is to
        /// have some sort of generic method, whose parameters and return value are, as defined, non-generic,
        /// but whose code depends on some sort of generic type parameter. This utility method exists to make
        /// this common pattern more convenient, and also safer so that the arguments, if any, can be type
        /// checked at compile time instead of at runtime.
        ///
        /// Because it is strongly typed, this can only be applied to methods whose return type
        /// is known at compile time, that is, that do not depend on the type parameter of the method itself.
        /// </summary>
        /// <typeparam name="TRet">The return value</typeparam>
        /// <param name="func">A delegate that should be a generic method with a single type parameter.
        /// The generic method definition will be extracted, then a new method will be created with the
        /// given type parameter, then the method will be invoked.</param>
        /// <param name="genArg">The new type parameter for the generic method</param>
        /// <returns>The return value of the invoked function</returns>
        public static TRet MarshalInvoke<TRet>(Func<TRet> func, Type genArg)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, null);
        }

        /// <summary>
        /// A one-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TRet>(Func<TArg1, TRet> func, Type genArg, TArg1 arg1)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1 });
        }

        /// <summary>
        /// A two-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TRet>(Func<TArg1, TArg2, TRet> func, Type genArg, TArg1 arg1, TArg2 arg2)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2 });
        }

        /// <summary>
        /// A three-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TRet>(Func<TArg1, TArg2, TArg3, TRet> func, Type genArg,
            TArg1 arg1, TArg2 arg2, TArg3 arg3)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3 });
        }

        /// <summary>
        /// A four-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TArg4, TRet>(Func<TArg1, TArg2, TArg3, TArg4, TRet> func,
            Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3, arg4 });
        }

        /// <summary>
        /// A five-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TArg4, TArg5, TRet>(Func<TArg1, TArg2, TArg3, TArg4, TArg5, TRet> func,
            Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4, TArg5 arg5)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3, arg4, arg5 });
        }

        /// <summary>
        /// A six-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TRet>(Func<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TRet> func,
            Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4, TArg5 arg5, TArg6 arg6)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3, arg4, arg5, arg6 });
        }

        /// <summary>
        /// A seven-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TRet>(Func<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TRet> func,
            Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4, TArg5 arg5, TArg6 arg6, TArg7 arg7)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3, arg4, arg5, arg6, arg7 });
        }

        /// <summary>
        /// An eight-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TArg8, TRet>(Func<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TArg8, TRet> func,
            Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4, TArg5 arg5, TArg6 arg6, TArg7 arg7, TArg8 arg8)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 });
        }

        /// <summary>
        /// A nine-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TArg8, TArg9, TRet>(
            Func<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TArg8, TArg9, TRet> func,
            Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4, TArg5 arg5, TArg6 arg6, TArg7 arg7, TArg8 arg8, TArg9 arg9)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 });
        }

        /// <summary>
        /// A ten-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TArg8, TArg9, TArg10, TRet>(
            Func<TArg1, TArg2, TArg3, TArg4, TArg5, TArg6, TArg7, TArg8, TArg9, TArg10, TRet> func,
            Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4, TArg5 arg5, TArg6 arg6, TArg7 arg7, TArg8 arg8, TArg9 arg9, TArg10 arg10)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10 });
        }

        private static MethodInfo MarshalActionInvokeCheckAndCreate(Type genArg, Delegate func)
        {
            Contracts.CheckValue(genArg, nameof(genArg));
            Contracts.CheckValue(func, nameof(func));
            var meth = func.GetMethodInfo();
            Contracts.CheckParam(meth.IsGenericMethod, nameof(func), "Should be generic but is not");
            Contracts.CheckParam(meth.GetGenericArguments().Length == 1, nameof(func),
                "Should have exactly one generic type parameter but does not");
            meth = meth.GetGenericMethodDefinition().MakeGenericMethod(genArg);
            return meth;
        }

        /// <summary>
        /// This is akin to <see cref="MarshalInvoke{TRet}(Func{TRet}, Type)"/>, except applied to
        /// <see cref="Action"/> instead of <see cref="Func{TRet}"/>.
        /// </summary>
        /// <param name="act">A delegate that should be a generic method with a single type parameter.
        /// The generic method definition will be extracted, then a new method will be created with the
        /// given type parameter, then the method will be invoked.</param>
        /// <param name="genArg">The new type parameter for the generic method</param>
        public static void MarshalActionInvoke(Action act, Type genArg)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, act);
            meth.Invoke(act.Target, null);
        }

        /// <summary>
        /// A one-argument version of <see cref="MarshalActionInvoke(Action, Type)"/>.
        /// </summary>
        public static void MarshalActionInvoke<TArg1>(Action<TArg1> act, Type genArg, TArg1 arg1)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, act);
            meth.Invoke(act.Target, new object[] { arg1 });
        }

        /// <summary>
        /// A two-argument version of <see cref="MarshalActionInvoke(Action, Type)"/>.
        /// </summary>
        public static void MarshalActionInvoke<TArg1, TArg2>(Action<TArg1, TArg2> act, Type genArg, TArg1 arg1, TArg2 arg2)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, act);
            meth.Invoke(act.Target, new object[] { arg1, arg2 });
        }

        /// <summary>
        /// A three-argument version of <see cref="MarshalActionInvoke(Action, Type)"/>.
        /// </summary>
        public static void MarshalActionInvoke<TArg1, TArg2, TArg3>(Action<TArg1, TArg2, TArg3> act, Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, act);
            meth.Invoke(act.Target, new object[] { arg1, arg2, arg3 });
        }

        /// <summary>
        /// A four-argument version of <see cref="MarshalActionInvoke(Action, Type)"/>.
        /// </summary>
        public static void MarshalActionInvoke<TArg1, TArg2, TArg3, TArg4>(Action<TArg1, TArg2, TArg3, TArg4> act, Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, act);
            meth.Invoke(act.Target, new object[] { arg1, arg2, arg3, arg4 });
        }

        public static string GetDescription(this Enum value)
        {
            Type type = value.GetType();
            string name = Enum.GetName(type, value);
            if (name != null)
            {
                FieldInfo field = type.GetField(name);
                if (field != null)
                {
                    DescriptionAttribute attr =
                        Attribute.GetCustomAttribute(field,
                            typeof(DescriptionAttribute)) as DescriptionAttribute;
                    if (attr != null)
                    {
                        return attr.Description;
                    }
                }
            }
            return null;
        }

        public static int Count<TSource>(this ReadOnlySpan<TSource> source, Func<TSource, bool> predicate)
        {
            Contracts.CheckValue(predicate, nameof(predicate));

            int result = 0;
            for (int i = 0; i < source.Length; i++)
            {
                if (predicate(source[i]))
                    result++;
            }
            return result;
        }

        public static bool All<TSource>(this ReadOnlySpan<TSource> source, Func<TSource, bool> predicate)
        {
            Contracts.CheckValue(predicate, nameof(predicate));

            for (int i = 0; i < source.Length; i++)
            {
                if (!predicate(source[i]))
                    return false;
            }
            return true;
        }
    }
}
