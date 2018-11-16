// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public static class LinqExtensions
    {
        public static int ArgMin<T>(this T[] arr) where T : IComparable<T>
        {
            if (arr.Length == 0)
                return -1;
            int argMin = 0;
            for (int i = 1; i < arr.Length; i++)
            {
                if (arr[i].CompareTo(arr[argMin]) < 0)
                    argMin = i;
            }
            return argMin;
        }

        public static int ArgMax<T>(this T[] arr) where T : IComparable<T>
        {
            if (arr.Length == 0)
                return -1;
            int argMax = 0;
            for (int i = 1; i < arr.Length; i++)
            {
                if (arr[i].CompareTo(arr[argMax]) > 0)
                    argMax = i;
            }
            return argMax;
        }

        public static int ArgMin<T>(this T[] arr, int prefix) where T : IComparable<T>
        {
            int length = arr.Length < prefix ? arr.Length : prefix;
            if (length == 0)
                return -1;
            int argMin = 0;
            for (int i = 1; i < length; i++)
            {
                if (arr[i].CompareTo(arr[argMin]) < 0)
                    argMin = i;
            }
            return argMin;
        }

        public static int ArgMax<T>(this T[] arr, int prefix) where T : IComparable<T>
        {
            int length = arr.Length < prefix ? arr.Length : prefix;
            if (length == 0)
                return -1;
            int argMax = 0;
            for (int i = 1; i < length; i++)
            {
                if (arr[i].CompareTo(arr[argMax]) > 0)
                    argMax = i;
            }
            return argMax;
        }

        public static int ArgMax<T>(this IEnumerable<T> e) where T : IComparable<T>
        {
            T max = e.First();
            int argMax = 0;
            int i = 1;
            foreach (T d in e.Skip(1))
            {
                if (d.CompareTo(max) > 0)
                {
                    argMax = i;
                    max = d;
                }
                ++i;
            }
            return argMax;
        }

        public static int ArgMaxRand<T>(this IEnumerable<T> e, Random rnd, double fraction) where T : IComparable<T>
        {
            T max = e.First();
            int argMax = 0;
            int i = 1;
            foreach (T d in e.Skip(1))
            {
                if (d.CompareTo(max) > 0 && rnd.NextDouble() < fraction)
                {
                    argMax = i;
                    max = d;
                }
                ++i;
            }
            return argMax;
        }

        public static int ArgMax<T>(this IEnumerable<T> e, int prefix) where T : IComparable<T>
        {
            if (prefix <= 1)
                return 0;

            T max = e.First();
            int argMax = 0;
            int i = 0;
            foreach (T d in e)
            {
                if (i == prefix)
                    break;

                if (d.CompareTo(max) > 0)
                {
                    argMax = i;
                    max = d;
                }
                ++i;
            }
            return argMax;
        }

        public static int ArgMaxRand<T>(this IEnumerable<T> e, int prefix, Random rnd, double fraction) where T : IComparable<T>
        {
            if (prefix <= 1)
                return 0;

            T max = e.First();
            int argMax = 0;
            int i = 0;
            foreach (T d in e)
            {
                if (i == prefix)
                    break;

                if (d.CompareTo(max) > 0 && rnd.NextDouble() < fraction)
                {
                    argMax = i;
                    max = d;
                }
                ++i;
            }
            return argMax;
        }

        public static int ArgMin<T>(this IEnumerable<T> e) where T : IComparable<T>
        {
            T max = e.First();
            int argMin = 0;
            int i = 0;
            foreach (T d in e)
            {
                if (d.CompareTo(max) < 0)
                {
                    argMin = i;
                    max = d;
                }
                ++i;
            }
            return argMin;
        }

        public static int ArgMin<T>(this IEnumerable<T> e, int prefix) where T : IComparable<T>
        {
            if (prefix <= 1)
                return 0;

            T max = e.First();
            int argMin = 0;
            int i = 0;
            foreach (T d in e)
            {
                if (i == prefix)
                    break;

                if (d.CompareTo(max) < 0)
                {
                    argMin = i;
                    max = d;
                }
                ++i;
            }
            return argMin;
        }

        // More efficient ToArray pre-allocates the length of array neccessary
        //  Will truncate the IEnumerable at the given length.
        public static T[] ToArray<T>(this IEnumerable<T> me, int length)
        {
            T[] items = new T[length];
            int itemsIndex = 0;
            foreach (T item in me)
            {
                items[itemsIndex++] = item;
                if (itemsIndex >= length)       // OPTIMIZE: Could have a separate routine that doesn't do this, for efficiency
                    break;
            }
            return items;
        }

        /// <summary>
        /// RunningLength. Converts sequence like 1, 2, 3, 4
        /// to  1, 3, 6, 10
        /// </summary>
        public static IEnumerable<int> CumulativeSum<T>(this IEnumerable<int> s)
        {
            int sum = 0;
            ;
            foreach (var x in s)
            {
                sum = sum + x;
                yield return sum;
            }
        }

        //Merges 2 sorted lists in an ascending order
        public static IEnumerable<T> MergeSortedList<T>(this IEnumerable<T> s1, IEnumerable<T> s2) where T : IComparable<T>
        {
            var e1 = s1.GetEnumerator();
            var e2 = s2.GetEnumerator();

            bool moreE1 = e1.MoveNext();
            bool moreE2 = e2.MoveNext();

            while (moreE1 && moreE2)
            {
                if (e1.Current.CompareTo(e2.Current) <= 0)
                {
                    yield return e1.Current;
                    moreE1 = e1.MoveNext();
                }
                else
                {
                    yield return e2.Current;
                    moreE2 = e2.MoveNext();
                }
            }
            while (moreE1)
            {
                yield return e1.Current;
                moreE1 = e1.MoveNext();
            }
            while (moreE2)
            {
                yield return e2.Current;
                moreE2 = e2.MoveNext();
            }
        }

        public static int SoftArgMax(this IEnumerable<double> values, Random rand)
        {
            int len = 0;
            double max = double.NegativeInfinity;
            foreach (double value in values)
            {
                ++len;
                if (value > max)
                    max = value;
            }

            if (len == 0)
                return -1;
            else if (double.IsNegativeInfinity(max))
            {
                lock (rand)
                    return rand.Next(len);
            }

            double total = values.Sum(value => Math.Exp(value - max));

            // Loop just in case due to roundoff we don't choose anything in first pass -- very unlikely.
            for (; ; )
            {
                double r;
                lock (rand)
                    r = rand.NextDouble() * total;

                int i = 0;
                foreach (double value in values)
                {
                    r -= Math.Exp(value - max);
                    if (r <= 0)
                        return i;
                    ++i;
                }
            }
        }
    }
}
