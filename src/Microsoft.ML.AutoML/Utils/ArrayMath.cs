// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.AutoML
{
    public class ArrayMath
    {
        /* x + y */
        public static double[] Add(double[] xArray, double y)
        {
            return xArray.Select(x => x + y).ToArray();
        }

        /* np.argmax */
        public static int ArgMax(double[] array)
        {
            int index = 0;
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] > array[index])
                {
                    index = i;
                }
            }

            return index;
        }

        /* np.argsort */
        public static int[] ArgSort(double[] array)
        {
            return Enumerable.Range(0, array.Length).OrderBy(index => array[index]).ToArray();
        }

        /* np.clip */
        public static double[] Clip(double[] xArray, double min, double max)
        {
            return xArray.Select(x => Math.Min(Math.Max(x, min), max)).ToArray();
        }

        /* x / y */
        public static double[] Div(double[] xArray, double y)
        {
            return xArray.Select(x => x / y).ToArray();
        }

        /* x[y] */
        public static double[] Index(double[] array, int[] indices)
        {
            return indices.Select(index => array[index]).ToArray();
        }

        /* List.Insert */
        public static double[] Insert(double[] array, int index, double item)
        {
            double[] ret = new double[array.Length + 1];
            Array.Copy(array, 0, ret, 0, index);
            ret[index] = item;
            Array.Copy(array, index, ret, index + 1, array.Length - index);
            return ret;
        }

        /* np.linalg.norm */
        public static double Norm(double[] array)
        {
            double s = 0;
            foreach (double x in array)
            {
                s += x * x;
            }

            return Math.Sqrt(s);
        }

        /* x * y */
        public static double[] Mul(double[] xArray, double y)
        {
            return xArray.Select(x => x * y).ToArray();
        }

        /* np.log */
        public static double[] Log(double[] xArray)
        {
            return xArray.Select(x => Math.Log(x)).ToArray();
        }

        /* x * y */
        public static double[] Mul(double[] xArray, double[] yArray)
        {
            return Enumerable.Zip(xArray, yArray, (x, y) => x * y).ToArray();
        }

        /* np.searchsorted */
        public static int SearchSorted(double[] array, double item)
        {
            int index = Array.BinarySearch(array, item);
            return index >= 0 ? index : ~index;
        }

        /* x - y */
        public static double[] Add(double[] xArray, double[] yArray)
        {
            return Enumerable.Zip(xArray, yArray, (x, y) => x + y).ToArray();
        }

        public static double[] Sub(double[] xArray, double[] yArray)
        {
            return Add(xArray, Mul(yArray, -1));
        }

        public static double[] Inverse(double[] array)
        {
            return array.Select(v => 1 / v).ToArray();
        }

        public static double[] Normalize(double[] array)
        {
            var sum = array.Sum();
            return array.Select(v => v / sum).ToArray();
        }

        public static double Rmse(double[] truth, double[] pred)
        {
            if (truth.Length != pred.Length)
            {
                throw new ArgumentException($"length doesn't match, {truth.Length} != {pred.Length}");
            }

            var diff = Enumerable.Range(0, truth.Length).Select(i => truth[i] - pred[i]).ToArray();
            var sqaure = diff.Select(x => x * x);
            var mean = sqaure.Average();
            var rmse = Math.Sqrt(mean);

            return rmse;
        }

        public static double Mape(double[] truth, double[] pred)
        {
            if (truth.Length != pred.Length)
            {
                throw new ArgumentException($"length doesn't match, {truth.Length} != {pred.Length}");
            }

            var diff = Enumerable.Range(0, truth.Length).Select(i => truth[i] - pred[i]).ToArray();
            var ape = diff.Select((x, i) => Math.Abs(x) / truth[i]);
            var mape = ape.Average();

            return mape;
        }

        public static double Mae(double[] truth, double[] pred)
        {
            if (truth.Length != pred.Length)
            {
                throw new ArgumentException($"length doesn't match, {truth.Length} != {pred.Length}");
            }

            var diff = Enumerable.Range(0, truth.Length).Select(i => Math.Abs(truth[i] - pred[i])).ToArray();
            var mae = diff.Average();

            return mae;
        }
    }
}
