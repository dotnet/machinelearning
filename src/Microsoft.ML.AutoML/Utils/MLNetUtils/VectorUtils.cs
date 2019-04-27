// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML
{
    internal static class VectorUtils
    {
        public static double GetMean(double[] vector)
        {
            double sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i];
            }
            return sum / vector.Length;
        }

        public static double GetStandardDeviation(double[] vector)
        {
            return GetStandardDeviation(vector, GetMean(vector));
        }

        private static double GetStandardDeviation(double[] vector, double mean)
        {
            double sum = 0;
            int length = vector.Length;
            double tmp;
            for (int i = 0; i < length; i++)
            {
                tmp = vector[i] - mean;
                sum += tmp * tmp;
            }
            return Math.Sqrt(sum / length);
        }
    }
}
