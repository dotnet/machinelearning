﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;

namespace Microsoft.ML.Trainers.FastTree
{
    [BestFriend]
    internal class VectorUtils
    {
        public static double GetVectorSize(double[] vector)
        {
            double sum = GetDotProduct(vector, vector);
            sum /= Math.Sqrt(sum);
            return sum;
        }

        // Normalizes the vector to have size of 1
        public static void NormalizeVectorSize(double[] vector)
        {
            double size = GetVectorSize(vector);
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] /= size;
            }
        }

        // Center vector to have mean = 0
        public static void CenterVector(double[] vector)
        {
            double mean = GetMean(vector);
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] -= mean;
            }
        }

        // Normalizes the vector to have mean = 0 and std = 1
        public static void NormalizeVector(double[] vector)
        {
            double mean = GetMean(vector);
            double std = GetStandardDeviation(vector, mean);
            NormalizeVector(vector, mean, std);
        }

        // Normalizes the vector to have mean = 0 and std = 1
        public static void NormalizeVector(double[] vector, double mean, double std)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = (vector[i] - mean) / std;
            }
        }

        public static double GetDotProduct(double[] vector1, double[] vector2)
        {
            return GetDotProduct(vector1, vector2, vector1.Length);
        }

        public static double GetDotProduct(float[] vector1, float[] vector2)
        {
            return GetDotProduct(vector1, vector2, vector1.Length);
        }

        public static unsafe double GetDotProduct(double[] vector1, double[] vector2, int length)
        {
            double product = 0;
            unsafe
            {
                fixed (double* pVector1 = vector1)
                fixed (double* pVector2 = vector2)
                {
                    for (int i = 0; i < length; i++)
                    {
                        product += pVector1[i] * pVector2[i];
                    }
                }
            }
            return product;
        }

        public static unsafe double GetDotProduct(float[] vector1, float[] vector2, int length)
        {
            double product = 0;
            unsafe
            {
                fixed (float* pVector1 = vector1)
                fixed (float* pVector2 = vector2)
                {
                    for (int i = 0; i < length; i++)
                    {
                        product += pVector1[i] * pVector2[i];
                    }
                }
            }
            return product;
        }

        public static double GetMean(double[] vector)
        {
            double sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i];
            }
            return sum / vector.Length;
        }

        public static double GetMean(float[] vector)
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

        public static double GetStandardDeviation(double[] vector, double mean)
        {
            double sum = 0;
            double tmp;
            for (int i = 0; i < vector.Length; i++)
            {
                tmp = vector[i] - mean;
                sum += tmp * tmp;
            }
            return Math.Sqrt(sum / vector.Length);
        }

        public static int GetIndexOfMax(double[] vector)
        {
            double max = vector[0];
            int maxIdx = 0;
            for (int i = 1; i < vector.Length; i++)
            {
                if (vector[i] > max)
                {
                    max = vector[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        // Subtracts the second vector from the first one (vector1[i] -= vector2[i])
        public static unsafe void SubtractInPlace(double[] vector1, double[] vector2)
        {
            int length = vector1.Length;
            unsafe
            {
                fixed (double* pVector1 = vector1)
                fixed (double* pVector2 = vector2)
                {
                    for (int i = 0; i < length; i++)
                    {
                        pVector1[i] -= pVector2[i];
                    }
                }
            }
        }

        public static unsafe double[] Subtract(double[] vector1, double[] vector2)
        {
            int length = vector1.Length;
            double[] result = new double[length];
            unsafe
            {
                fixed (double* pResult = result)
                fixed (double* pVector1 = vector1)
                fixed (double* pVector2 = vector2)
                {
                    for (int i = 0; i < length; i++)
                    {
                        pResult[i] = pVector1[i] - pVector2[i];
                    }
                }
            }
            return result;
        }

        // Subtracts the second vector from the first one (vector1[i] += vector2[i])
        public static unsafe void AddInPlace(double[] vector1, double[] vector2)
        {
            int length = vector1.Length;
            unsafe
            {
                fixed (double* pVector1 = vector1)
                fixed (double* pVector2 = vector2)
                {
                    for (int i = 0; i < length; i++)
                    {
                        pVector1[i] += pVector2[i];
                    }
                }
            }
        }

        // Mutiplies the second vector from the first one (vector1[i] /= val)
        public static void MutiplyInPlace(double[] vector, double val)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= val;
            }
        }

        // Divides the second vector from the first one (vector1[i] /= val)
        public static void DivideInPlace(double[] vector, double val)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] /= val;
            }
        }

        // Divides the second vector from the first one (vector1[i] /= val)
        public static void DivideInPlace(float[] vector, float val)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] /= val;
            }
        }

        public static unsafe double GetEuclideanDistance(double[] vector1, double[] vector2)
        {
            double sum = 0;
            double diff;
            int length = vector1.Length;
            unsafe
            {
                fixed (double* pVector1 = vector1)
                fixed (double* pVector2 = vector2)
                {
                    for (int i = 0; i < length; i++)
                    {
                        diff = pVector1[i] - pVector2[i];
                        sum += diff * diff;
                    }
                }
            }
            return Math.Sqrt(sum);
        }

        public static double[][] AllocateDoubleMatrix(int m, int n)
        {
            double[][] mat = new double[m][];
            for (int i = 0; i < m; i++)
            {
                mat[i] = new double[n];
            }
            return mat;
        }

        public static string ToString(double[] vector)
        {
            StringBuilder sb = new StringBuilder();
            for (int f = 0; f < vector.Length; f++)
            {
                if (f > 0)
                {
                    sb.Append(", ");
                }
                sb.Append(vector[f]);
            }
            return sb.ToString();
        }
    }
}
