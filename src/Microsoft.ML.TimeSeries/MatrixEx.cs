using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    internal static class MatrixEx
    {
        /// <summary>
        /// Calculate the inverse of a matrix
        /// Please make sure that:
        /// 1. the input matrix is a square matrix;
        /// 2. the input matrix is not singular.
        /// </summary>
        /// <param name="matrix"> the input matrix</param>
        /// <returns>the inverse of a matrix</returns>
        public static double[,] ReverseMatrix(this double[,] matrix)
        {
            if (matrix.GetLength(0) != matrix.GetLength(1))
                return null;
            int level = matrix.GetLength(0);

            // Calculate the determinant value of the matrix
            double determinantMatrixValue = MatrixValue(matrix, level);
            if (determinantMatrixValue == 0)
                return null;

            double[,] reverseMatrix = new double[level, 2 * level];
            double x;
            double c;

            // Init Reverse matrix
            for (int i = 0; i < level; i++)
            {
                for (int j = 0; j < 2 * level; j++)
                {
                    if (j < level)
                        reverseMatrix[i, j] = matrix[i, j];
                    else
                        reverseMatrix[i, j] = 0;
                }

                reverseMatrix[i, level + i] = 1;
            }

            for (int i = 0, j = 0; i < level && j < level; i++, j++)
            {
                if (reverseMatrix[i, j] == 0)
                {
                    int m = i;
                    for (; matrix[m, j] == 0; m++)
                    {
                    }

                    if (m == level)
                    {
                        return null;
                    }
                    else
                    {
                        // Add i-row with m-row
                        for (int n = j; n < 2 * level; n++)
                            reverseMatrix[i, n] += reverseMatrix[m, n];
                    }
                }

                // Format the i-row with "1" start
                x = reverseMatrix[i, j];
                if (x != 1)
                {
                    for (int n = j; n < 2 * level; n++)
                    {
                        if (reverseMatrix[i, n] != 0)
                            reverseMatrix[i, n] /= x;
                    }
                }

                // Set 0 to the current column in the rows after current row
                for (int s = level - 1; s > i; s--)
                {
                    x = reverseMatrix[s, j];
                    for (int t = j; t < 2 * level; t++)
                        reverseMatrix[s, t] -= reverseMatrix[i, t] * x;
                }
            }

            // Format the first matrix into unit-matrix
            for (int i = level - 2; i >= 0; i--)
            {
                for (int j = i + 1; j < level; j++)
                {
                    if (reverseMatrix[i, j] != 0)
                    {
                        c = reverseMatrix[i, j];
                        for (int n = j; n < 2 * level; n++)
                            reverseMatrix[i, n] -= c * reverseMatrix[j, n];
                    }
                }
            }

            double[,] result = new double[level, level];
            for (int i = 0; i < level; i++)
            {
                for (int j = 0; j < level; j++)
                    result[i, j] = reverseMatrix[i, j + level];
            }
            return result;
        }

        /// <summary>
        /// Calculate the determinant value of a matrix
        /// </summary>
        public static double MatrixValue(double[,] matrixList, int level)
        {
            double[,] matrix = new double[level, level];
            for (int i = 0; i < level; i++)
            {
                for (int j = 0; j < level; j++)
                    matrix[i, j] = matrixList[i, j];
            }
            double c;
            double x;
            int k = 1;
            for (int i = 0, j = 0; i < level && j < level; i++, j++)
            {
                if (matrix[i, j] == 0)
                {
                    int m = i;
                    for (; matrix[m, j] == 0; m++)
                    {
                    }

                    if (m == level)
                    {
                        return 0;
                    }
                    else
                    {
                        // Row change between i-row and m-row
                        for (int n = j; n < level; n++)
                        {
                            c = matrix[i, n];
                            matrix[i, n] = matrix[m, n];
                            matrix[m, n] = c;
                        }

                        // Change value pre-value
                        k *= -1;
                    }
                }

                // Set 0 to the current column in the rows after current row
                for (int s = level - 1; s > i; s--)
                {
                    x = matrix[s, j];
                    for (int t = j; t < level; t++)
                        matrix[s, t] -= matrix[i, t] * (x / matrix[i, j]);
                }
            }

            double sn = 1;
            for (int i = 0; i < level; i++)
            {
                if (matrix[i, i] != 0)
                    sn *= matrix[i, i];
                else
                    return 0;
            }
            return k * sn;
        }
    }
}
