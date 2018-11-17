// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal static class MatrixTransposeOps
    {
        private const int _block = 32;

        /// <summary>
        /// Swap the first m*n elements within a given array so that, for any
        /// non-negative i and j less than m and n respectively, dst[i*n+j] == src[j*m+i].
        /// This variant of the function is single threaded, and useful in that
        /// case where the workflow is otherwise single threaded.
        /// </summary>
        /// <typeparam name="T">Elements of the array are this type</typeparam>
        /// <param name="src"></param>
        /// <param name="dst">Where to write the transpose. Note that dst cannot be the same as src.</param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static void TransposeSingleThread<T>(T[] src, T[] dst, int m, int n)
        {
            Contracts.AssertValue(src);
            Contracts.AssertValue(dst);
            Contracts.Assert(src != dst, "Transpose in place not supported");
            Contracts.Assert(src.Length <= m * n);
            Contracts.Assert(dst.Length <= m * n);

            T[] work = new T[_block * _block];
            for (int i = 0; i < m; i += _block)
            {
                int iend = Math.Min(m - i, _block);
                for (int j = 0; j < n; j += _block)
                {
                    int jend = Math.Min(n - j, _block);
                    // Copy things over to the work block.
                    int off = j * m + i;
                    for (int jsub = 0; jsub < jend; ++jsub)
                    {
                        int workoff = jsub * _block;
                        for (int isub = 0; isub < iend; ++isub)
                        {
                            //work[jsub * BLOCK + isub] = src[(j + jsub) * m + (i + isub)];
                            work[workoff + isub] = src[off + isub];
                        }
                        off += m;
                    }
                    // Copy the elements of the work block over to B.
                    off = i * n + j;
                    for (int isub = 0; isub < iend; ++isub)
                    {
                        for (int jsub = 0; jsub < jend; ++jsub)
                        {
                            //dst[(i + isub) * n + (j + jsub)] = work[jsub * BLOCK + isub];
                            dst[off + jsub] = work[jsub * _block + isub];
                        }
                        off += n;
                    }
                }
            }
        }

        /// <summary>
        /// Swap the first m*n elements within a given array so that, for any
        /// non-negative i and j less than m and n respectively, dst[i*n+j] == src[j*m+i]
        /// </summary>
        /// <typeparam name="T">Elements of the array are this type</typeparam>
        /// <param name="src">The source elements of the transpose. Must contain at least m*n elements.</param>
        /// <param name="dst">Where to write the transpose. Note that dst cannot be the same as src. Must contain at least m*n elements.</param>
        /// <param name="m">The major index.</param>
        /// <param name="n">The minor index. Elements are currently stored in "m" blocks of "n" items.</param>
        public static void Transpose<T>(T[] src, T[] dst, int m, int n)
        {
            Contracts.AssertValue(src);
            Contracts.AssertValue(dst);
            Contracts.Assert(src != dst, "Transpose in place not supported");
            Contracts.Assert(src.Length <= m * n);
            Contracts.Assert(dst.Length <= m * n);

            MadeObjectPool<T[]> workPool = new MadeObjectPool<T[]>(() => new T[_block * _block]);
            int isteps = (m - 1) / _block + 1;
            int jsteps = (n - 1) / _block + 1;
            IEnumerable<int> jenum = Enumerable.Range(0, jsteps).Select(j => j * _block);
            IEnumerable<int> ienum = Enumerable.Range(0, isteps).Select(i => i * _block);
            IEnumerable<Tuple<int, int>> ijenum = ienum.SelectMany(i => jenum.Select(j => new Tuple<int, int>(i, j)));

            Parallel.ForEach(ijenum, ij =>
            {
                int i = ij.Item1;
                int j = ij.Item2;
                int iend = Math.Min(m - i, _block);
                int jend = Math.Min(n - j, _block);
                T[] work = workPool.Get();
                // Copy things over to the work block.
                int off = j * m + i;
                for (int jsub = 0; jsub < jend; ++jsub)
                {
                    int workoff = jsub * _block;
                    for (int isub = 0; isub < iend; ++isub)
                    {
                        //work[jsub * BLOCK + isub] = a[(j + jsub) * m + (i + isub)];
                        work[workoff + isub] = src[off + isub];
                    }
                    off += m;
                }
                // Copy the elements of the work block over to B.
                off = i * n + j;
                for (int isub = 0; isub < iend; ++isub)
                {
                    for (int jsub = 0; jsub < jend; ++jsub)
                    {
                        //dst[(i + isub) * n + (j + jsub)] = work[jsub * BLOCK + isub];
                        dst[off + jsub] = work[jsub * _block + isub];
                    }
                    off += n;
                }
                workPool.Return(work);
            });
        }

        /// <summary>
        /// Swap the first m*n elements within a given array so that, for any
        /// non-negative i and j less than m and n respectively, b[i*n+j] == a[j*m+i]
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dst">Where to write the transpose. Note that dst cannot be the same as src.</param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static unsafe void Transpose(float* src, float* dst, int m, int n)
        {
            MadeObjectPool<float[]> workPool = new MadeObjectPool<float[]>(() => new float[_block * _block]);
            //T[] work = new T[BLOCK * BLOCK];
            int isteps = (m - 1) / _block + 1;
            int jsteps = (n - 1) / _block + 1;
            IEnumerable<int> jenum = Enumerable.Range(0, jsteps).Select(j => j * _block);
            IEnumerable<int> ienum = Enumerable.Range(0, isteps).Select(i => i * _block);
            IEnumerable<Tuple<int, int>> ijenum = ienum.SelectMany(i => jenum.Select(j => new Tuple<int, int>(i, j)));

            Parallel.ForEach(ijenum, ij =>
            {
                int i = ij.Item1;
                int j = ij.Item2;
                int iend = Math.Min(m - i, _block);
                int jend = Math.Min(n - j, _block);
                float[] work = workPool.Get();
                // Copy things over to the work block.
                float* srcP = src + j * m + i;
                for (int jsub = 0; jsub < jend; ++jsub)
                {
                    for (int isub = 0; isub < iend; ++isub)
                    {
                        // This inner loop is equivalent to the following assignment:
                        // work[j * BLOCK + i] = src[(J + j) * m + (I + i)];
                        work[jsub * _block + isub] = srcP[isub];
                    }
                    srcP += m;
                }
                // Copy the elements of the work block over to B.
                float* dstP = dst + i * n + j;
                for (int isub = 0; isub < iend; ++isub)
                {
                    for (int jsub = 0; jsub < jend; ++jsub)
                    {
                        // This inner loop is equivalent to the following assignment:
                        // dst[(I + i) * n + (J + j)] = work[j * BLOCK + i];
                        dstP[jsub] = work[jsub * _block + isub];
                    }
                    dstP += n;
                }
                workPool.Return(work);
            });
        }

        /// <summary>
        /// Swap the first m*n elements within a given array so that, for any
        /// non-negative i and j less than m and n respectively, b[i*n+j] == a[j*m+i]
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dst">Where to write the transpose. Note that dst cannot be the same as src.</param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        public static unsafe void Transpose(double* src, double* dst, int m, int n)
        {
            MadeObjectPool<double[]> workPool = new MadeObjectPool<double[]>(() => new double[_block * _block]);
            //T[] work = new T[BLOCK * BLOCK];
            int isteps = (m - 1) / _block + 1;
            int jsteps = (n - 1) / _block + 1;
            IEnumerable<int> jenum = Enumerable.Range(0, jsteps).Select(j => j * _block);
            IEnumerable<int> ienum = Enumerable.Range(0, isteps).Select(i => i * _block);
            IEnumerable<Tuple<int, int>> ijenum = ienum.SelectMany(i => jenum.Select(j => new Tuple<int, int>(i, j)));

            Parallel.ForEach(ijenum, ij =>
            {
                int i = ij.Item1;
                int j = ij.Item2;
                int iend = Math.Min(m - i, _block);
                int jend = Math.Min(n - j, _block);
                double[] work = workPool.Get();
                // Copy things over to the work block.
                double* srcP = src + j * m + i;
                for (int jsub = 0; jsub < jend; ++jsub)
                {
                    for (int isub = 0; isub < iend; ++isub)
                    {
                        // This inner loop is equivalent to the following assignment:
                        // work[j * BLOCK + i] = src[(J + j) * m + (I + i)];
                        work[jsub * _block + isub] = srcP[isub];
                    }
                    srcP += m;
                }
                // Copy the elements of the work block over to B.
                double* dstP = dst + i * n + j;
                for (int isub = 0; isub < iend; ++isub)
                {
                    for (int jsub = 0; jsub < jend; ++jsub)
                    {
                        // This inner loop is equivalent to the following assignment:
                        // dst[(I + i) * n + (J + j)] = work[j * BLOCK + i];
                        dstP[jsub] = work[jsub * _block + isub];
                    }
                    dstP += n;
                }
                workPool.Return(work);
            });
        }
    }
}
