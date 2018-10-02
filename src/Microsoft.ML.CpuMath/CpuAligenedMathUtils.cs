// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static class CpuAligenedMathUtils<TMatrix>
        where TMatrix : CpuAlignedMatrixBase, ICpuFullMatrix
    {
        /// <summary>
        /// Assert the compatibility of the underlying AlignedArray for the input matrix in terms of alignment amount.
        /// </summary>
        /// <param name="values">The input matrix</param>
        public static void AssertCompatible(ICpuFullMatrix values)
        {
#if DEBUG
            var mat = values as TMatrix;
            Contracts.AssertValue(mat);
            Contracts.Assert((mat.Items.CbAlign % CpuMathUtils.GetVectorAlignment()) == 0);
#endif
        }

        /// <summary>
        /// Assert the compatibility of the underlying AlignedArray for the input vector in terms of alignment amount.
        /// </summary>
        /// <param name="values">The input vector</param>
        public static void AssertCompatible(ICpuVector values)
        {
#if DEBUG
            CpuAlignedVector vec = values as CpuAlignedVector;
            Contracts.AssertValue(vec);
            Contracts.Assert((vec.Items.CbAlign % CpuMathUtils.GetVectorAlignment()) == 0);
#endif
        }

        private static TMatrix A(ICpuFullMatrix x)
        {
            AssertCompatible(x);
            return (TMatrix)x;
        }

        private static CpuAlignedVector A(ICpuVector x)
        {
            AssertCompatible(x);
            return (CpuAlignedVector)x;
        }

        private static void AssertCompatibleCore(ICpuMatrix mat, ICpuVector src, ICpuVector dst)
        {
            AssertCompatible(src);
            AssertCompatible(dst);
            Contracts.Assert(mat.ColCount == src.VectorSize);
            Contracts.Assert(mat.RowCount == dst.VectorSize);
        }

        /// <summary>
        /// Asserts the following:
        /// 1. The compatibility of the underlying AlignedArray for mat in terms of alignment amount.
        /// 2. The compatibility of the underlying AlignedArray for src in terms of alignment amount.
        /// 3. The compatibility of the underlying AlignedArray for dst in terms of alignment amount.
        /// 4. The compatibility of the matrix-vector multiplication mat * src = dst.
        /// </summary>
        /// <param name="mat"></param>
        /// <param name="src"></param>
        /// <param name="dst"></param>
        public static void AssertCompatible(ICpuFullMatrix mat, ICpuVector src, ICpuVector dst)
        {
            // Also check the physical sizes.
            AssertCompatible(mat);
            AssertCompatibleCore(mat, src, dst);
            var m = A(mat);
            Contracts.Assert(m.ColCountPhy == A(src).Items.Size);
            Contracts.Assert(m.RowCountPhy == A(dst).Items.Size);
        }

        /// <summary>
        /// Matrix multiplication:
        /// if (add)
        ///     dst = mat * src
        /// else
        ///     dest += mat * src
        /// </summary>
        /// <param name="add">The addition flag</param>
        /// <param name="mat">The multiplier matrix</param>
        /// <param name="src">The source vector</param>
        /// <param name="dst">The destination vector</param>
        public static void MatTimesSrc(bool add, ICpuFullMatrix mat, ICpuVector src, ICpuVector dst)
        {
            bool colMajor = typeof(TMatrix) == typeof(CpuAlignedMatrixCol);
            AssertCompatible(mat, src, dst);
            var m = A(mat);
            CpuMathUtils.MatTimesSrc(colMajor, add, m.Items, A(src).Items, A(dst).Items, m.RunCnt);
        }

        /// <summary>
        /// Matrix transpose multiplication:
        /// if (add)
        ///     dst = mat' * src
        /// else
        ///     dest += mat' * src
        /// </summary>
        /// <param name="add">The addition flag</param>
        /// <param name="mat">The multiplier matrix</param>
        /// <param name="src">The source vector</param>
        /// <param name="dst">The destination vector</param>
        public static void MatTranTimesSrc(bool add, ICpuFullMatrix mat, ICpuVector src, ICpuVector dst)
        {
            bool colMajor = typeof(TMatrix) == typeof(CpuAlignedMatrixCol);
            AssertCompatible(mat, dst, src);
            var m = A(mat);
            CpuMathUtils.MatTimesSrc(!colMajor, add, m.Items, A(src).Items, A(dst).Items, m.RunCnt);
        }
    }

    public static class GeneralUtils
    {
        /// <summary>
        /// Count the number of zero bits in the lonest string of zero's from the lowest significant bit of the input integer.
        /// </summary>
        /// <param name="u">The input integer</param>
        /// <returns></returns>
        public static int CbitLowZero(uint u)
        {
            if (u == 0)
                return 32;

            int cbit = 0;
            if ((u & 0x0000FFFF) == 0)
            {
                cbit += 16;
                u >>= 16;
            }
            if ((u & 0x000000FF) == 0)
            {
                cbit += 8;
                u >>= 8;
            }
            if ((u & 0x0000000F) == 0)
            {
                cbit += 4;
                u >>= 4;
            }
            if ((u & 0x00000003) == 0)
            {
                cbit += 2;
                u >>= 2;
            }
            if ((u & 0x00000001) == 0)
                cbit += 1;
            return cbit;
        }
    }
}
