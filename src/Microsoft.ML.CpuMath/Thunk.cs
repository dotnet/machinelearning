// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Security;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal unsafe static class Thunk
    {
        internal const string NativePath = "CpuMathNative";

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern bool ChkAvx();

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulA(bool add, float* pmat, float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulX(bool add, float* pmat, float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulPA(bool add, float* pmat, int* pposSrc, float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulPX(bool add, float* pmat, int* pposSrc, float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulRU(bool add, int* pstarts, int* pindices, float* pcoefs,
            float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulRX(bool add, int* pstarts, int* pindices, float* pcoefs,
            float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulCU(bool add, int* pmprowiv, int* pmprowcol,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulDU(bool add, int* pmprowiv, int* pmprowcol, int* pmprowrun,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulCX(bool add, int* pmprowiv, int* pmprowcol,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulDX(bool add, int* pmprowiv, int* pmprowcol, int* pmprowrun,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MeanU(bool add, int* pmprowcol, int* pmprowindices, int* pindices,
            float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MaxU(bool add, int* pmprowcol, int* pmprowindices, int* pindices,
            float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void RespNormU(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            int* pmprowcol, int* pmprowindices, int* pindices,
            float* psrc, float* pdst, int crow);

        // These treat pmat as if it is stored in column-major order. Thus, crow and ccol are the numbers of rows
        // and columns from that perspective. Alternatively, crow is the number of rows in the transpose of pmat
        // (thought of as row-major order).
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranA(bool add, float* pmat, float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranX(bool add, float* pmat, float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranPA(bool add, float* pmat, int* pposSrc, float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranPX(bool add, float* pmat, int* pposSrc, float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranRU(bool add, int* pstarts, int* pindices, float* pcoefs,
            float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranRX(bool add, int* pstarts, int* pindices, float* pcoefs,
            float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranCU(bool add, int* pmpcoliv, int* pmpcolrow,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranDU(bool add, int* pmpcoliv, int* pmpcolrow, int* pmpcolrun,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranCX(bool add, int* pmpcoliv, int* pmpcolrow,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranDX(bool add, int* pmpcoliv, int* pmpcolrow, int* pmpcolrun,
            int* pruns, float* pcoefs, float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MeanBackU(bool add, int* pmpcolrow, int* pmpcolindices, int* pindices,
            float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MaxBackU(bool add, int* pmpcolrow, int* pmpcolindices, int* pindices,
            float* psrc, float* pdst, float* pval, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void RespNormBackU(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            int* pmpcolrow, int* pmpcolindices, int* pindices,
            float* perrors, float* perrorsPrev, float* pvaluesPrev, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranA(float a, float* px, float* py, float* pmat, int crow, int ccol, float decay);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranX(float a, float* px, float* py, float* pmat, int crow, int ccol, float decay);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranPA(float a, float* px, int* pposY, float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranPX(float a, float* px, int* pposY, float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranRU(float a, float* px, float* py,
            int* pstarts, int* pindices, float* pcoefs, int crow, float decay);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranRX(float a, float* px, float* py,
            int* pstarts, int* pindices, float* pcoefs, int crow, float decay);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranCU(float a, float* px, float* py, int* pmprowiv, int* pmprowcol,
            int* pruns, float* pcoefs, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranDU(float a, float* px, float* py, int* pmprowiv, int* pmprowcol,
            int* pmprowrun, int* pruns, float* pcoefs, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranCX(float a, float* px, float* py, int* pmprowiv, int* pmprowcol,
            int* pruns, float* pcoefs, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranDX(float a, float* px, float* py, int* pmprowiv, int* pmprowcol,
            int* pmprowrun, int* pruns, float* pcoefs, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranMomA(float a, float* px, float* py, float* pmat, float momentum, float* pdel, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranMomX(float a, float* px, float* py, float* pmat, float momentum, float* pdel, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradA(float* px, float* py, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradX(float* px, float* py, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradRU(float* px, float* py, int* pstarts, int* pindices,
            float* pcoefs, float* paccGrads, float* paccUpdates, float decay, float cond, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradRX(float* px, float* py, int* pstarts, int* pindices,
            float* pcoefs, float* paccGrads, float* paccUpdates, float decay, float cond, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradPA(float* px, int* pposY, float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradPX(float* px, int* pposY, float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleU(float a, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleA(float a, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleX(float a, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleSrcU(float a, float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleAddU(float a, float b, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormA(float maxNorm, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormX(float maxNorm, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormTranU(float maxNorm, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormRU(float maxNorm, int* pstarts, float* pmat, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormCU(float maxNorm, int kernCount, int kernSize, float* pmat);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleA(float a, float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleU(float a, float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleX(float a, float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleSU(float a, float* ps, int* pi, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleCopyU(float a, float* ps, float* pd, float* pr, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleMomA(float a, float* ps, float* pd, float momentum, float* pdel, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleMomX(float a, float* ps, float* pd, float momentum, float* pdel, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleGradA(float* ps, float* pd, float* paccGrads, float* paccUpdates,
            float decay, float cond, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleGradX(float* ps, float* pd, float* paccGrads, float* paccUpdates,
            float decay, float cond, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleMultiA(int count, float* ps, float* pd, float* paccGrads,
            float* paccUpdates, float decay, float cond, int size);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScalarU(float a, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddU(float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddA(float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddX(float* ps, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddSU(float* ps, int* pi, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumA(float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumU(float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumX(float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumSqU(float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumSqDiffU(float mean, float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumAbsU(float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumAbsDiffU(float mean, float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MulElementWiseU(float* ps1, float* ps2, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MulElementWiseSU(float* ps1, float* ps2,  int* pi, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MaxAbsU(float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MaxAbsDiffU(float mean, float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float DotU(float* pa, float* pb, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float DotSU(float* pa, float* pb, int* pi, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float Dist2(float* px, float* py, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySigmoidA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySigmoidX(float* ps, float* pd, int c)
        {
            ApplySigmoidA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySoftMaxU(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftMaxA(float* ps, float* pd, int c)
        {
            ApplySoftMaxU(ps, pd, c);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftMaxX(float* ps, float* pd, int c)
        {
            ApplySoftMaxU(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyRectifiedLinearA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyRectifiedLinearX(float* ps, float* pd, int c)
        {
            ApplyRectifiedLinearA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySquareA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySquareX(float* ps, float* pd, int c)
        {
            ApplySquareA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySqrtA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySqrtX(float* ps, float* pd, int c)
        {
            ApplySqrtA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySoftRectifiedLinearU(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftRectifiedLinearA(float* ps, float* pd, int c)
        {
            ApplySoftRectifiedLinearU(ps, pd, c);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftRectifiedLinearX(float* ps, float* pd, int c)
        {
            ApplySoftRectifiedLinearU(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyAbsA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyAbsX(float* ps, float* pd, int c)
        {
            ApplyAbsA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyTanhA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyTanhX(float* ps, float* pd, int c)
        {
            ApplyTanhA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyBoundedRectifiedLinearA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyBoundedRectifiedLinearX(float* ps, float* pd, int c)
        {
            ApplyBoundedRectifiedLinearA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySigmoidDerivativeA(float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySigmoidDerivativeX(float* pv, float* pg, int c)
        {
            ApplySigmoidDerivativeA(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyRectifiedLinearDerivativeA(float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyRectifiedLinearDerivativeX(float* pv, float* pg, int c)
        {
            ApplyRectifiedLinearDerivativeA(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySquareDerivativeA(float* px, float* py, float* pg, int c, bool drop);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySquareDerivativeX(float* px, float* py, float* pg, int c, bool drop)
        {
            ApplySquareDerivativeA(px, py, pg, c, drop);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySqrtDerivativeA(float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySqrtDerivativeX(float* pv, float* pg, int c)
        {
            ApplySqrtDerivativeA(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySoftRectifiedLinearDerivativeU(float* px, float* py, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftRectifiedLinearDerivativeA(float* px, float* py, float* pg, int c)
        {
            ApplySoftRectifiedLinearDerivativeU(px, py, pg, c);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftRectifiedLinearDerivativeX(float* px, float* py, float* pg, int c)
        {
            ApplySoftRectifiedLinearDerivativeU(px, py, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyAbsDerivativeA(float* px, float* py, float* pg, int c, bool drop);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyAbsDerivativeX(float* px, float* py, float* pg, int c, bool drop)
        {
            ApplyAbsDerivativeA(px, py, pg, c, drop);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyTanhDerivativeA(float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyTanhDerivativeX(float* pv, float* pg, int c)
        {
            ApplyTanhDerivativeA(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyBoundedRectifiedLinearDerivativeA(float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyBoundedRectifiedLinearDerivativeX(float* pv, float* pg, int c)
        {
            ApplyBoundedRectifiedLinearDerivativeA(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ZeroItemsU(float* pd, int c, int* pindices, int cindices);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ZeroMatrixItemsCore(float* pd, int c, int ccol, int cfltRow, int* pindices, int cindices);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void SdcaL1UpdateU(float primalUpdate, float* ps, float threshold, float* pd1, float* pd2, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void SdcaL1UpdateSU(float primalUpdate, float* ps, int* pi, float threshold, float* pd1, float* pd2, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleAdadeltaU(float* mat, float* accGrads, float* accUpdates, float decay, float cond, float* grads, int size);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleAdadeltaA(float* mat, float* accGrads, float* accUpdates, float decay, float cond, float* grads, int size);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleAdadeltaX(float* mat, float* accGrads, float* accUpdates, float decay, float cond, float* grads, int size);

#if !CORECLR
        // In CoreCLR we use Buffer.MemoryCopy directly instead of
        // plumbing our own version.
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MemCpy(void* dst, void* src, long count);
#endif
    }
}
