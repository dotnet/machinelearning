// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Security;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    internal static unsafe class Thunk
    {
        internal const string NativePath = "CpuMathNative";

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern bool ChkAvx();

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMul(bool add, /*const*/ float* pmat, /*const*/ float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulX(bool add, /*const*/ float* pmat, /*const*/ float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulP(bool add, /*const*/ float* pmat, /*const*/ int* pposSrc, /*const*/ float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulPX(bool add, /*const*/ float* pmat, /*const*/ int* pposSrc, /*const*/ float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulRU(bool add, /*const*/ int* pstarts, /*const*/ int* pindices, /*const*/ float* pcoefs,
            /*const*/ float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulRX(bool add, /*const*/ int* pstarts, /*const*/ int* pindices, /*const*/ float* pcoefs,
            /*const*/ float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulCU(bool add, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulDU(bool add, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol, /*const*/ int* pmprowrun,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulCX(bool add, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulDX(bool add, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol, /*const*/ int* pmprowrun,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MeanU(bool add, /*const*/ int* pmprowcol, /*const*/ int* pmprowindices, /*const*/ int* pindices,
            /*const*/ float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MaxU(bool add, /*const*/ int* pmprowcol, /*const*/ int* pmprowindices, /*const*/ int* pindices,
            /*const*/ float* psrc, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void RespNormU(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            /*const*/ int* pmprowcol, /*const*/ int* pmprowindices, /*const*/ int* pindices,
            /*const*/ float* psrc, float* pdst, int crow);

        // These treat pmat as if it is stored in column-major order. Thus, crow and ccol are the numbers of rows
        // and columns from that perspective. Alternatively, crow is the number of rows in the transpose of pmat
        // (thought of as row-major order).
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTran(bool add, /*const*/ float* pmat, /*const*/ float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranX(bool add, /*const*/ float* pmat, /*const*/ float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranP(bool add, /*const*/ float* pmat, /*const*/ int* pposSrc, /*const*/ float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranPX(bool add, /*const*/ float* pmat, /*const*/ int* pposSrc, /*const*/ float* psrc,
            int posMin, int iposMin, int iposLim, float* pdst, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranRU(bool add, /*const*/ int* pstarts, /*const*/ int* pindices, /*const*/ float* pcoefs,
            /*const*/ float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranRX(bool add, /*const*/ int* pstarts, /*const*/ int* pindices, /*const*/ float* pcoefs,
            /*const*/ float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranCU(bool add, /*const*/ int* pmpcoliv, /*const*/ int* pmpcolrow,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranDU(bool add, /*const*/ int* pmpcoliv, /*const*/ int* pmpcolrow, /*const*/ int* pmpcolrun,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranCX(bool add, /*const*/ int* pmpcoliv, /*const*/ int* pmpcolrow,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MatMulTranDX(bool add, /*const*/ int* pmpcoliv, /*const*/ int* pmpcolrow, /*const*/ int* pmpcolrun,
            /*const*/ int* pruns, /*const*/ float* pcoefs, /*const*/ float* psrc, float* pdst, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MeanBackU(bool add, /*const*/ int* pmpcolrow, /*const*/ int* pmpcolindices, /*const*/ int* pindices,
            /*const*/ float* psrc, float* pdst, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MaxBackU(bool add, /*const*/ int* pmpcolrow, /*const*/ int* pmpcolindices, /*const*/ int* pindices,
            /*const*/ float* psrc, float* pdst, /*const*/ float* pval, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void RespNormBackU(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            /*const*/ int* pmpcolrow, /*const*/ int* pmpcolindices, /*const*/ int* pindices,
            /*const*/ float* perrors, float* perrorsPrev, /*const*/ float* pvaluesPrev, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTran(float a, /*const*/ float* px, /*const*/ float* py, float* pmat, int crow, int ccol, float decay);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranX(float a, /*const*/ float* px, /*const*/ float* py, float* pmat, int crow, int ccol, float decay);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranP(float a, /*const*/ float* px, /*const*/ int* pposY, /*const*/ float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranPX(float a, /*const*/ float* px, /*const*/ int* pposY, /*const*/ float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranRU(float a, /*const*/ float* px, /*const*/ float* py,
            /*const*/ int* pstarts, /*const*/ int* pindices, float* pcoefs, int crow, float decay);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranRX(float a, /*const*/ float* px, /*const*/ float* py,
            /*const*/ int* pstarts, /*const*/ int* pindices, float* pcoefs, int crow, float decay);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranCU(float a, /*const*/ float* px, /*const*/ float* py, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol,
            /*const*/ int* pruns, float* pcoefs, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranDU(float a, /*const*/ float* px, /*const*/ float* py, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol,
            /*const*/ int* pmprowrun, /*const*/ int* pruns, float* pcoefs, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranCX(float a, /*const*/ float* px, /*const*/ float* py, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol,
            /*const*/ int* pruns, float* pcoefs, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranDX(float a, /*const*/ float* px, /*const*/ float* py, /*const*/ int* pmprowiv, /*const*/ int* pmprowcol,
            /*const*/ int* pmprowrun, /*const*/ int* pruns, float* pcoefs, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranMom(float a, /*const*/ float* px, /*const*/ float* py, float* pmat, float momentum, float* pdel, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranMomX(float a, /*const*/ float* px, /*const*/ float* py, float* pmat, float momentum, float* pdel, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGrad(/*const*/ float* px, /*const*/ float* py, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradX(/*const*/ float* px, /*const*/ float* py, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradRU(/*const*/ float* px, /*const*/ float* py, /*const*/ int* pstarts, /*const*/ int* pindices,
            float* pcoefs, float* paccGrads, float* paccUpdates, float decay, float cond, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradRX(/*const*/ float* px, /*const*/ float* py, /*const*/ int* pstarts, /*const*/ int* pindices,
            float* pcoefs, float* paccGrads, float* paccUpdates, float decay, float cond, int crow);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradP(/*const*/ float* px, /*const*/ int* pposY, /*const*/ float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddXYTranGradPX(/*const*/ float* px, /*const*/ int* pposY, /*const*/ float* pvaluesY,
            int posMinY, int iposMinY, int iposLimY, float* pmat, float* paccGrads, float* paccUpdates,
            float decay, float cond, int crow, int ccol);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleU(float a, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void Scale(float a, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleX(float a, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleSrcU(float a, /*const*/ float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleAddU(float a, float b, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNorm(float maxNorm, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormX(float maxNorm, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormTranU(float maxNorm, float* pmat, int crow, int ccol);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormRU(float maxNorm, /*const*/ int* pstarts, float* pmat, int crow);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaleMaxNormCU(float maxNorm, int kernCount, int kernSize, float* pmat);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScale(float a, /*const*/ float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleU(float a, /*const*/ float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleX(float a, /*const*/ float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleSU(float a, /*const*/ float* ps, /*const*/ int* pi, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleCopyU(float a, /*const*/ float* ps, /*const*/ float* pd, float* pr, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleMom(float a, /*const*/ float* ps, float* pd, float momentum, float* pdel, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleMomX(float a, /*const*/ float* ps, float* pd, float momentum, float* pdel, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleGrad(/*const*/ float* ps, float* pd, float* paccGrads, float* paccUpdates,
            float decay, float cond, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleGradX(/*const*/ float* ps, float* pd, float* paccGrads, float* paccUpdates,
            float decay, float cond, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScaleMulti(int count, /*const*/ float* ps, float* pd, float* paccGrads,
            float* paccUpdates, float decay, float cond, int size);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddScalarU(float a, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddU(/*const*/ float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void Add(/*const*/ float* ps, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddX(/*const*/ float* ps, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void AddSU(/*const*/ float* ps, /*const*/ int* pi, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float Sum(/*const*/ float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumU(/*const*/ float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumX(/*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumSqU(/*const*/ float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumSqDiffU(float mean, /*const*/ float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumAbsU(/*const*/ float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float SumAbsDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MulElementWiseU(/*const*/ float* ps1, /*const*/float* ps2, float* pd, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MulElementWiseSU(/*const*/ float* ps1, /*const*/float* ps2,  /*const*/ int* pi, float* pd, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MaxAbsU(/*const*/ float* ps, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float MaxAbsDiffU(float mean, /*const*/ float* ps, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float DotU(/*const*/ float* pa, /*const*/ float* pb, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float DotSU(/*const*/ float* pa, /*const*/ float* pb, /*const*/ int* pi, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern float Dist2(/*const*/ float* px, /*const*/ float* py, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySigmoid(/*const*/ float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySigmoidX(/*const*/ float* ps, float* pd, int c)
        {
            ApplySigmoid(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySoftMaxU(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftMax(float* ps, float* pd, int c)
        {
            ApplySoftMaxU(ps, pd, c);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftMaxX(float* ps, float* pd, int c)
        {
            ApplySoftMaxU(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyRectifiedLinear(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyRectifiedLinearX(float* ps, float* pd, int c)
        {
            ApplyRectifiedLinear(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySquare(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySquareX(float* ps, float* pd, int c)
        {
            ApplySquare(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySqrt(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySqrtX(float* ps, float* pd, int c)
        {
            ApplySqrt(ps, pd, c);
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
        public static extern void ApplyAbs(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyAbsX(float* ps, float* pd, int c)
        {
            ApplyAbs(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyTanh(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyTanhX(float* ps, float* pd, int c)
        {
            ApplyTanh(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyBoundedRectifiedLinearA(float* ps, float* pd, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyBoundedRectifiedLinearX(float* ps, float* pd, int c)
        {
            ApplyBoundedRectifiedLinearA(ps, pd, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySigmoidDerivative(/*const*/ float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySigmoidDerivativeX(/*const*/ float* pv, float* pg, int c)
        {
            ApplySigmoidDerivative(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyRectifiedLinearDerivative(/*const*/ float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyRectifiedLinearDerivativeX(/*const*/ float* pv, float* pg, int c)
        {
            ApplyRectifiedLinearDerivative(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySquareDerivative(/*const*/ float* px, /*const*/ float* py, float* pg, int c, bool drop);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySquareDerivativeX(/*const*/ float* px, /*const*/ float* py, float* pg, int c, bool drop)
        {
            ApplySquareDerivative(px, py, pg, c, drop);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySqrtDerivative(/*const*/ float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySqrtDerivativeX(/*const*/ float* pv, float* pg, int c)
        {
            ApplySqrtDerivative(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplySoftRectifiedLinearDerivativeU(/*const*/ float* px, /*const*/ float* py, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftRectifiedLinearDerivativeA(/*const*/ float* px, /*const*/ float* py, float* pg, int c)
        {
            ApplySoftRectifiedLinearDerivativeU(px, py, pg, c);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySoftRectifiedLinearDerivativeX(/*const*/ float* px, /*const*/ float* py, float* pg, int c)
        {
            ApplySoftRectifiedLinearDerivativeU(px, py, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyAbsDerivative(/*const*/ float* px, /*const*/ float* py, float* pg, int c, bool drop);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyAbsDerivativeX(/*const*/ float* px, /*const*/ float* py, float* pg, int c, bool drop)
        {
            ApplyAbsDerivative(px, py, pg, c, drop);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyTanhDerivative(/*const*/ float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyTanhDerivativeX(/*const*/ float* pv, float* pg, int c)
        {
            ApplyTanhDerivative(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ApplyBoundedRectifiedLinearDerivative(/*const*/ float* pv, float* pg, int c);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyBoundedRectifiedLinearDerivativeX(/*const*/ float* pv, float* pg, int c)
        {
            ApplyBoundedRectifiedLinearDerivative(pv, pg, c);
        }

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ZeroItemsU(float* pd, int c, /*const*/ int* pindices, int cindices);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ZeroMatrixItemsCore(float* pd, int c, int ccol, int cfltRow, /*const*/ int* pindices, int cindices);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void SdcaL1UpdateU(float primalUpdate, /*const*/ float* ps, float threshold, float* pd1, float* pd2, int c);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void SdcaL1UpdateSU(float primalUpdate, /*const*/ float* ps, /*const*/ int* pi, float threshold, float* pd1, float* pd2, int c);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaledadeltaU(float* mat, float* accGrads, float* accUpdates, float decay, float cond, float* grads, int size);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void Scaledadelta(float* mat, float* accGrads, float* accUpdates, float decay, float cond, float* grads, int size);
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void ScaledadeltaX(float* mat, float* accGrads, float* accUpdates, float decay, float cond, float* grads, int size);

#if !CORECLR
        // In CoreCLR we use Buffer.MemoryCopy directly instead of
        // plumbing our own version.
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        public static extern void MemCpy(void* dst, /*const*/ void* src, long count);
#endif
    }
}
