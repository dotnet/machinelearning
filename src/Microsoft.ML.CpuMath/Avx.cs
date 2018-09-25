// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    /// <summary>
    /// Keep Avx.cs in sync with Sse.cs. When making changes to one, use BeyondCompare or a similar tool
    /// to view diffs and propagate appropriate changes to the other.
    /// </summary>
    public static class AvxUtils
    {
        public static bool CheckAvx()
        {
            return Thunk.ChkAvx();
        }

        public static void MatTimesSrc(bool tran, bool add, float[] mat, float[] src, float[] dst, int crun)
        {
            Contracts.Assert(mat.Length == dst.Length * src.Length);

            unsafe
            {
                fixed (float* pmat = &mat[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    if (!tran)
                    {
                        Contracts.Assert(0 <= crun && crun <= dst.Length);
                        Thunk.MatMulX(add, pmat, psrc, pdst, crun, src.Length);
                    }
                    else
                    {
                        Contracts.Assert(0 <= crun && crun <= src.Length);
                        Thunk.MatMulTranX(add, pmat, psrc, pdst, dst.Length, crun);
                    }
                }
            }
        }

        public static void MatTimesSrc(bool tran, bool add, float[] mat, int[] rgposSrc, float[] srcValues,
            int posMin, int iposMin, int iposLim, float[] dst, int crun)
        {
            Contracts.AssertValue(rgposSrc);
            Contracts.Assert(0 <= iposMin && iposMin <= iposLim && iposLim <= rgposSrc.Length);
            Contracts.Assert(mat.Length == dst.Length * srcValues.Length);

            if (iposMin >= iposLim)
            {
                if (!add)
                    Array.Clear(dst, 0, dst.Length);
                return;
            }
            Contracts.AssertNonEmpty(rgposSrc);
            unsafe
            {
                fixed (float* pdst = &dst[0])
                fixed (float* pmat = &mat[0])
                fixed (float* psrc = &srcValues[0])
                fixed (int* ppossrc = &rgposSrc[0])
                {
                    if (!tran)
                    {
                        Contracts.Assert(0 <= crun && crun <= dst.Length);
                        Thunk.MatMulPX(add, pmat, ppossrc, psrc, posMin, iposMin, iposLim, pdst, crun, srcValues.Length);
                    }
                    else
                    {
                        Contracts.Assert(0 <= crun && crun <= srcValues.Length);
                        Thunk.MatMulTranPX(add, pmat, ppossrc, psrc, posMin, iposMin, iposLim, pdst, dst.Length);
                    }
                }
            }
        }

        public static void MatTimesSrc(bool add, int[] starts, int[] indices, float[] coefs,
            float[] src, float[] dst, int crow)
        {
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == crow + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[crow] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(0 < crow && crow <= dst.Length);
            Contracts.Assert(crow * src.Length >= coefs.Length);

            unsafe
            {
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.MatMulRX(add, pstarts, pindices, pcoefs, psrc, pdst, crow);
            }
        }

        public static void MatTimesSrc(bool add, int[] mprowiv, int[] mprowcol,
            int[] mprowrun, int[] runs, float[] coefs,
            float[] src, float[] dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowiv);
            Contracts.Assert(mprowiv.Length == crow);
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowrun == null || mprowrun.Length == crow);
            Contracts.AssertNonEmpty(runs);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(0 < crow && crow <= dst.Length);

            unsafe
            {
                fixed (int* pmprowiv = &mprowiv[0])
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pruns = &runs[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    if (mprowrun == null)
                    {
                        Thunk.MatMulCX(add, pmprowiv, pmprowcol, pruns, pcoefs,
                            psrc, pdst, crow);
                    }
                    else
                    {
                        fixed (int* pmprowrun = &mprowrun[0])
                        {
                            Thunk.MatMulDX(add, pmprowiv, pmprowcol, pmprowrun, pruns, pcoefs,
                                psrc, pdst, crow);
                        }
                    }
                }
            }
        }

        public static void MeanOfSrc(bool add, int[] mprowcol, int[] mprowindices,
            int[] indices, float[] src, float[] dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowindices == null || mprowindices.Length == crow);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(0 < crow && crow <= dst.Length);

            unsafe
            {
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pmprowindices = mprowindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    // REVIEW: Implement using AVX
                    Thunk.MeanU(add, pmprowcol, pmprowindices, pindices, psrc, pdst, crow);
                }
            }
        }

        public static void MaxOfSrc(bool add, int[] mprowcol, int[] mprowindices,
            int[] indices, float[] src, float[] dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowindices == null || mprowindices.Length == crow);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(0 < crow && crow <= dst.Length);

            unsafe
            {
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pmprowindices = mprowindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    // REVIEW: Implement using AVX
                    Thunk.MaxU(add, pmprowcol, pmprowindices, pindices, psrc, pdst, crow);
                }
            }
        }

        public static void RespNormOfSrc(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            int[] mprowcol, int[] mprowindices, int[] indices,
            float[] src, float[] dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowindices == null || mprowindices.Length == crow);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(0 < crow && crow <= dst.Length);

            unsafe
            {
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pmprowindices = mprowindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    // REVIEW: Implement using AVX
                    Thunk.RespNormU(add, alpha, beta, avgOverFullKernel, offset, pmprowcol, pmprowindices, pindices,
                        psrc, pdst, crow);
                }
            }
        }

        public static void MatTranTimesSrc(bool add, int[] starts, int[] indices, float[] coefs,
            float[] src, float[] dst, int ccol)
        {
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == ccol + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[ccol] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(0 < ccol && ccol <= src.Length);
            Contracts.Assert(dst.Length * ccol >= coefs.Length);

            unsafe
            {
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.MatMulTranRX(add, pstarts, pindices, pcoefs, psrc, pdst, dst.Length, ccol);
            }
        }

        public static void MatTranTimesSrc(bool add, int[] mpcoliv, int[] mpcolrow, int[] mpcolrun,
            int[] runs, float[] coefs, float[] src, float[] dst, int ccol)
        {
            Contracts.AssertNonEmpty(mpcoliv);
            Contracts.Assert(mpcoliv.Length == ccol);
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.AssertNonEmpty(runs);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(mpcolrun == null || mpcolrun.Length == ccol);
            Contracts.Assert(0 < ccol && ccol <= src.Length);

            unsafe
            {
                fixed (int* pmpcoliv = &mpcoliv[0])
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pruns = &runs[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    if (mpcolrun == null)
                    {
                        Thunk.MatMulTranCX(add, pmpcoliv, pmpcolrow, pruns, pcoefs,
                            psrc, pdst, dst.Length, ccol);
                    }
                    else
                    {
                        fixed (int* pmpcolrun = &mpcolrun[0])
                        {
                            Thunk.MatMulTranDX(add, pmpcoliv, pmpcolrow, pmpcolrun, pruns, pcoefs,
                                psrc, pdst, dst.Length, ccol);
                        }
                    }
                }
            }
        }

        public static void MeanBackOfSrc(bool add, int[] mpcolrow, int[] mpcolindices,
            int[] indices, float[] src, float[] dst, int ccol)
        {
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.Assert(mpcolindices == null || mpcolindices.Length == ccol);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(0 < ccol && ccol <= src.Length);

            unsafe
            {
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pmpcolindices = mpcolindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    // REVIEW: Implement using AVX
                    Thunk.MeanBackU(add, pmpcolrow, pmpcolindices, pindices, psrc, pdst, dst.Length, ccol);
                }
            }
        }

        public static void MaxBackOfSrc(bool add, int[] mpcolrow, int[] mpcolindices,
            int[] indices, float[] src, float[] dst, float[] val, int ccol)
        {
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.Assert(mpcolindices == null || mpcolindices.Length == ccol);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(0 < ccol && ccol <= src.Length);
            Contracts.Assert(dst.Length == val.Length);

            unsafe
            {
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pmpcolindices = mpcolindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                fixed (float* pval = &val[0])
                {
                    // REVIEW: Implement using AVX
                    Thunk.MaxBackU(add, pmpcolrow, pmpcolindices, pindices, psrc, pdst, pval, dst.Length, ccol);
                }
            }
        }

        public static void RespNormBackOfSrc(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            int[] mpcolrow, int[] mpcolindices, int[] indices,
            float[] errors, float[] errorsPrev, float[] valuesPrev, int ccol)
        {
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.Assert(mpcolindices == null || mpcolindices.Length == ccol);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(0 < ccol && ccol <= errors.Length);
            Contracts.Assert(errorsPrev.Length == valuesPrev.Length);

            unsafe
            {
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pmpcolindices = mpcolindices)
                fixed (int* pindices = &indices[0])
                fixed (float* perr = &errors[0])
                fixed (float* perrPrev = &errorsPrev[0])
                fixed (float* pvalPrev = &valuesPrev[0])
                {
                    // REVIEW: Implement using AVX
                    Thunk.RespNormBackU(add, alpha, beta, avgOverFullKernel, offset, pmpcolrow, pmpcolindices, pindices,
                        perr, perrPrev, pvalPrev, errorsPrev.Length, ccol);
                }
            }
        }

        public static void AddXYTran(float a, float[] x, float[] y, float[] mat, int crow, float decay)
        {
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.Assert(x.Length * y.Length == mat.Length);
            Contracts.Assert(decay >= 0);

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &y[0])
                fixed (float* pmat = &mat[0])
                    Thunk.AddXYTranX(a, px, py, pmat, crow, y.Length, decay);
            }
        }

        public static void AddXYTran(float a, float[] x, int[] rgposY, float[] valuesY,
            int posMinY, int iposMinY, int iposLimY, float[] mat, int crow)
        {
            Contracts.AssertNonEmpty(rgposY);
            Contracts.Assert(0 <= iposMinY && iposMinY <= iposLimY && iposLimY <= rgposY.Length);
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.Assert(x.Length * valuesY.Length == mat.Length);

            if (iposMinY >= iposLimY)
                return;

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &valuesY[0])
                fixed (int* pposy = &rgposY[0])
                fixed (float* pmat = &mat[0])
                {
                    Thunk.AddXYTranPX(a, px, pposy, py, posMinY, iposMinY, iposLimY, pmat,
                        crow, valuesY.Length);
                }
            }
        }

        public static void AddXYTran(float a, float[] x, float[] y,
            int[] starts, int[] indices, float[] coefs, int crow, float decay)
        {
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == crow + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[crow] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(crow * y.Length >= coefs.Length);
            Contracts.Assert(decay >= 0);

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &y[0])
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                    Thunk.AddXYTranRX(a, px, py, pstarts, pindices, pcoefs, crow, decay);
            }
        }

        public static void AddXYTran(float a, float[] x, float[] y, int[] mprowiv,
            int[] mprowcol, int[] mprowrun, int[] runs, float[] coefs, int crow)
        {
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.AssertNonEmpty(mprowiv);
            Contracts.Assert(mprowiv.Length == crow);
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowrun == null || mprowrun.Length == crow);
            Contracts.AssertNonEmpty(runs);
            Contracts.AssertNonEmpty(coefs);

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &y[0])
                fixed (int* pmprowiv = &mprowiv[0])
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pruns = &runs[0])
                fixed (float* pcoefs = &coefs[0])
                {
                    if (mprowrun == null)
                        Thunk.AddXYTranCX(a, px, py, pmprowiv, pmprowcol, pruns, pcoefs, crow);
                    else
                    {
                        fixed (int* pmprowrun = mprowrun)
                            Thunk.AddXYTranDX(a, px, py, pmprowiv, pmprowcol, pmprowrun, pruns, pcoefs, crow);
                    }
                }
            }
        }

        public static void AddXYTran(float a, float[] x, float[] y, float[] mat, float momentum, float[] delta, int crow)
        {
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.Assert(x.Length * y.Length == mat.Length);
            Contracts.Assert(mat.Length == delta.Length);

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &y[0])
                fixed (float* pmat = &mat[0])
                fixed (float* pdel = &delta[0])
                    Thunk.AddXYTranMomX(a, px, py, pmat, momentum, pdel, crow, y.Length);
            }
        }

        public static void AddXYTran(float[] x, float[] y, float[] mat, float[] accGrads, float[] accUpdates,
            float decay, float cond, int crow)
        {
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.Assert(x.Length * y.Length == mat.Length);
            Contracts.Assert(mat.Length == accGrads.Length);
            Contracts.Assert(mat.Length == accUpdates.Length);

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &y[0])
                fixed (float* pmat = &mat[0])
                fixed (float* pag = &accGrads[0])
                fixed (float* pau = &accUpdates[0])
                    Thunk.AddXYTranGradX(px, py, pmat, pag, pau, decay, cond, crow, y.Length);
            }
        }

        public static void AddXYTran(float[] x, float[] y, int[] starts, int[] indices,
            float[] coefs, float[] accGrads, float[] accUpdates, float decay, float cond, int crow)
        {
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == crow + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[crow] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(crow * y.Length >= coefs.Length);
            Contracts.AssertNonEmpty(accGrads);
            Contracts.Assert(coefs.Length == accGrads.Length);
            Contracts.AssertNonEmpty(accUpdates);
            Contracts.Assert(coefs.Length == accUpdates.Length);

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &y[0])
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* pag = &accGrads[0])
                fixed (float* pau = &accUpdates[0])
                    Thunk.AddXYTranGradRX(px, py, pstarts, pindices, pcoefs, pag, pau, decay, cond, crow);
            }
        }

        public static void AddXYTran(float[] x, int[] rgposY, float[] valuesY,
            int posMinY, int iposMinY, int iposLimY, float[] mat,
            float[] accGrads, float[] accUpdates, float decay, float cond, int crow)
        {
            Contracts.Assert(0 <= iposMinY && iposMinY <= iposLimY && iposLimY <= rgposY.Length);
            Contracts.Assert(0 < crow && crow <= x.Length);
            Contracts.Assert(x.Length * valuesY.Length == mat.Length);
            Contracts.Assert(mat.Length == accGrads.Length);
            Contracts.Assert(mat.Length == accUpdates.Length);

            if (iposMinY >= iposLimY)
                return;

            unsafe
            {
                fixed (float* px = &x[0])
                fixed (float* py = &valuesY[0])
                fixed (int* pposy = &rgposY[0])
                fixed (float* pmat = &mat[0])
                fixed (float* pag = &accGrads[0])
                fixed (float* pau = &accUpdates[0])
                {
                    Thunk.AddXYTranGradPX(px, pposy, py, posMinY, iposMinY, iposLimY, pmat,
                        pag, pau, decay, cond, crow, valuesY.Length);
                }
            }
        }

        public static void Scale(float a, float[] dst)
        {
            unsafe
            {
                fixed (float* pdst = &dst[0])
                    Thunk.ScaleX(a, pdst, dst.Length);
            }
        }

        public static void Scale(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count && count <= dst.Length);

            unsafe
            {
                fixed (float* pd = &dst[0])
                    Thunk.ScaleU(a, pd, count);
            }
        }

        public static void ScaleConvWeights(float a, int kernelSize, float[] dst)
        {
            Contracts.AssertValue(dst);

            // REVIEW: implement in SSE/AVX.
            for (int istart = 0; istart < dst.Length; istart += kernelSize + 1)
            {
                for (int i = 0; i < kernelSize; i++)
                    dst[istart + i] *= a;
            }
        }

        public static void ScaleMaxNorm(bool tran, float maxNorm, float[] mat, int crun, int runLenPhy)
        {
            unsafe
            {
                fixed (float* pmat = &mat[0])
                {
                    if (!tran)
                        Thunk.ScaleMaxNormX(maxNorm, pmat, crun, runLenPhy);
                    else
                        Thunk.ScaleMaxNormTranU(maxNorm, pmat, crun, runLenPhy);
                }
            }
        }

        public static void ScaleMaxNorm(float maxNorm, int[] starts, int[] indices, float[] mat)
        {
            Contracts.AssertNonEmpty(starts);

            int crow = starts.Length - 1;
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertValue(indices);
            Contracts.Assert(starts[crow] == indices.Length);
            Contracts.AssertNonEmpty(mat);

            unsafe
            {
                fixed (int* pstarts = &starts[0])
                fixed (float* pmat = &mat[0])
                    Thunk.ScaleMaxNormRU(maxNorm, pstarts, pmat, crow);
            }
        }

        public static void ScaleMaxNorm(float maxNorm, int kernCount, int kernSize, float[] mat)
        {
            Contracts.AssertNonEmpty(mat);

            unsafe
            {
                fixed (float* pmat = &mat[0])
                    Thunk.ScaleMaxNormCU(maxNorm, kernCount, kernSize, pmat);
            }
        }

        public static void AddScale(float a, float[] src, float[] dst)
        {
            Contracts.Assert(src.Length == dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.AddScaleX(a, psrc, pdst, dst.Length);
            }
        }

        public static void AddScale(float a, float[] src, float[] dst, float momentum, float[] delta)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(src.Length == delta.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                fixed (float* pdel = &delta[0])
                    Thunk.AddScaleMomX(a, psrc, pdst, momentum, pdel, dst.Length);
            }
        }

        public static void AddScale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.AddScaleU(a, psrc, pdst, count);
            }
        }

        public static void AddScale(float[] src, float[] dst,
            float[] accGrads, float[] accUpdates, float decay, float cond)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(src.Length == accGrads.Length);
            Contracts.Assert(src.Length == accUpdates.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                fixed (float* pag = &accGrads[0])
                fixed (float* pau = &accUpdates[0])
                    Thunk.AddScaleGradX(psrc, pdst, pag, pau, decay, cond, dst.Length);
            }
        }

        public static void Add(float[] src, float[] dst)
        {
            Contracts.Assert(src.Length == dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.AddX(psrc, pdst, dst.Length);
            }
        }

        public static void Add(float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* ps = &src[0])
                fixed (float* pd = &dst[0])
                    Thunk.AddU(ps, pd, count);
            }
        }

        public static float Sum(float[] src)
        {
            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.SumX(psrc, src.Length);
            }
        }

        public static float Sum(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.SumU(psrc, count);
            }
        }

        public static float SumSq(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.SumSqU(psrc, count);
            }
        }

        public static float SumAbs(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.SumAbsU(psrc, count);
            }
        }

        public static float MaxAbs(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.MaxAbsU(psrc, count);
            }
        }

        public static float DotProductSparse(float[] a, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(count < a.Length);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            unsafe
            {
                fixed (float* pa = &a[0])
                fixed (float* pb = &b[0])
                fixed (int* pi = &indices[0])
                    return Thunk.DotSU(pa, pb, pi, count);
            }
        }

        public static float DotProductSparse(float[] a, int offset, float[] b, int[] indices, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset < a.Length);
            Contracts.Assert(a.Length - offset > count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(count <= b.Length);
            Contracts.Assert(count <= indices.Length);

            unsafe
            {
                fixed (float* pa = &a[offset])
                fixed (float* pb = &b[0])
                fixed (int* pi = &indices[0])
                    return Thunk.DotSU(pa, pb, pi, count);
            }
        }

        public static void ApplySigmoid(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplySigmoidX(psrc, pdst, c);
            }
        }

        public static void ApplySoftMax(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplySoftMaxX(psrc, pdst, c);
            }
        }

        public static void ApplyRectifiedLinear(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplyRectifiedLinearX(psrc, pdst, c);
            }
        }

        public static void ApplySquare(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplySquareX(psrc, pdst, c);
            }
        }

        public static void ApplySqrt(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplySqrtX(psrc, pdst, c);
            }
        }

        public static void ApplySoftRectifiedLinear(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplySoftRectifiedLinearX(psrc, pdst, c);
            }
        }

        public static void ApplyAbs(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplyAbsX(psrc, pdst, c);
            }
        }

        public static void ApplyTanh(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 < c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplyTanhX(psrc, pdst, c);
            }
        }

        public static void ApplyBoundedRectifiedLinear(float[] src, float[] dst, int c)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 <= c && c <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                    Thunk.ApplyBoundedRectifiedLinearX(psrc, pdst, c);
            }
        }

        public static void ApplySigmoidDerivative(float[] value, float[] grad)
        {
            Contracts.Assert(value.Length == grad.Length);

            unsafe
            {
                fixed (float* pvalue = &value[0])
                fixed (float* pgrad = &grad[0])
                    Thunk.ApplySigmoidDerivativeX(pvalue, pgrad, grad.Length);
            }
        }

        public static void ApplyRectifiedLinearDerivative(float[] value, float[] grad)
        {
            Contracts.Assert(value.Length == grad.Length);

            unsafe
            {
                fixed (float* pvalue = &value[0])
                fixed (float* pgrad = &grad[0])
                    Thunk.ApplyRectifiedLinearDerivativeX(pvalue, pgrad, grad.Length);
            }
        }

        public static void ApplySquareDerivative(float[] input, float[] output, float[] grad, bool drop)
        {
            Contracts.Assert(output.Length == input.Length);
            Contracts.Assert(output.Length == grad.Length);

            unsafe
            {
                fixed (float* px = &input[0])
                fixed (float* py = &output[0])
                fixed (float* pg = &grad[0])
                    Thunk.ApplySquareDerivativeX(px, py, pg, grad.Length, drop);
            }
        }

        public static void ApplySqrtDerivative(float[] value, float[] grad)
        {
            Contracts.Assert(value.Length == grad.Length);

            unsafe
            {
                fixed (float* pvalue = &value[0])
                fixed (float* pgrad = &grad[0])
                    Thunk.ApplySqrtDerivativeX(pvalue, pgrad, grad.Length);
            }
        }

        public static void ApplySoftRectifiedLinearDerivative(float[] input, float[] output, float[] grad)
        {
            Contracts.Assert(output.Length == input.Length);
            Contracts.Assert(output.Length == grad.Length);

            unsafe
            {
                fixed (float* px = &input[0])
                fixed (float* py = &output[0])
                fixed (float* pg = &grad[0])
                    Thunk.ApplySoftRectifiedLinearDerivativeX(px, py, pg, grad.Length);
            }
        }

        public static void ApplyAbsDerivative(float[] input, float[] output, float[] grad, bool drop)
        {
            Contracts.Assert(output.Length == input.Length);
            Contracts.Assert(output.Length == grad.Length);

            unsafe
            {
                fixed (float* px = &input[0])
                fixed (float* py = &output[0])
                fixed (float* pg = &grad[0])
                    Thunk.ApplyAbsDerivativeX(px, py, pg, grad.Length, drop);
            }
        }

        public static void ApplyTanhDerivative(float[] value, float[] grad)
        {
            Contracts.Assert(value.Length == grad.Length);

            unsafe
            {
                fixed (float* pvalue = &value[0])
                fixed (float* pgrad = &grad[0])
                    Thunk.ApplyTanhDerivativeX(pvalue, pgrad, grad.Length);
            }
        }

        public static void ApplyBoundedRectifiedLinearDerivative(float[] value, float[] grad)
        {
            Contracts.Assert(value.Length == grad.Length);

            unsafe
            {
                fixed (float* pvalue = &value[0])
                fixed (float* pgrad = &grad[0])
                    Thunk.ApplyBoundedRectifiedLinearDerivativeX(pvalue, pgrad, grad.Length);
            }
        }

        public static void ZeroMatrixItems(float[] dst, int ccol, int cfltRow, int[] indices)
        {
            Contracts.Assert(0 < ccol && ccol <= cfltRow);

            unsafe
            {
                fixed (float* pdst = &dst[0])
                fixed (int* pi = &indices[0])
                {
                    if (ccol == cfltRow)
                        Thunk.ZeroItemsU(pdst, dst.Length, pi, indices.Length);
                    else
                        Thunk.ZeroMatrixItemsCore(pdst, dst.Length, ccol, cfltRow, pi, indices.Length);
                }
            }
        }

        public static void ScaleAdadelta(float[] mat, float[] accGrads, float[] accUpdates, float decay, float cond, float[] grads)
        {
            Contracts.AssertNonEmpty(mat);
            Contracts.AssertNonEmpty(accGrads);
            Contracts.AssertNonEmpty(accUpdates);
            Contracts.Assert(mat.Length == accGrads.Length);
            Contracts.Assert(mat.Length == accUpdates.Length);
            Contracts.Assert(mat.Length <= grads.Length);

            unsafe
            {
                fixed (float* pm = &mat[0])
                fixed (float* pag = &accGrads[0])
                fixed (float* pau = &accUpdates[0])
                fixed (float* pg = &grads[0])
                    Thunk.ScaleAdadeltaX(pm, pag, pau, decay, cond, pg, mat.Length);
            }
        }
    }
}