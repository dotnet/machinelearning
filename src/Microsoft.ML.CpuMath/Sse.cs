// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    /// <summary>
    /// Keep Sse.cs in sync with Avx.cs. When making changes to one, use BeyondCompare or a similar tool
    /// to view diffs and propagate appropriate changes to the other.
    /// </summary>
    public static class SseUtils
    {
        public const int CbAlign = 16;

        private static bool Compat(AlignedArray a)
        {
            Contracts.AssertValue(a);
            Contracts.Assert(a.Size > 0);
            return a.CbAlign == CbAlign;
        }

        private unsafe static float* Ptr(AlignedArray a, float* p)
        {
            Contracts.AssertValue(a);
            float* q = p + a.GetBase((long)p);
            Contracts.Assert(((long)q & (CbAlign - 1)) == 0);
            return q;
        }

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, AlignedArray src, AlignedArray dst, int crun)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(mat.Size == dst.Size * src.Size);

            unsafe
            {
                fixed (float* pmat = &mat.Items[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                {
                    if (!tran)
                    {
                        Contracts.Assert(0 <= crun && crun <= dst.Size);
                        Thunk.MatMulA(add, Ptr(mat, pmat), Ptr(src, psrc), Ptr(dst, pdst), crun, src.Size);
                    }
                    else
                    {
                        Contracts.Assert(0 <= crun && crun <= src.Size);
                        Thunk.MatMulTranA(add, Ptr(mat, pmat), Ptr(src, psrc), Ptr(dst, pdst), dst.Size, crun);
                    }
                }
            }
        }

        public static void MatTimesSrc(bool tran, bool add, AlignedArray mat, int[] rgposSrc, AlignedArray srcValues,
            int posMin, int iposMin, int iposLim, AlignedArray dst, int crun)
        {
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(srcValues));
            Contracts.Assert(Compat(dst));
            Contracts.AssertValue(rgposSrc);
            Contracts.Assert(0 <= iposMin && iposMin <= iposLim && iposLim <= rgposSrc.Length);
            Contracts.Assert(mat.Size == dst.Size * srcValues.Size);

            if (iposMin >= iposLim)
            {
                if (!add)
                    dst.ZeroItems();
                return;
            }
            Contracts.AssertNonEmpty(rgposSrc);
            unsafe
            {
                fixed (float* pdst = &dst.Items[0])
                fixed (float* pmat = &mat.Items[0])
                fixed (float* psrc = &srcValues.Items[0])
                fixed (int* ppossrc = &rgposSrc[0])
                {
                    if (!tran)
                    {
                        Contracts.Assert(0 <= crun && crun <= dst.Size);
                        Thunk.MatMulPA(add, Ptr(mat, pmat), ppossrc, Ptr(srcValues, psrc), posMin, iposMin, iposLim, Ptr(dst, pdst), crun, srcValues.Size);
                    }
                    else
                    {
                        Contracts.Assert(0 <= crun && crun <= srcValues.Size);
                        Thunk.MatMulTranPA(add, Ptr(mat, pmat), ppossrc, Ptr(srcValues, psrc), posMin, iposMin, iposLim, Ptr(dst, pdst), dst.Size);
                    }
                }
            }
        }

        public static void MatTimesSrc(bool add, int[] starts, int[] indices, float[] coefs,
            AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == crow + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[crow] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(0 < crow && crow <= dst.Size);
            Contracts.Assert(crow * src.Size >= coefs.Length);

            unsafe
            {
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.MatMulRU(add, pstarts, pindices, pcoefs, Ptr(src, psrc), Ptr(dst, pdst), crow);
            }
        }

        public static void MatTimesSrc(bool add, int[] mprowiv, int[] mprowcol,
            int[] mprowrun, int[] runs, float[] coefs,
            AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowiv);
            Contracts.Assert(mprowiv.Length == crow);
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowrun == null || mprowrun.Length == crow);
            Contracts.AssertNonEmpty(runs);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(0 < crow && crow <= dst.Size);

            unsafe
            {
                fixed (int* pmprowiv = &mprowiv[0])
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pruns = &runs[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                {
                    if (mprowrun == null)
                    {
                        Thunk.MatMulCU(add, pmprowiv, pmprowcol, pruns, pcoefs,
                            Ptr(src, psrc), Ptr(dst, pdst), crow);
                    }
                    else
                    {
                        fixed (int* pmprowrun = &mprowrun[0])
                        {
                            Thunk.MatMulDU(add, pmprowiv, pmprowcol, pmprowrun, pruns, pcoefs,
                                Ptr(src, psrc), Ptr(dst, pdst), crow);
                        }
                    }
                }
            }
        }

        public static void MeanOfSrc(bool add, int[] mprowcol, int[] mprowindices,
            int[] indices, AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowindices == null || mprowindices.Length == crow);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(0 < crow && crow <= dst.Size);

            unsafe
            {
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pmprowindices = mprowindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.MeanU(add, pmprowcol, pmprowindices, pindices, Ptr(src, psrc), Ptr(dst, pdst), crow);
            }
        }

        public static void MaxOfSrc(bool add, int[] mprowcol, int[] mprowindices,
            int[] indices, AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowindices == null || mprowindices.Length == crow);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(0 < crow && crow <= dst.Size);

            unsafe
            {
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pmprowindices = mprowindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.MaxU(add, pmprowcol, pmprowindices, pindices, Ptr(src, psrc), Ptr(dst, pdst), crow);
            }
        }

        public static void RespNormOfSrc(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            int[] mprowcol, int[] mprowindices, int[] indices,
            AlignedArray src, AlignedArray dst, int crow)
        {
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowindices == null || mprowindices.Length == crow);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(0 < crow && crow <= dst.Size);

            unsafe
            {
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pmprowindices = mprowindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                {
                    Thunk.RespNormU(add, alpha, beta, avgOverFullKernel, offset, pmprowcol, pmprowindices, pindices,
                        Ptr(src, psrc), Ptr(dst, pdst), crow);
                }
            }
        }

        public static void MatTranTimesSrc(bool add, int[] starts, int[] indices, float[] coefs,
            AlignedArray src, AlignedArray dst, int ccol)
        {
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == ccol + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[ccol] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(0 < ccol && ccol <= src.Size);
            Contracts.Assert(dst.Size * ccol >= coefs.Length);

            unsafe
            {
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.MatMulTranRU(add, pstarts, pindices, pcoefs, Ptr(src, psrc), Ptr(dst, pdst), dst.Size, ccol);
            }
        }

        public static void MatTranTimesSrc(bool add, int[] mpcoliv, int[] mpcolrow, int[] mpcolrun,
            int[] runs, float[] coefs, AlignedArray src, AlignedArray dst, int ccol)
        {
            Contracts.AssertNonEmpty(mpcoliv);
            Contracts.Assert(mpcoliv.Length == ccol);
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.AssertNonEmpty(runs);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(mpcolrun == null || mpcolrun.Length == ccol);
            Contracts.Assert(0 < ccol && ccol <= src.Size);

            unsafe
            {
                fixed (int* pmpcoliv = &mpcoliv[0])
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pruns = &runs[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                {
                    if (mpcolrun == null)
                    {
                        Thunk.MatMulTranCU(add, pmpcoliv, pmpcolrow, pruns, pcoefs,
                            Ptr(src, psrc), Ptr(dst, pdst), dst.Size, ccol);
                    }
                    else
                    {
                        fixed (int* pmpcolrun = &mpcolrun[0])
                        {
                            Thunk.MatMulTranDU(add, pmpcoliv, pmpcolrow, pmpcolrun, pruns, pcoefs,
                                Ptr(src, psrc), Ptr(dst, pdst), dst.Size, ccol);
                        }
                    }
                }
            }
        }

        public static void MeanBackOfSrc(bool add, int[] mpcolrow, int[] mpcolindices,
            int[] indices, AlignedArray src, AlignedArray dst, int ccol)
        {
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.Assert(mpcolindices == null || mpcolindices.Length == ccol);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(0 < ccol && ccol <= src.Size);

            unsafe
            {
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pmpcolindices = mpcolindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.MeanBackU(add, pmpcolrow, pmpcolindices, pindices, Ptr(src, psrc), Ptr(dst, pdst), dst.Size, ccol);
            }
        }

        public static void MaxBackOfSrc(bool add, int[] mpcolrow, int[] mpcolindices,
            int[] indices, AlignedArray src, AlignedArray dst, AlignedArray val, int ccol)
        {
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.Assert(mpcolindices == null || mpcolindices.Length == ccol);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(Compat(val));
            Contracts.Assert(0 < ccol && ccol <= src.Size);
            Contracts.Assert(dst.Size == val.Size);

            unsafe
            {
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pmpcolindices = mpcolindices)
                fixed (int* pindices = &indices[0])
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                fixed (float* pval = &val.Items[0])
                    Thunk.MaxBackU(add, pmpcolrow, pmpcolindices, pindices, Ptr(src, psrc), Ptr(dst, pdst), Ptr(val, pval), dst.Size, ccol);
            }
        }

        public static void RespNormBackOfSrc(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
            int[] mpcolrow, int[] mpcolindices, int[] indices,
            AlignedArray errors, AlignedArray errorsPrev, AlignedArray valuesPrev, int ccol)
        {
            Contracts.AssertNonEmpty(mpcolrow);
            Contracts.Assert(mpcolrow.Length == ccol);
            Contracts.Assert(mpcolindices == null || mpcolindices.Length == ccol);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(Compat(errors));
            Contracts.Assert(Compat(errorsPrev));
            Contracts.Assert(Compat(valuesPrev));
            Contracts.Assert(0 < ccol && ccol <= errors.Size);
            Contracts.Assert(errorsPrev.Size == valuesPrev.Size);

            unsafe
            {
                fixed (int* pmpcolrow = &mpcolrow[0])
                fixed (int* pmpcolindices = mpcolindices)
                fixed (int* pindices = &indices[0])
                fixed (float* perr = &errors.Items[0])
                fixed (float* perrPrev = &errorsPrev.Items[0])
                fixed (float* pvalPrev = &valuesPrev.Items[0])
                {
                    Thunk.RespNormBackU(add, alpha, beta, avgOverFullKernel, offset, pmpcolrow, pmpcolindices, pindices,
                        Ptr(errors, perr), Ptr(errorsPrev, perrPrev), Ptr(valuesPrev, pvalPrev), errorsPrev.Size, ccol);
                }
            }
        }

        public static void AddXYTran(float a, AlignedArray x, AlignedArray y, AlignedArray mat, int crow, float decay)
        {
            Contracts.Assert(Compat(x));
            Contracts.Assert(Compat(y));
            Contracts.Assert(Compat(mat));
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.Assert(x.Size * y.Size == mat.Size);
            Contracts.Assert(decay >= 0);

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &y.Items[0])
                fixed (float* pmat = &mat.Items[0])
                    Thunk.AddXYTranA(a, Ptr(x, px), Ptr(y, py), Ptr(mat, pmat), crow, y.Size, decay);
            }
        }

        public static void AddXYTran(float a, AlignedArray x, int[] rgposY, AlignedArray valuesY,
            int posMinY, int iposMinY, int iposLimY, AlignedArray mat, int crow)
        {
            Contracts.Assert(Compat(x));
            Contracts.Assert(Compat(valuesY));
            Contracts.Assert(Compat(mat));
            Contracts.AssertNonEmpty(rgposY);
            Contracts.Assert(0 <= iposMinY && iposMinY <= iposLimY && iposLimY <= rgposY.Length);
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.Assert(x.Size * valuesY.Size == mat.Size);

            if (iposMinY >= iposLimY)
                return;

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &valuesY.Items[0])
                fixed (int* pposy = &rgposY[0])
                fixed (float* pmat = &mat.Items[0])
                {
                    Thunk.AddXYTranPA(a, Ptr(x, px), pposy, Ptr(valuesY, py), posMinY, iposMinY, iposLimY, Ptr(mat, pmat),
                        crow, valuesY.Size);
                }
            }
        }

        public static void AddXYTran(float a, AlignedArray x, AlignedArray y,
            int[] starts, int[] indices, float[] coefs, int crow, float decay)
        {
            Contracts.Assert(Compat(x));
            Contracts.Assert(Compat(y));
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == crow + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[crow] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(crow * y.Size >= coefs.Length);
            Contracts.Assert(decay >= 0);

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &y.Items[0])
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                    Thunk.AddXYTranRU(a, Ptr(x, px), Ptr(y, py), pstarts, pindices, pcoefs, crow, decay);
            }
        }

        public static void AddXYTran(float a, AlignedArray x, AlignedArray y, int[] mprowiv,
            int[] mprowcol, int[] mprowrun, int[] runs, float[] coefs, int crow)
        {
            Contracts.Assert(Compat(x));
            Contracts.Assert(Compat(y));
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.AssertNonEmpty(mprowiv);
            Contracts.Assert(mprowiv.Length == crow);
            Contracts.AssertNonEmpty(mprowcol);
            Contracts.Assert(mprowcol.Length == crow);
            Contracts.Assert(mprowrun == null || mprowrun.Length == crow);
            Contracts.AssertNonEmpty(runs);
            Contracts.AssertNonEmpty(coefs);

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &y.Items[0])
                fixed (int* pmprowiv = &mprowiv[0])
                fixed (int* pmprowcol = &mprowcol[0])
                fixed (int* pruns = &runs[0])
                fixed (float* pcoefs = &coefs[0])
                {
                    if (mprowrun == null)
                        Thunk.AddXYTranCU(a, Ptr(x, px), Ptr(y, py), pmprowiv, pmprowcol, pruns, pcoefs, crow);
                    else
                    {
                        fixed (int* pmprowrun = &mprowrun[0])
                            Thunk.AddXYTranDU(a, Ptr(x, px), Ptr(y, py), pmprowiv, pmprowcol, pmprowrun, pruns, pcoefs, crow);
                    }
                }
            }
        }

        public static void AddXYTran(float a, AlignedArray x, AlignedArray y, AlignedArray mat, float momentum, AlignedArray delta, int crow)
        {
            Contracts.Assert(Compat(x));
            Contracts.Assert(Compat(y));
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(delta));
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.Assert(x.Size * y.Size == mat.Size);
            Contracts.Assert(mat.Size == delta.Size);

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &y.Items[0])
                fixed (float* pmat = &mat.Items[0])
                fixed (float* pdel = &delta.Items[0])
                    Thunk.AddXYTranMomA(a, Ptr(x, px), Ptr(y, py), Ptr(mat, pmat), momentum, Ptr(delta, pdel), crow, y.Size);
            }
        }

        public static void AddXYTran(AlignedArray x, AlignedArray y, AlignedArray mat, AlignedArray accGrads, AlignedArray accUpdates,
            float decay, float cond, int crow)
        {
            Contracts.Assert(Compat(x));
            Contracts.Assert(Compat(y));
            Contracts.Assert(Compat(mat));
            Contracts.Assert(Compat(accGrads));
            Contracts.Assert(Compat(accUpdates));
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.Assert(x.Size * y.Size == mat.Size);
            Contracts.Assert(mat.Size == accGrads.Size);
            Contracts.Assert(mat.Size == accUpdates.Size);

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &y.Items[0])
                fixed (float* pmat = &mat.Items[0])
                fixed (float* pag = &accGrads.Items[0])
                fixed (float* pau = &accUpdates.Items[0])
                    Thunk.AddXYTranGradA(Ptr(x, px), Ptr(y, py), Ptr(mat, pmat), Ptr(accGrads, pag), Ptr(accUpdates, pau), decay, cond, crow, y.Size);
            }
        }

        public static void AddXYTran(AlignedArray x, AlignedArray y, int[] starts, int[] indices,
            float[] coefs, float[] accGrads, float[] accUpdates, float decay, float cond, int crow)
        {
            Contracts.Assert(Compat(x));
            Contracts.Assert(Compat(y));
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.AssertNonEmpty(starts);
            Contracts.Assert(starts.Length == crow + 1);
            Contracts.Assert(starts[0] == 0);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(starts[crow] == indices.Length);
            Contracts.AssertNonEmpty(coefs);
            Contracts.Assert(indices.Length == coefs.Length);
            Contracts.Assert(crow * y.Size >= coefs.Length);
            Contracts.AssertNonEmpty(accGrads);
            Contracts.Assert(coefs.Length == accGrads.Length);
            Contracts.AssertNonEmpty(accUpdates);
            Contracts.Assert(coefs.Length == accUpdates.Length);

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &y.Items[0])
                fixed (int* pstarts = &starts[0])
                fixed (int* pindices = &indices[0])
                fixed (float* pcoefs = &coefs[0])
                fixed (float* pag = &accGrads[0])
                fixed (float* pau = &accUpdates[0])
                    Thunk.AddXYTranGradRU(Ptr(x, px), Ptr(y, py), pstarts, pindices, pcoefs, pag, pau, decay, cond, crow);
            }
        }

        public static void AddXYTran(AlignedArray x, int[] rgposY, AlignedArray valuesY,
            int posMinY, int iposMinY, int iposLimY, AlignedArray mat,
            AlignedArray accGrads, AlignedArray accUpdates, float decay, float cond, int crow)
        {
            Contracts.Assert(Compat(x));
            Contracts.AssertNonEmpty(rgposY);
            Contracts.Assert(Compat(valuesY));
            Contracts.Assert(Compat(mat));
            Contracts.Assert(0 <= iposMinY && iposMinY <= iposLimY && iposLimY <= rgposY.Length);
            Contracts.Assert(0 < crow && crow <= x.Size);
            Contracts.Assert(x.Size * valuesY.Size == mat.Size);
            Contracts.Assert(mat.Size == accGrads.Size);
            Contracts.Assert(mat.Size == accUpdates.Size);

            if (iposMinY >= iposLimY)
                return;

            unsafe
            {
                fixed (float* px = &x.Items[0])
                fixed (float* py = &valuesY.Items[0])
                fixed (int* pposy = &rgposY[0])
                fixed (float* pmat = &mat.Items[0])
                fixed (float* pag = &accGrads.Items[0])
                fixed (float* pau = &accUpdates.Items[0])
                {
                    Thunk.AddXYTranGradPA(Ptr(x, px), pposy, Ptr(valuesY, py), posMinY, iposMinY, iposLimY, Ptr(mat, pmat),
                        Ptr(accGrads, pag), Ptr(accUpdates, pau), decay, cond, crow, valuesY.Size);
                }
            }
        }

        // dst += a
        public static void Add(float a, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 < count && count <= dst.Length);

            unsafe
            {
                fixed (float* pdst = &dst[0])
                    Thunk.AddScalarU(a, pdst, count);
            }
        }

        public static void Scale(float a, AlignedArray dst)
        {
            Contracts.Assert(Compat(dst));

            unsafe
            {
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ScaleA(a, Ptr(dst, pdst), dst.Size);
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

        public static void Scale(float a, float[] dst, int offset, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset < dst.Length - count);

            unsafe
            {
                fixed (float* pd = &dst[offset])
                    Thunk.ScaleU(a, pd, count);
            }
        }

        // dst = a * src
        public static void Scale(float a, float[] src, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count <= dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[0])
                {
                    Thunk.ScaleSrcU(a, psrc, pdst, count);
                }
            }
        }

        // dst[i] = a * (dst[i] + b)
        public static void ScaleAdd(float a, float b, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 < count && count <= dst.Length);

            unsafe
            {
                fixed (float* pdst = &dst[0])
                    Thunk.ScaleAddU(a, b, pdst, count);
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

        public static void ScaleMaxNorm(bool tran, float maxNorm, AlignedArray mat, int crun, int runLenPhy)
        {
            // Called also by MklMath which uses Avx alignment, which is a multiple of Sse alignment.
            // Hence, Compat(mat) cannot be asserted here since it checks for exact Sse alignment (mat.CbAlign == CbAlign).
            Contracts.AssertValue(mat);
            Contracts.Assert(mat.Size > 0);
            Contracts.Assert((mat.CbAlign % CbAlign) == 0);

            unsafe
            {
                fixed (float* pmat = &mat.Items[0])
                {
                    if (!tran)
                        Thunk.ScaleMaxNormA(maxNorm, Ptr(mat, pmat), crun, runLenPhy);
                    else
                        Thunk.ScaleMaxNormTranU(maxNorm, Ptr(mat, pmat), crun, runLenPhy);
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

        public static void AddScale(float a, AlignedArray src, AlignedArray dst)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.AddScaleA(a, Ptr(src, psrc), Ptr(dst, pdst), dst.Size);
            }
        }

        public static void AddScale(float a, AlignedArray src, AlignedArray dst, float momentum, AlignedArray delta)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(Compat(delta));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(src.Size == delta.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                fixed (float* pdel = &delta.Items[0])
                    Thunk.AddScaleMomA(a, Ptr(src, psrc), Ptr(dst, pdst), momentum, Ptr(delta, pdel), dst.Size);
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

        public static void AddScale(float a, float[] src, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(0 < count && count <= dst.Length - dstOffset);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pdst = &dst[dstOffset])
                    Thunk.AddScaleU(a, psrc, pdst, count);
            }
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pdst = &dst[0])
                    Thunk.AddScaleSU(a, psrc, pi, pdst, count);
            }
        }

        public static void AddScaleCopy(float a, float[] src, float[] dst, float[] res, int count)
        {
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 < count && count <= dst.Length);
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(res);
            Contracts.Assert(count <= res.Length);

            unsafe
            {
                fixed (float* pdst = &dst[0])
                fixed (float* psrc = &src[0])
                fixed (float* pres = &res[0])
                    Thunk.AddScaleCopyU(a, psrc, pdst, pres, count);
            }
        }

        public static void AddScale(float a, float[] src, int[] indices, float[] dst,
            int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(count < dst.Length - dstOffset);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pdst = &dst[dstOffset])
                    Thunk.AddScaleSU(a, psrc, pi, pdst, count);
            }
        }

        public static void AddScale(AlignedArray src, AlignedArray dst,
            AlignedArray accGrads, AlignedArray accUpdates, float decay, float cond)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(Compat(accGrads));
            Contracts.Assert(Compat(accUpdates));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(src.Size == accGrads.Size);
            Contracts.Assert(src.Size == accUpdates.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                fixed (float* pag = &accGrads.Items[0])
                fixed (float* pau = &accUpdates.Items[0])
                    Thunk.AddScaleGradA(Ptr(src, psrc), Ptr(dst, pdst), Ptr(accGrads, pag), Ptr(accUpdates, pau), decay, cond, dst.Size);
            }
        }

        public static void Add(AlignedArray src, AlignedArray dst)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.AddA(Ptr(src, psrc), Ptr(dst, pdst), dst.Size);
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

        public static void Add(float[] src, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(count < dst.Length);

            unsafe
            {
                fixed (float* ps = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pd = &dst[0])
                    Thunk.AddSU(ps, pi, pd, count);
            }
        }

        public static void Add(float[] src, int[] indices, float[] dst, int dstOffset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.Assert(0 <= dstOffset && dstOffset < dst.Length);
            Contracts.Assert(count <= dst.Length - dstOffset);

            unsafe
            {
                fixed (float* ps = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pd = &dst[dstOffset])
                    Thunk.AddSU(ps, pi, pd, count);
            }
        }

        public static void MulElementWise(float[] src1, float[] src2, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.Assert(0 < count && count <= src1.Length);
            Contracts.AssertNonEmpty(src2);
            Contracts.Assert(0 < count && count <= src2.Length);
            Contracts.AssertNonEmpty(dst);
            unsafe
            {
                fixed (float* ps1 = &src1[0])
                fixed (float* ps2 = &src2[0])
                fixed (float* pd = &dst[0])
                    Thunk.MulElementWiseU(ps1, ps2, pd, count);
            }
        }

        public static void MulElementWise(float[] src1, float[] src2, int[] indices, float[] dst, int count)
        {
            Contracts.AssertNonEmpty(src1);
            Contracts.Assert(0 < count && count <= src1.Length);
            Contracts.AssertNonEmpty(src2);
            Contracts.Assert(0 < count && count <= src2.Length);
            Contracts.AssertNonEmpty(dst);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            unsafe
            {
                fixed (float* ps1 = &src1[0])
                fixed (float* ps2 = &src2[0])
                fixed (int* pi = &indices[0])
                fixed (float* pd = &dst[0])
                    Thunk.MulElementWiseSU(ps1, ps2, pi, pd, count);
            }
        }

        public static float Sum(AlignedArray src)
        {
            Contracts.Assert(Compat(src));

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                    return Thunk.SumA(Ptr(src, psrc), src.Size);
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

        public static float Sum(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
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

        public static float SumSq(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return Thunk.SumSqU(psrc, count);
            }
        }

        public static float SumSq(float mean, float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return (mean == 0 ? Thunk.SumSqU(psrc, count) : Thunk.SumSqDiffU(mean, psrc, count));
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

        public static float SumAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return Thunk.SumAbsU(psrc, count);
            }
        }

        public static float SumAbs(float mean, float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return (mean == 0 ? Thunk.SumAbsU(psrc, count) : Thunk.SumAbsDiffU(mean, psrc, count));
            }
        }

        public static float MaxAbs(float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.MaxAbsU(psrc, src.Length);
            }
        }

        public static float MaxAbsDiff(float mean, float[] src, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count && count <= src.Length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                    return Thunk.MaxAbsDiffU(mean, psrc, count);
            }
        }

        public static float MaxAbs(float[] src, int offset, int count)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= src.Length - count);

            unsafe
            {
                fixed (float* psrc = &src[offset])
                    return Thunk.MaxAbsU(psrc, count);
            }
        }

        public static float DotProductDense(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count);
            Contracts.Assert(a.Length >= count);
            Contracts.Assert(b.Length >= count);

            unsafe
            {
                fixed (float* pa = &a[0])
                fixed (float* pb = &b[0])
                    return Thunk.DotU(pa, pb, count);
            }
        }

        public static float DotProductDense(float[] a, int offset, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.Assert(0 < count);
            Contracts.Assert(0 <= offset && offset <= a.Length - count);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(b.Length >= count);

            unsafe
            {
                fixed (float* pa = &a[offset])
                fixed (float* pb = &b[0])
                    return Thunk.DotU(pa, pb, count);
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

        public static float L2DistSquared(float[] a, float[] b, int count)
        {
            Contracts.AssertNonEmpty(a);
            Contracts.AssertNonEmpty(b);
            Contracts.Assert(0 < count && count <= a.Length);
            Contracts.Assert(count <= b.Length);

            unsafe
            {
                fixed (float* pa = &a[0])
                fixed (float* pb = &b[0])
                    return Thunk.Dist2(pa, pb, count);
            }
        }

        public static void ApplySigmoid(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplySigmoidA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplySoftMax(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplySoftMaxA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplyRectifiedLinear(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplyRectifiedLinearA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplySquare(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplySquareA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplySqrt(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplySqrtA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplySoftRectifiedLinear(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplySoftRectifiedLinearA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplyAbs(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplyAbsA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplyTanh(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 < c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplyTanhA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplyBoundedRectifiedLinear(AlignedArray src, AlignedArray dst, int c)
        {
            Contracts.Assert(Compat(src));
            Contracts.Assert(Compat(dst));
            Contracts.Assert(src.Size == dst.Size);
            Contracts.Assert(0 <= c && c <= dst.Size);

            unsafe
            {
                fixed (float* psrc = &src.Items[0])
                fixed (float* pdst = &dst.Items[0])
                    Thunk.ApplyBoundedRectifiedLinearA(Ptr(src, psrc), Ptr(dst, pdst), c);
            }
        }

        public static void ApplySigmoidDerivative(AlignedArray value, AlignedArray grad)
        {
            Contracts.Assert(Compat(value));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(value.Size == grad.Size);

            unsafe
            {
                fixed (float* pvalue = &value.Items[0])
                fixed (float* pgrad = &grad.Items[0])
                    Thunk.ApplySigmoidDerivativeA(Ptr(value, pvalue), Ptr(grad, pgrad), grad.Size);
            }
        }

        public static void ApplyRectifiedLinearDerivative(AlignedArray value, AlignedArray grad)
        {
            Contracts.Assert(Compat(value));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(value.Size == grad.Size);

            unsafe
            {
                fixed (float* pvalue = &value.Items[0])
                fixed (float* pgrad = &grad.Items[0])
                    Thunk.ApplyRectifiedLinearDerivativeA(Ptr(value, pvalue), Ptr(grad, pgrad), grad.Size);
            }
        }

        public static void ApplySquareDerivative(AlignedArray input, AlignedArray output, AlignedArray grad, bool drop)
        {
            Contracts.Assert(Compat(input));
            Contracts.Assert(Compat(output));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(output.Size == input.Size);
            Contracts.Assert(output.Size == grad.Size);

            unsafe
            {
                fixed (float* px = &input.Items[0])
                fixed (float* py = &output.Items[0])
                fixed (float* pg = &grad.Items[0])
                    Thunk.ApplySquareDerivativeA(Ptr(input, px), Ptr(output, py), Ptr(grad, pg), grad.Size, drop);
            }
        }

        public static void ApplySqrtDerivative(AlignedArray value, AlignedArray grad)
        {
            Contracts.Assert(Compat(value));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(value.Size == grad.Size);

            unsafe
            {
                fixed (float* pvalue = &value.Items[0])
                fixed (float* pgrad = &grad.Items[0])
                    Thunk.ApplySqrtDerivativeA(Ptr(value, pvalue), Ptr(grad, pgrad), grad.Size);
            }
        }

        public static void ApplySoftRectifiedLinearDerivative(AlignedArray input, AlignedArray output, AlignedArray grad)
        {
            Contracts.Assert(Compat(input));
            Contracts.Assert(Compat(output));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(output.Size == input.Size);
            Contracts.Assert(output.Size == grad.Size);

            unsafe
            {
                fixed (float* px = &input.Items[0])
                fixed (float* py = &output.Items[0])
                fixed (float* pg = &grad.Items[0])
                    Thunk.ApplySoftRectifiedLinearDerivativeA(Ptr(input, px), Ptr(output, py), Ptr(grad, pg), grad.Size);
            }
        }

        public static void ApplyAbsDerivative(AlignedArray input, AlignedArray output, AlignedArray grad, bool drop)
        {
            Contracts.Assert(Compat(input));
            Contracts.Assert(Compat(output));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(output.Size == input.Size);
            Contracts.Assert(output.Size == grad.Size);

            unsafe
            {
                fixed (float* px = &input.Items[0])
                fixed (float* py = &output.Items[0])
                fixed (float* pg = &grad.Items[0])
                    Thunk.ApplyAbsDerivativeA(Ptr(input, px), Ptr(output, py), Ptr(grad, pg), grad.Size, drop);
            }
        }

        public static void ApplyTanhDerivative(AlignedArray value, AlignedArray grad)
        {
            Contracts.Assert(Compat(value));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(value.Size == grad.Size);

            unsafe
            {
                fixed (float* pvalue = &value.Items[0])
                fixed (float* pgrad = &grad.Items[0])
                    Thunk.ApplyTanhDerivativeA(Ptr(value, pvalue), Ptr(grad, pgrad), grad.Size);
            }
        }

        public static void ApplyBoundedRectifiedLinearDerivative(AlignedArray value, AlignedArray grad)
        {
            Contracts.Assert(Compat(value));
            Contracts.Assert(Compat(grad));
            Contracts.Assert(value.Size == grad.Size);

            unsafe
            {
                fixed (float* pvalue = &value.Items[0])
                fixed (float* pgrad = &grad.Items[0])
                    Thunk.ApplyBoundedRectifiedLinearDerivativeA(Ptr(value, pvalue), Ptr(grad, pgrad), grad.Size);
            }
        }

        public static void ZeroMatrixItems(AlignedArray dst, int ccol, int cfltRow, int[] indices)
        {
            Contracts.Assert(0 < ccol && ccol <= cfltRow);

            unsafe
            {
                fixed (float* pdst = &dst.Items[0])
                fixed (int* pi = &indices[0])
                {
                    if (ccol == cfltRow)
                        Thunk.ZeroItemsU(Ptr(dst, pdst), dst.Size, pi, indices.Length);
                    else
                        Thunk.ZeroMatrixItemsCore(Ptr(dst, pdst), dst.Size, ccol, cfltRow, pi, indices.Length);
                }
            }
        }

        public static void SdcaL1UpdateDense(float primalUpdate, int length, float[] src, float threshold, float[] v, float[] w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(length <= src.Length);
            Contracts.AssertNonEmpty(v);
            Contracts.Assert(length <= v.Length);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(length <= w.Length);
            Contracts.Assert(length > 0);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (float* pd1 = &v[0])
                fixed (float* pd2 = &w[0])
                    Thunk.SdcaL1UpdateU(primalUpdate, psrc, threshold, pd1, pd2, length);
            }
        }

        public static void SdcaL1UpdateSparse(float primalUpdate, int length, float[] src, int[] indices, int count, float threshold, float[] v, float[] w)
        {
            Contracts.AssertNonEmpty(src);
            Contracts.Assert(count <= src.Length);
            Contracts.AssertNonEmpty(indices);
            Contracts.Assert(count <= indices.Length);
            Contracts.AssertNonEmpty(w);
            Contracts.Assert(length <= w.Length);
            Contracts.AssertNonEmpty(v);
            Contracts.Assert(length <= v.Length);
            Contracts.Assert(0 < count);
            Contracts.Assert(count < length);

            unsafe
            {
                fixed (float* psrc = &src[0])
                fixed (int* pi = &indices[0])
                fixed (float* pd1 = &v[0])
                fixed (float* pd2 = &w[0])
                    Thunk.SdcaL1UpdateSU(primalUpdate, psrc, pi, threshold, pd1, pd2, count);
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
                    Thunk.ScaleAdadeltaU(pm, pag, pau, decay, cond, pg, mat.Length);
            }
        }
    }
}