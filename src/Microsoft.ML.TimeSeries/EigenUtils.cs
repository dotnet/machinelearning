//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime.Internal.Utilities;
using Float = System.Single;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    //REVIEW ktran: improve perf with SSE and Multithreading
    public static class EigenUtils
    {
        //Compute the Eigen-decomposition of a symmetric matrix
        //REVIEW ktran: use matrix/vector operations, not Array Math
        public static void EigenDecomposition(Float[] a, out Float[] eigenvalues, out Float[] eigenvectors)
        {
            var count = a.Length;
            var n = (int)Math.Sqrt(count);
            Contracts.Assert(n * n == count);

            eigenvectors = new Float[count];
            eigenvalues = new Float[n];

            //Reduce A to tridiagonal form
            //REVIEW ktran: it's not ideal to keep using the same variable name for different purposes
            // - After the operation, "eigenvalues" means the diagonal elements of the reduced matrix
            //and "eigenvectors" means the orthogonal similarity transformation matrix
            // - Consider aliasing variables
            var w = new Float[n];
            Tred(a, eigenvalues, w, eigenvectors, n);

            //Eigen-decomposition of the tridiagonal matrix
            //After this operation, "eigenvalues" means eigenvalues^2
            Imtql(eigenvalues, w, eigenvectors, n);

            for (int i = 0; i < n; i++)
                eigenvalues[i] = eigenvalues[i] <= 0 ? (Float)(0.0) : (Float)Math.Sqrt(eigenvalues[i]);
        }

        private static Float Hypot(Float x, Float y)
        {
            x = Math.Abs(x);
            y = Math.Abs(y);

            if (x == 0 || y == 0)
                return x + y;

            if (x < y)
            {
                double t = x / y;
                return y * (Float)Math.Sqrt(1 + t * t);
            }
            else
            {
                double t = y / x;
                return x * (Float)Math.Sqrt(1 + t * t);
            }
        }

        private static Float CopySign(Float x, Float y)
        {
            Float xx = Math.Abs(x);
            return y < 0 ? -xx : xx;
        }

        private static void Tred(Float[] a, Float[] d, Float[] e, Float[] z, int n)
        {
            float g;
            float h;
            int i;
            int j;
            int k;
            int l;

            /*     this subroutine reduces a Float symmetric matrix to a */
            /*     symmetric tridiagonal matrix using and accumulating */
            /*     orthogonal similarity transformations. */

            /*     on input */

            /*	  n is the order of the matrix. */

            /*	  a contains the Float symmetric input matrix. only the */
            /*	    lower triangle of the matrix need be supplied. */

            /*     on output */

            /*	  d contains the diagonal elements of the tridiagonal matrix. */

            /*	  e contains the sub-diagonal elements of the tridiagonal */
            /*	    matrix in its last n-1 positions. e(1) is set to zero. */
            /*    z contains the orthogonal similarity transformation */

            /*     ------------------------------------------------------------------ */

            /* Function Body */

            for (i = 0; i < n; ++i)
            {
                for (j = i; j < n; ++j)
                {
                    z[j + i * n] = a[j + i * n];
                }

                d[i] = a[n - 1 + i * n];
            }

            if (n == 1)
            {
                d[0] = z[0];
                z[0] = 1;
                e[0] = 0;
                return;
            }
            //     .......... for i=n step -1 until 2 do -- ..........
            for (i = n; i-- > 1;)
            {
                l = i - 1;
                h = 0;
                Float scale = 0;
                if (l == 0)
                {
                    e[1] = d[0];
                    d[0] = z[0];
                    z[1] = 0;
                    z[n] = 0;
                    d[1] = h;
                    continue;
                }
                //     .......... scale row ..........
                for (k = 0; k < i; ++k)
                {
                    scale += Math.Abs(d[k]);
                }

                if (scale == 0)
                {
                    e[i] = d[l];

                    for (j = 0; j < i; ++j)
                    {
                        d[j] = z[l + j * n];
                        z[i + j * n] = 0;
                        z[j + i * n] = 0;
                    }
                    d[i] = h;
                    continue;

                }
                for (k = 0; k < i; ++k)
                {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }

                Float f = d[l];
                g = CopySign((Float)Math.Sqrt(h), f);
                e[i] = scale * g;
                h -= f * g;
                d[l] = f - g;
                //     .......... form a*u ..........
                for (j = 0; j < i; ++j)
                {
                    e[j] = 0;
                }

                for (j = 0; j < i; ++j)
                {
                    f = d[j];
                    z[j + i * n] = f;
                    g = e[j] + z[j + j * n] * f;
                    if (j + 1 == i)
                    {
                        e[j] = g;
                        continue;
                    }

                    for (k = j + 1; k < i; ++k)
                    {
                        g += z[k + j * n] * d[k];
                        e[k] += z[k + j * n] * f;
                    }

                    e[j] = g;
                }
                //     .......... form p ..........
                f = 0;

                for (j = 0; j < i; ++j)
                {
                    e[j] /= h;
                    f += e[j] * d[j];
                }

                Float hh = f / (h + h);
                //     .......... form q ..........
                for (j = 0; j < i; ++j)
                {
                    e[j] -= hh * d[j];
                }
                //     .......... form reduced a ..........
                for (j = 0; j < i; ++j)
                {
                    f = d[j];
                    g = e[j];

                    for (k = j; k < i; ++k)
                    {
                        z[k + j * n] = (float)((double)z[k + j * n] - (double)f * e[k] - (double)g * d[k]);
                    }

                    d[j] = z[l + j * n];
                    z[i + j * n] = 0;
                }

                d[i] = h;
            }

            //     .......... accumulation of transformation matrices ..........

            for (i = 1; i < n; ++i)
            {
                l = i - 1;
                z[n - 1 + l * n] = z[l + l * n];
                z[l + l * n] = 1;
                h = d[i];
                if (h != 0)
                {
                    for (k = 0; k < i; ++k)
                    {
                        d[k] = z[k + i * n] / h;
                    }

                    for (j = 0; j < i; ++j)
                    {
                        g = 0;

                        for (k = 0; k < i; ++k)
                        {
                            g += z[k + i * n] * z[k + j * n];
                        }

                        for (k = 0; k < i; ++k)
                        {
                            z[k + j * n] -= g * d[k];
                        }
                    }
                }

                for (k = 0; k < i; ++k)
                {
                    z[k + i * n] = 0;
                }
            }

            for (i = 0; i < n; ++i)
            {
                d[i] = z[n - 1 + i * n];
                z[n - 1 + i * n] = 0;
            }
            z[n * n - 1] = 1;
            e[0] = 0;
        } /* Tred */

        /* Subroutine */
        private static int Imtql(Float[] d, Float[] e, Float[] z, int n)
        {
            /* Local variables */
            double b;
            double c;
            double f;
            double g;
            int i;
            int j;
            int k;
            int l;
            int m;
            double p;
            double r;
            double s;
            double tst1;
            double tst2;

            /*     this subroutine is a translation of the algol procedure imtql2, */
            /*     num. math. 12, 377-383(1968) by martin and wilkinson, */
            /*     as modified in num. math. 15, 450(1970) by dubrulle. */
            /*     handbook for auto. comp., vol.ii-linear algebra, 241-248(1971). */

            /*     this subroutine finds the eigenvalues and eigenvectors */
            /*     of a symmetric tridiagonal matrix by the implicit ql method. */
            /*     the eigenvectors of a full symmetric matrix can also */
            /*     be found if  tred2  has been used to reduce this */
            /*     full matrix to tridiagonal form. */

            /*     on input */

            /*        nm must be set to the row dimension of two-dimensional */
            /*          array parameters as declared in the calling program */
            /*          dimension statement. */

            /*        n is the order of the matrix. */

            /*        d contains the diagonal elements of the input matrix. */

            /*        e contains the subdiagonal elements of the input matrix */
            /*          in its last n-1 positions. e(1) is arbitrary. */

            /*        z contains the transformation matrix produced in the */
            /*          reduction by  tred2, if performed. if the eigenvectors */
            /*          of the tridiagonal matrix are desired, z must contain */
            /*          the identity matrix. */

            /*      on output */

            /*        d contains the eigenvalues in ascending order. if an */
            /*          error exit is made, the eigenvalues are correct but */
            /*          unordered for indices 1,2,...,ierr-1. */

            /*        e has been destroyed. */

            /*        z contains orthonormal eigenvectors of the symmetric */
            /*          tridiagonal (or full) matrix. if an error exit is made, */
            /*          z contains the eigenvectors associated with the stored */
            /*          eigenvalues. */

            /*        ierr is set to */
            /*          zero       for normal return, */
            /*          j          if the j-th eigenvalue has not been */
            /*                     determined after 30 iterations. */

            /*     calls pythag for  dsqrt(a*a + b*b) . */

            /*     questions and comments should be directed to burton s. garbow, */
            /*     mathematics and computer science div, argonne national laboratory */

            /*     this version dated august 1983. */

            /*     ------------------------------------------------------------------ */

            /* Function Body */
            if (n == 1)
                return 0;
            for (i = 1; i < n; ++i)
            {
                e[i - 1] = e[i];
            }
            e[n - 1] = (Float)(0.0);

            for (l = 0; l < n; ++l)
            {
                j = 0;
                do
                {
                    /*     .......... look for small sub-diagonal element .......... */
                    for (m = l; m + 1 < n; ++m)
                    {
                        tst1 = Math.Abs(d[m]) + Math.Abs(d[m + 1]);
                        tst2 = tst1 + Math.Abs(e[m]);
                        if (tst2 == tst1)
                            break;
                    }
                    p = d[l];
                    if (m != l)
                    {
                        if (j++ >= 30)
                        {
                            return l;
                        }
                        /*     .......... form shift .......... */
                        g = (d[l + 1] - p) / (e[l] * (Float)(2.0));
                        r = Hypot((float)g, (Float)(1.0));
                        g = d[m] - p + e[l] / (g + CopySign((float)r, (float)g));
                        s = (Float)(1.0);
                        c = (Float)(1.0);
                        p = (Float)(0.0);
                        /*     .......... for i=m-1 step -1 until l do -- .......... */
                        for (i = m - 1; i >= l; i--)
                        {
                            f = s * e[i];
                            b = c * e[i];
                            r = Hypot((float)f, (float)g);
                            e[i + 1] = (float)r;
                            if (r == (Float)(0.0))
                            {
                                /*     .......... recover from underflow .......... */
                                d[i + 1] -= (float)p;
                                e[m] = 0;
                                break;
                            }
                            s = f / r;
                            c = g / r;
                            g = d[i + 1] - p;
                            r = (d[i] - g) * s + c * (Float)(2.0) * b;
                            p = s * r;
                            d[i + 1] = (float)(g + p);
                            g = c * r - b;
                            /*     .......... form vector .......... */
                            for (k = 0; k < n; ++k)
                            {
                                f = z[k + (i + 1) * n];
                                z[k + (i + 1) * n] = (float)(s * z[k + i * n] + c * f);
                                z[k + i * n] = (float)(c * z[k + i * n] - s * f);
                            }
                        }
                        if (r == (Float)(0.0) && i >= l)
                            continue;
                        d[l] -= (float)p;
                        e[l] = (float)g;
                        e[m] = (Float)(0.0);
                    }
                } while (m != l);
            }
            /*     .......... order eigenvalues and eigenvectors .......... */
            for (i = 0; i < n; ++i)
            {
                k = i;
                p = d[i];

                for (j = i + 1; j < n; ++j)
                {
                    if (d[j] <= p)
                        continue;
                    k = j;
                    p = d[j];
                }

                if (k == i)
                    continue;
                d[k] = d[i];
                d[i] = (float)p;

                for (j = 0; j < n; ++j)
                {
                    p = z[j + i * n];
                    z[j + i * n] = z[j + k * n];
                    z[j + k * n] = (float)p;
                }
            }

            return 0;
        }

        private const string DllName = "MklImports";

        public enum Layout
        {
            RowMajor = 101,
            ColMajor = 102
        }

        public enum Job : byte
        {
            EigenValues = (byte)'E',
            Schur = (byte)'S'
        }

        public enum Compz : byte
        {
            None = (byte)'N',
            SchurH = (byte)'I',
            SchurA = (byte)'V'
        }

        public enum Uplo : byte
        {
            UpperTriangular = (byte)'U',
            LowerTriangular = (byte)'L'
        }

        // See: https://software.intel.com/en-us/node/521087#4C9F4214-70BC-4483-A814-1E7F927B30CF
        [DllImport(DllName, EntryPoint = "LAPACKE_shseqr", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Shseqr(Layout matrixLayout, Job job, Compz compz, int n, int ilo, int ihi,
            [In] float[] h, int idh, [Out] float[] wr, [Out] float[] wi, [Out] float[] z, int ldz);

        // See: https://software.intel.com/en-us/node/521087#4C9F4214-70BC-4483-A814-1E7F927B30CF
        [DllImport(DllName, EntryPoint = "LAPACKE_dhseqr", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Dhseqr(Layout matrixLayout, Job job, Compz compz, int n, int ilo, int ihi,
            [In] double[] h, int idh, [Out] double[] wr, [Out] double[] wi, [Out] double[] z, int ldz);

        // See: https://software.intel.com/en-us/node/521046#7EF85A82-423A-4ABC-A208-88326CD0B887
        [DllImport(DllName, EntryPoint = "LAPACKE_ssytrd", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Ssytrd(Layout matrixLayout, Uplo uplo, int n, float[] a, int lda, float[] d,
            float[] e, float[] tau);

        // See: https://software.intel.com/en-us/node/521046#7EF85A82-423A-4ABC-A208-88326CD0B887
        [DllImport(DllName, EntryPoint = "LAPACKE_dsytrd", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Dsytrd(Layout matrixLayout, Uplo uplo, int n, double[] a, int lda, double[] d,
            double[] e, double[] tau);

        // See: https://software.intel.com/en-us/node/521067#E2C5B8B3-D275-4000-821D-1ABF245D2E30
        [DllImport(DllName, EntryPoint = "LAPACKE_ssteqr", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Ssteqr(Layout matrixLayout, Compz compz, int n, float[] d, float[] e, float[] z,
            int ldz);

        // See: https://software.intel.com/en-us/node/521067#E2C5B8B3-D275-4000-821D-1ABF245D2E30
        [DllImport(DllName, EntryPoint = "LAPACKE_dsteqr", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Dsteqr(Layout matrixLayout, Compz compz, int n, double[] d, double[] e, double[] z,
            int ldz);

        // See: https://software.intel.com/en-us/node/521049#106F8646-1C99-4A9D-8604-D60DAAF7BE0C
        [DllImport(DllName, EntryPoint = "LAPACKE_sorgtr", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Sorgtr(Layout matrixLayout, Uplo uplo, int n, float[] a, int lda, float[] tau);

        // See: https://software.intel.com/en-us/node/521049#106F8646-1C99-4A9D-8604-D60DAAF7BE0C
        [DllImport(DllName, EntryPoint = "LAPACKE_dorgtr", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Dorgtr(Layout matrixLayout, Uplo uplo, int n, double[] a, int lda, double[] tau);

        public static bool MklSymmetricEigenDecomposition(Single[] input, int size, out Single[] eigenValues, out Single[] eigenVectors)
        {
            Contracts.CheckParam(size > 0, nameof(size), "The input matrix size must be strictly positive.");
            var n2 = size * size;
            Contracts.Check(Utils.Size(input) >= n2, "The input matrix must at least have " + n2 + " elements");

            eigenValues = null;
            eigenVectors = null;
            if (size == 1)
            {
                eigenValues = new[] { input[0] };
                eigenVectors = new[] { 1f };
                return true;
            }

            Double[] a = new Double[n2];
            Array.Copy(input, 0, a, 0, n2);
            Double[] d = new Double[size];
            Double[] e = new Double[size - 1];
            Double[] tau = new Double[size];
            int info;

            info = Dsytrd(Layout.ColMajor, Uplo.UpperTriangular, size, a, size, d, e, tau);
            if (info != 0)
                return false;

            info = Dorgtr(Layout.ColMajor, Uplo.UpperTriangular, size, a, size, tau);
            if (info != 0)
                return false;

            info = Dsteqr(Layout.ColMajor, Compz.SchurA, size, d, e, a, size);
            if (info != 0)
                return false;

            eigenValues = new Single[size];
            for (var i = 0; i < size; ++i)
                eigenValues[i] = (Single)d[i];

            eigenVectors = new Single[n2];
            for (var i = 0; i < n2; ++i)
                eigenVectors[i] = (Single)a[i];

            return true;
        }

    }
}