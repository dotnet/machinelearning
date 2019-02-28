// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath
{
    [BestFriend]
    // REVIEW: improve perf with SSE and Multithreading
    internal static class EigenUtils
    {
        //Compute the Eigen-decomposition of a symmetric matrix
        // REVIEW: use matrix/vector operations, not Array Math
        public static void EigenDecomposition(float[] a, out float[] eigenvalues, out float[] eigenvectors)
        {
            var count = a.Length;
            var n = (int)Math.Sqrt(count);
            Contracts.Assert(n * n == count);

            eigenvectors = new float[count];
            eigenvalues = new float[n];

            //Reduce A to tridiagonal form
            // REVIEW: it's not ideal to keep using the same variable name for different purposes
            // - After the operation, "eigenvalues" means the diagonal elements of the reduced matrix
            //and "eigenvectors" means the orthogonal similarity transformation matrix
            // - Consider aliasing variables
            var w = new float[n];
            Tred(a, eigenvalues, w, eigenvectors, n);

            //Eigen-decomposition of the tridiagonal matrix
            //After this operation, "eigenvalues" means eigenvalues^2
            Imtql(eigenvalues, w, eigenvectors, n);

            for (int i = 0; i < n; i++)
                eigenvalues[i] = eigenvalues[i] <= 0 ? (float)(0.0) : (float)Math.Sqrt(eigenvalues[i]);
        }

        private static float Hypot(float x, float y)
        {
            x = Math.Abs(x);
            y = Math.Abs(y);

            if (x == 0 || y == 0)
                return x + y;

            if (x < y)
            {
                float t = x / y;
                return y * (float)Math.Sqrt(1 + t * t);
            }
            else
            {
                float t = y / x;
                return x * (float)Math.Sqrt(1 + t * t);
            }
        }

        private static float CopySign(float x, float y)
        {
            float xx = Math.Abs(x);
            return y < 0 ? -xx : xx;
        }

        private static void Tred(float[] a, float[] d, float[] e, float[] z, int n)
        {
            Double g;
            Double h;
            int i;
            int j;
            int k;
            int l;

            /*     this subroutine reduces a float symmetric matrix to a */
            /*     symmetric tridiagonal matrix using and accumulating */
            /*     orthogonal similarity transformations. */

            /*     on input */

            /*	  n is the order of the matrix. */

            /*	  a contains the float symmetric input matrix. only the */
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
            for (i = n; i-- > 1; )
            {
                l = i - 1;
                h = 0;
                Double scale = 0;
                if (l == 0)
                {
                    e[1] = d[0];
                    d[0] = z[0];
                    z[1] = 0;
                    z[n] = 0;
                    d[1] = (float)h;
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
                    d[i] = (float)h;
                    continue;

                }
                for (k = 0; k < i; ++k)
                {
                    d[k] = (float) (d[k]/scale);
                    h += d[k] * d[k];
                }

                Double f = d[l];
                g = CopySign((float)Math.Sqrt(h), (float)f);
                e[i] = (float)(scale * g);
                h -= f * g;
                d[l] = (float)(f - g);
                //     .......... form a*u ..........
                for (j = 0; j < i; ++j)
                {
                    e[j] = 0;
                }

                for (j = 0; j < i; ++j)
                {
                    f = d[j];
                    z[j + i * n] = (float)f;
                    g = e[j] + z[j + j * n] * f;
                    if (j + 1 == i)
                    {
                        e[j] = (float)g;
                        continue;
                    }

                    for (k = j + 1; k < i; ++k)
                    {
                        g += z[k + j * n] * d[k];
                        e[k] = (float)(e[k] + z[k + j * n] * f);
                    }

                    e[j] = (float)g;
                }
                //     .......... form p ..........
                f = 0;

                for (j = 0; j < i; ++j)
                {
                    e[j] = (float)(e[j] / h);
                    f += e[j] * d[j];
                }

                Double hh = f / (h + h);
                //     .......... form q ..........
                for (j = 0; j < i; ++j)
                {
                    e[j] = (float)(e[j] - hh * d[j]);
                }
                //     .......... form reduced a ..........
                for (j = 0; j < i; ++j)
                {
                    f = d[j];
                    g = e[j];

                    for (k = j; k < i; ++k)
                    {
                        z[k + j * n] = (float)(z[k + j * n] - f * e[k] - g * d[k]);
                    }

                    d[j] = z[l + j * n];
                    z[i + j * n] = 0;
                }

                d[i] = (float)h;
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
                        d[k] = (float)(z[k + i * n] / h);
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
                            z[k + j * n] = (float)(z[k + j * n] - g * d[k]);
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
        private static int Imtql(float[] d, float[] e, float[] z, int n)
        {
            /* Local variables */
            float b;
            float c;
            float f;
            float g;
            int i;
            int j;
            int k;
            int l;
            int m;
            float p;
            float r;
            float s;
            float tst1;
            float tst2;

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
            e[n - 1] = (float)(0.0);

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
                        g = (d[l + 1] - p) / (e[l] * (float)(2.0));
                        r = Hypot(g, (float)(1.0));
                        g = d[m] - p + e[l] / (g + CopySign(r, g));
                        s = (float)(1.0);
                        c = (float)(1.0);
                        p = (float)(0.0);
                        /*     .......... for i=m-1 step -1 until l do -- .......... */
                        for (i = m - 1; i >= l; i--)
                        {
                            f = s * e[i];
                            b = c * e[i];
                            r = Hypot(f, g);
                            e[i + 1] = r;
                            if (r == (float)(0.0))
                            {
                                /*     .......... recover from underflow .......... */
                                d[i + 1] -= p;
                                e[m] = 0;
                                break;
                            }
                            s = f / r;
                            c = g / r;
                            g = d[i + 1] - p;
                            r = (d[i] - g) * s + c * (float)(2.0) * b;
                            p = s * r;
                            d[i + 1] = g + p;
                            g = c * r - b;
                            /*     .......... form vector .......... */
                            for (k = 0; k < n; ++k)
                            {
                                f = z[k + (i + 1) * n];
                                z[k + (i + 1) * n] = s * z[k + i * n] + c * f;
                                z[k + i * n] = c * z[k + i * n] - s * f;
                            }
                        }
                        if (r == (float)(0.0) && i >= l)
                            continue;
                        d[l] -= p;
                        e[l] = g;
                        e[m] = (float)(0.0);
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
                d[i] = p;

                for (j = 0; j < n; ++j)
                {
                    p = z[j + i * n];
                    z[j + i * n] = z[j + k * n];
                    z[j + k * n] = p;
                }
            }

            return 0;
        }
    }
}