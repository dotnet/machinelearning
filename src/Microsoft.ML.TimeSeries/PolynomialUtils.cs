// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    public static class PolynomialUtils
    {
        // Part 1: Computing the polynomial real and complex roots from its real coefficients

        private static Double _tol;

        private static bool IsZero(double x)
        {
            return Math.Abs(x) <= _tol;
        }

        internal static void FindQuadraticRoots(Double b, Double c, out Complex root1, out Complex root2)
        {
            var delta = b * b - 4 * c;
            var sqrtDelta = Math.Sqrt(Math.Abs(delta));

            if (delta >= 0)
            {
                root1 = new Complex((-b + sqrtDelta) / 2, 0);
                root2 = new Complex((-b - sqrtDelta) / 2, 0);
            }
            else
            {
                root1 = new Complex(-b / 2, sqrtDelta / 2);
                root2 = new Complex(-b / 2, -sqrtDelta / 2);
            }
        }

        private static void CreateFullCompanionMatrix(Double[] coefficients, ref Double[] companionMatrix)
        {
            Contracts.Assert(Utils.Size(coefficients) > 1);

            var n = coefficients.Length;
            var n2 = n * n;
            if (Utils.Size(companionMatrix) < n2)
                companionMatrix = new Double[n2];

            int i;
            for (i = 1; i <= n - 1; ++i)
                companionMatrix[n * (i - 1) + i] = 1;

            for (i = 0; i < n; ++i)
                companionMatrix[n2 - n + i] = -coefficients[i];
        }

        /// <summary>
        /// Computes the real and the complex roots of a real monic polynomial represented as:
        /// coefficients[0] + coefficients[1] * X + coefficients[2] * X^2 + ... + coefficients[n-1] * X^(n-1) + X^n
        /// by computing the eigenvalues of the Companion matrix. (https://en.wikipedia.org/wiki/Companion_matrix)
        /// </summary>
        /// <param name="coefficients">The monic polynomial coefficients in the ascending order</param>
        /// <param name="roots">The computed (complex) roots</param>
        /// <param name="roundOffDigits">The number decimal digits to keep after round-off</param>
        /// <param name="doublePrecision">The machine precision</param>
        /// <returns>A boolean flag indicating whether the algorithm was successful.</returns>
        public static bool FindPolynomialRoots(Double[] coefficients, ref Complex[] roots,
            int roundOffDigits = 6, Double doublePrecision = 2.22 * 1e-100)
        {
            Contracts.CheckParam(doublePrecision > 0, nameof(doublePrecision), "The double precision must be positive.");
            Contracts.CheckParam(Utils.Size(coefficients) >= 1, nameof(coefficients), "There must be at least one input coefficient.");

            int i;
            int n = coefficients.Length;
            bool result = true;

            _tol = doublePrecision;

            if (Utils.Size(roots) < n)
                roots = new Complex[n];

            // Extracting the zero roots
            for (i = 0; i < n; ++i)
            {
                if (IsZero(coefficients[i]))
                    roots[n - i - 1] = Complex.Zero;
                else
                    break;
            }

            if (i == n) // All zero roots
                return true;

            if (i == n - 1) // Handling the linear case
                roots[0] = new Complex(-coefficients[i], 0);
            else if (i == n - 2) // Handling the quadratic case
                FindQuadraticRoots(coefficients[i + 1], coefficients[i], out roots[0], out roots[1]);
            else // Handling higher-order cases by computing the eigenvalues of the Companion matrix
            {
                var coeff = coefficients;
                if (i > 0)
                {
                    coeff = new Double[n - i];
                    Array.Copy(coefficients, i, coeff, 0, n - i);
                }

                // REVIEW: the eigen decomposition of the companion matrix should be done using the FactorizedCompanionMatrix class
                // instead of MKL.
                //FactorizedCompanionMatrix companionMatrix = new FactorizedCompanionMatrix(coeff);
                //result = companionMatrix.ComputeEigenValues(ref roots);

                Double[] companionMatrix = null;
                var realPart = new Double[n - i];
                var imaginaryPart = new Double[n - i];
                var dummy = new Double[1];

                CreateFullCompanionMatrix(coeff, ref companionMatrix);
                var info = EigenUtils.Dhseqr(EigenUtils.Layout.ColMajor, EigenUtils.Job.EigenValues, EigenUtils.Compz.None,
                    n - i, 1, n - i, companionMatrix, n - i, realPart, imaginaryPart, dummy, n - i);

                if (info != 0)
                    return false;

                for (var j = 0; j < n - i; ++j)
                    roots[j] = new Complex(realPart[j], imaginaryPart[j]);
            }

            return result;
        }

        // Part 2: Computing the polynomial coefficients from its real and complex roots
        private sealed class FactorMultiplicity
        {
            public int Multiplicity;

            public FactorMultiplicity(int multiplicity = 1)
            {
                Contracts.Assert(multiplicity > 0);
                Multiplicity = multiplicity;
            }
        }

        private sealed class PolynomialFactor
        {
            public List<decimal> Coefficients;
            public static decimal[] Destination;

            private decimal _key;
            public decimal Key { get { return _key; } }

            private void SetKey()
            {
                decimal absVal = -1;
                for (var i = 0; i < Coefficients.Count; ++i)
                {
                    var temp = Math.Abs(Coefficients[i]);
                    if (temp > absVal)
                    {
                        absVal = temp;
                        _key = Coefficients[i];
                    }
                }
            }

            public PolynomialFactor(decimal[] coefficients)
            {
                Coefficients = new List<decimal>(coefficients);
                SetKey();
            }

            internal PolynomialFactor(decimal key)
            {
                _key = key;
            }

            public void Multiply(PolynomialFactor factor)
            {
                var len = Coefficients.Count;
                Coefficients.AddRange(factor.Coefficients);

                PolynomialMultiplication(0, len, len, factor.Coefficients.Count, 0, 1, 1);

                for (var i = 0; i < Coefficients.Count; ++i)
                    Coefficients[i] = Destination[i];

                SetKey();
            }

            private void PolynomialMultiplication(int uIndex, int uLen, int vIndex, int vLen, int dstIndex, decimal uCoeff, decimal vCoeff)
            {
                Contracts.Assert(uIndex >= 0);
                Contracts.Assert(uLen >= 1);
                Contracts.Assert(uIndex + uLen <= Utils.Size(Coefficients));
                Contracts.Assert(vIndex >= 0);
                Contracts.Assert(vLen >= 1);
                Contracts.Assert(vIndex + vLen <= Utils.Size(Coefficients));
                Contracts.Assert(uIndex + uLen <= vIndex || vIndex + vLen <= uIndex); // makes sure the input ranges are non-overlapping.
                Contracts.Assert(dstIndex >= 0);
                Contracts.Assert(dstIndex + uLen + vLen <= Utils.Size(Destination));

                if (uLen == 1 && vLen == 1)
                {
                    Destination[dstIndex] = Coefficients[uIndex] * Coefficients[vIndex];
                    Destination[dstIndex + 1] = Coefficients[uIndex] + Coefficients[vIndex];
                }
                else
                    NaivePolynomialMultiplication(uIndex, uLen, vIndex, vLen, dstIndex, uCoeff, vCoeff);
            }

            private void NaivePolynomialMultiplication(int uIndex, int uLen, int vIndex, int vLen, int dstIndex, decimal uCoeff, decimal vCoeff)
            {
                int i;
                int j;
                int a;
                int b;
                int c;
                var len = uLen + vLen - 1;
                decimal temp;

                if (vLen < uLen)
                {
                    var t = vLen;
                    vLen = uLen;
                    uLen = t;

                    t = vIndex;
                    vIndex = uIndex;
                    uIndex = t;
                }

                for (i = 0; i <= len; ++i)
                {
                    b = Math.Min(uLen, i + 1) - 1;
                    a = i >= Math.Max(uLen, vLen) ? len - i : b + 1;
                    c = Math.Max(0, i - uLen + 1);
                    temp = 0;

                    if (i >= uLen)
                        temp = uCoeff * Coefficients[i - uLen + vIndex];

                    if (i >= vLen)
                        temp += (vCoeff * Coefficients[i - vLen + uIndex]);

                    for (j = 0; j < a; ++j)
                        temp += (Coefficients[b - j + uIndex] * Coefficients[c + j + vIndex]);

                    Destination[i + dstIndex] = temp;
                }
            }
        }

        private sealed class ByMaximumCoefficient : IComparer<PolynomialFactor>
        {
            public int Compare(PolynomialFactor x, PolynomialFactor y)
            {
                if (x.Key > y.Key)
                    return 1;

                if (x.Key < y.Key)
                    return -1;

                return 0;
            }
        }

        /// <summary>
        /// Computes the coefficients of a real monic polynomial given its real and complex roots.
        /// The final monic polynomial is represented as:
        /// coefficients[0] + coefficients[1] * X + coefficients[2] * X^2 + ... + coefficients[n-1] * X^(n-1) + X^n
        ///
        /// Note: the constant 1 coefficient of the highest degree term is implicit and not included in the output of the method.
        /// </summary>
        /// <param name="roots">The input (complex) roots</param>
        /// <param name="coefficients">The output real coefficients</param>
        /// <returns>A boolean flag indicating whether the algorithm was successful.</returns>
        public static bool FindPolynomialCoefficients(Complex[] roots, ref Double[] coefficients)
        {
            Contracts.CheckParam(Utils.Size(roots) > 0, nameof(roots), "There must be at least 1 input root.");

            int i;
            int n = roots.Length;
            var hash = new Dictionary<Complex, FactorMultiplicity>();
            int destinationOffset = 0;

            var factors = new List<PolynomialFactor>();

            for (i = 0; i < n; ++i)
            {
                if (Double.IsNaN(roots[i].Real) || Double.IsNaN(roots[i].Imaginary))
                    return false;

                if (roots[i].Equals(Complex.Zero)) // Zero roots
                    destinationOffset++;
                else if (roots[i].Imaginary == 0) // Real roots
                {
                    var f = new PolynomialFactor(new[] { (decimal)-roots[i].Real });
                    factors.Add(f);
                }
                else // Complex roots
                {
                    var conj = Complex.Conjugate(roots[i]);
                    FactorMultiplicity temp;
                    if (hash.TryGetValue(conj, out temp))
                    {
                        temp.Multiplicity--;

                        var f = new PolynomialFactor(new[]
                        {
                            (decimal) (roots[i].Real*roots[i].Real + roots[i].Imaginary*roots[i].Imaginary),
                            (decimal) (-2*roots[i].Real)
                        });

                        factors.Add(f);

                        if (temp.Multiplicity <= 0)
                            hash.Remove(conj);
                    }
                    else
                    {
                        if (hash.TryGetValue(roots[i], out temp))
                            temp.Multiplicity++;
                        else
                            hash.Add(roots[i], new FactorMultiplicity());
                    }
                }
            }

            if (hash.Count > 0)
                return false;

            var comparer = new ByMaximumCoefficient();

            factors.Sort(comparer);

            if (destinationOffset < n - 1)
            {
                if (Utils.Size(PolynomialFactor.Destination) < n)
                    PolynomialFactor.Destination = new decimal[n];

                while (factors.Count > 1)
                {
                    var k1 = Math.Abs(factors.ElementAt(0).Key);
                    var k2 = Math.Abs(factors.ElementAt(factors.Count - 1).Key);

                    PolynomialFactor f1;
                    if (k1 < k2)
                    {
                        f1 = factors.ElementAt(0);
                        factors.RemoveAt(0);
                    }
                    else
                    {
                        f1 = factors.ElementAt(factors.Count - 1);
                        factors.RemoveAt(factors.Count - 1);
                    }

                    var ind = factors.BinarySearch(new PolynomialFactor(-f1.Key), comparer);
                    if (ind < 0)
                        ind = ~ind;

                    ind = Math.Min(factors.Count - 1, ind);
                    var f2 = factors.ElementAt(ind);
                    factors.RemoveAt(ind);

                    f1.Multiply(f2);

                    ind = factors.BinarySearch(f1, comparer);
                    if (ind >= 0)
                        factors.Insert(ind, f1);
                    else
                        factors.Insert(~ind, f1);
                }
            }

            if (Utils.Size(coefficients) < n)
                coefficients = new Double[n];

            for (i = 0; i < destinationOffset; ++i)
                coefficients[i] = 0;

            if (destinationOffset < n)
            {
                var coeff = factors.ElementAt(0).Coefficients;
                for (i = destinationOffset; i < n; ++i)
                    coefficients[i] = Decimal.ToDouble(coeff[i - destinationOffset]);
            }

            return true;
        }
    }
}
