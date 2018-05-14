// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// Some useful math methods.
    /// </summary>
    public static class MathUtils
    {
        public static Float ToFloat(this Double dbl)
        {
            return (Single)dbl;
        }

        // The purpose of this is to catch (at compile time) invocations of ToFloat
        // that are not appropriate. Note that the return type is void.
        public static void ToFloat(this Single dbl)
        {
            Contracts.Assert(false, "Bad use of ToFloat");
            throw Contracts.Except();
        }

        public static Float Sqrt(Float x)
        {
            return Math.Sqrt(x).ToFloat();
        }

        public static Float Log(Float x)
        {
            return Math.Log(x).ToFloat();
        }

        public static Float Log(Float a, Float newBase)
        {
            return Math.Log(a, newBase).ToFloat();
        }

        public static Float Pow(Float x, Float y)
        {
            return Math.Pow(x, y).ToFloat();
        }

        /// <summary>
        /// Finds the best least-squares fit of y = ax + b
        /// </summary>
        /// <param name="x">The x values.</param>
        /// <param name="y">The y values.</param>
        /// <param name="a">The coefficent a.</param>
        /// <param name="b">The intercept b.</param>
        public static void SimpleLinearRegression(Float[] x, Float[] y, out Float a, out Float b)
        {
            Contracts.CheckValue(x, nameof(x));
            Contracts.CheckValue(y, nameof(y));
            Contracts.Check(x.Length == y.Length, "Input and output must be same length.");

            int m = x.Length;

            Float sumSqX = 0;
            Float sumX = 0;
            Float sumXY = 0;
            Float sumY = 0;
            Float sumSqY = 0;
            for (int i = 0; i < m; i++)
            {
                Float xVal = x[i];
                Float yVal = y[i];
                sumSqX += xVal * xVal;
                sumX += xVal;
                sumXY += xVal * yVal;
                sumY += yVal;
                sumSqY += yVal * yVal;
            }

            Float denom = sumSqX * m - sumX * sumX;
            a = (sumXY * m - sumY * sumX) / denom;
            b = (sumSqX * sumY - sumXY * sumX) / denom;
        }

        /************************
         * MATH ARRAY FUNCTIONS *
         ************************/
        // REVIEW: Move the vector methods into VectorUtils.

        /// <summary>
        /// The product of elements in a
        /// </summary>
        /// <param name="a">an array</param>
        /// <returns>the product of a's elements</returns>
        public static int Product(int[] a)
        {
            Contracts.AssertValue(a);
            int result = 1;
            foreach (var x in a)
                result = checked(result * x);
            return result;
        }

        /// <summary>
        /// Find the max element of a
        /// </summary>
        /// <param name="a">an array</param>
        /// <returns>the max element</returns>
        public static Float Max(Float[] a)
        {
            Contracts.AssertValue(a);
            Float result = Float.NegativeInfinity;
            foreach (var x in a)
                result = Math.Max(result, x);
            return result;
        }

        /// <summary>
        /// Find the minimum element of a
        /// </summary>
        /// <param name="a">an array</param>
        /// <returns>the minimum element</returns>
        public static Float Min(Float[] a)
        {
            Contracts.AssertValue(a);
            Float result = Float.PositiveInfinity;
            foreach (var x in a)
                result = Math.Min(result, x);
            return result;
        }

        /// <summary>
        /// Finds the first index of the max element of the array.
        /// NaNs are ignored. If all the elements to consider are NaNs, -1 is 
        /// returned. The caller should distinguish in this case between two
        /// possibilities:
        /// 1) The number of the element to consider is zero.
        /// 2) All the elements to consider are NaNs.
        /// </summary>
        /// <param name="a">an array</param>
        /// <returns>the first index of the max element</returns>
        public static int ArgMax(Float[] a)
        {
            return ArgMax(a, Utils.Size(a));
        }

        /// <summary>
        /// Finds the first index of the max element of the array. 
        /// NaNs are ignored. If all the elements to consider are NaNs, -1 is 
        /// returned. The caller should distinguish in this case between two
        /// possibilities:
        /// 1) The number of the element to consider is zero.
        /// 2) All the elements to consider are NaNs.
        /// </summary>
        /// <param name="a">an array</param>
        /// <param name="count">number of the element in the array to consider</param>
        /// <returns>the first index of the max element</returns>
        public static int ArgMax(Float[] a, int count)
        {
            Contracts.Assert(0 <= count && count <= Utils.Size(a));
            if (count == 0)
                return -1;

            int amax = -1;
            Float max = Float.NegativeInfinity;
            for (int i = count - 1; i >= 0; i--)
            {
                if (max <= a[i])
                {
                    amax = i;
                    max = a[i];
                }
            }

            return amax;
        }

        /// <summary>
        /// Finds the first index of the minimum element of the array.
        /// NaNs are ignored. If all the elements to consider are NaNs, -1 is 
        /// returned. The caller should distinguish in this case between two
        /// possibilities:
        /// 1) The number of the element to consider is zero.
        /// 2) All the elements to consider are NaNs.
        /// </summary>
        /// <param name="a">an array</param>
        /// <returns>the first index of the minimum element</returns>
        public static int ArgMin(Float[] a)
        {
            return ArgMin(a, Utils.Size(a));
        }

        /// <summary>
        /// Finds the first index of the minimum element of the array.
        /// NaNs are ignored. If all the elements to consider are NaNs, -1 is 
        /// returned. The caller should distinguish in this case between two
        /// possibilities:
        /// 1) The number of the element to consider is zero.
        /// 2) All the elements to consider are NaNs.
        /// </summary>
        /// <param name="a">an array</param>
        /// <param name="count">number of the element in the array to consider</param>
        /// <returns>the first index of the minimum element</returns>
        public static int ArgMin(Float[] a, int count)
        {
            Contracts.Assert(0 <= count && count <= Utils.Size(a));
            if (count == 0)
                return -1;

            int amin = -1;
            Float min = Float.PositiveInfinity;
            for (int i = count - 1; i >= 0; i--)
            {
                if (min >= a[i])
                {
                    amin = i;
                    min = a[i];
                }
            }

            return amin;
        }

        /*****************
         * LOG FUNCTIONS *
         *****************/

        private const Float LogTolerance = 30;

        /// <summary>
        /// computes the "softmax" function: log sum_i exp x_i
        /// </summary>
        /// <param name="inputs">Array of numbers to softmax</param>
        /// <param name="count">the number of input array elements to process</param>
        /// <returns>the softmax of the numbers</returns>
        /// <remarks>may have slightly lower roundoff error if inputs are sorted, smallest first</remarks>
        public static Float SoftMax(Float[] inputs, int count)
        {
            Contracts.AssertValue(inputs);
            Contracts.Assert(0 < count & count <= inputs.Length);

            int maxIdx = 0;
            Float max = Float.NegativeInfinity;
            for (int i = 0; i < count; i++)
            {
                if (inputs[i] > max)
                {
                    maxIdx = i;
                    max = inputs[i];
                }
            }

            if (Float.IsNegativeInfinity(max))
                return Float.NegativeInfinity;

            if (count == 1)
                return max;

            //else if (leng == 2) {
            //  return SoftMax(inputs[0], inputs[1]);
            //}

            double intermediate = 0.0;
            Float cutoff = max - LogTolerance;

            for (int i = 0; i < count; i++)
            {
                if (i == maxIdx)
                    continue;
                if (inputs[i] > cutoff)
                    intermediate += Math.Exp(inputs[i] - max);
            }

            if (intermediate > 0.0)
                return (Float)(max + Math.Log(1.0 + intermediate));
            return max;
        }

        /// <summary>
        /// computes "softmax" function of two arguments: log (exp x + exp y)
        /// </summary>
        public static Float SoftMax(Float lx, Float ly)
        {
            Float max;
            Float negDiff;
            if (lx > ly)
            {
                max = lx;
                negDiff = ly - lx;
            }
            else
            {
                max = ly;
                negDiff = lx - ly;
            }
            if (Float.IsNegativeInfinity(max) || negDiff < -LogTolerance)
            {
                return max;
            }
            else
            {
                return (Float)(max + Math.Log(1.0 + Math.Exp(negDiff)));
            }
        }

        /*******************
         * OTHER FUNCTIONS *
         *******************/

        public const Float DefaultMaxRelativeErr = (Float)1e-8;
        public const Float DefaultMaxAbsErr = (Float)1e-12;

        /// <summary>
        /// true if two Float values are close (using relative comparison)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool AlmostEqual(Float a, Float b)
        {
            return AlmostEqual(a, b, DefaultMaxRelativeErr, DefaultMaxAbsErr);
        }

        public static bool AlmostEqual(Float a, Float b, Float maxRelErr, Float maxAbsError)
        {
            Contracts.Assert(FloatUtils.IsFinite(maxRelErr));
            Contracts.Assert(FloatUtils.IsFinite(maxAbsError));

            Float absDiff = Math.Abs(a - b);
            if (absDiff < maxAbsError)
                return true;
            Float maxAbs = Math.Max(Math.Abs(a), Math.Abs(b));
            return (absDiff / maxAbs) <= maxRelErr;
        }

        private readonly static int[] _possiblePrimeMod30 = new int[] { 1, 7, 11, 13, 17, 19, 23, 29 };
        private readonly static double _constantForLogGamma = 0.5 * Math.Log(2 * Math.PI);
        private readonly static double[] _coeffsForLogGamma = { 12.0, -360.0, 1260.0, -1680.0, 1188.0 };

        /// <summary>
        /// Returns the log of the gamma function, using the Stirling approximation
        /// </summary>
        /// <param name="x">Argument of function</param>
        /// <returns>Log Gamma(x)</returns>
        /// <remarks>Accurate to eight digits for all x.</remarks>
        public static double LogGamma(double x)
        {
            Contracts.CheckParam(x > 0, nameof(x), "LogGamma invalid for x <= 0");

            double res = 0;
            if (x < 6)
            {
                int toAdd = (int)Math.Floor(7 - x);
                double v2 = 1;
                for (int i = 0; i < toAdd; i++)
                    v2 *= (x + i);
                res = -Math.Log(v2);
                x += toAdd;
            }
            x = x - 1;

            res += _constantForLogGamma + (x + 0.5) * Math.Log(x) - x;

            // correction terms
            double xSquared = x * x;
            double pow = x;
            foreach (double coeff in _coeffsForLogGamma)
            {
                double newRes = res + 1.0 / (coeff * pow);
                if (newRes == res)
                {
                    return res;
                }
                res = newRes;
                pow *= xSquared;
            }

            return res;
        }

        private static List<double> _logFactorialCache;
        private const int LogFactorialCacheSize = 1000;

        /// <summary>
        /// Computes the log factorial of n, using fast methods
        /// </summary>
        /// <param name="n">The number to compute the factorial of</param>
        /// <returns>The log factorial of n</returns>
        public static double LogFactorial(int n)
        {
            Contracts.CheckParam(n >= 0, nameof(n), "LogFactorial is invalid for n < 0.");

            if (n >= LogFactorialCacheSize)
                return LogGamma(n + 1);

            if (_logFactorialCache == null)
            {
                _logFactorialCache = new List<double>(LogFactorialCacheSize);
                _logFactorialCache.Add(0);
                _logFactorialCache.Add(0);
            }

            for (int i = _logFactorialCache.Count; i <= n; i++)
            {
                _logFactorialCache.Add(_logFactorialCache[i - 1] + Math.Log(i));
            }

            return _logFactorialCache[n];
        }

        /// <summary>
        /// Returns the two-tailed p-value given a t statistic from a distribution
        /// parameterized by the provided number of degrees of freedom.
        /// </summary>
        /// <param name="t">The t-statistic</param>
        /// <param name="df">The degrees of freedom</param>
        /// <returns>The corresponding two-tailed p-value</returns>
        public static Double TStatisticToPValue(Double t, Double df)
        {
            Contracts.CheckParam(df > 0, nameof(df), "Degrees of freedom must be positive");

            // REVIEW: Of some interest is calculating p-values for infinite
            // df, but this code will not handle that. In fact I have strong concerns
            // for the numeric stability of this method for larger df, since we have
            // a form of catastrophic cancellation for t. It may be worthwhile folding
            // the terms for incomplete beta into this evaluation, if this becomes
            // problematic.
            Double result = IncompleteBeta(df / (df + t * t), df / 2, 0.5);
            // Clamp the output to 0 to 1, in case for numerical reasons it strayed
            // outside these bounds.
            return Math.Max(0, Math.Min(result, 1));
        }

        private delegate Double Sequence(int i);

        /// <summary>
        /// Lentz's algorithm for evaluating the continued fraction
        /// b0 + a1 / (b1 + a2 / (b2 + a3 / (b3 + a4 / ...) ) )
        /// </summary>
        /// <param name="a">The <c>a</c> function mapping positive integers to a sequence term</param>
        /// <param name="b">The <c>b</c> function mapping non-negative integers to a sequence term</param>
        /// <param name="tol">Calculate the continued fraction to this tolerance</param>
        /// <returns>The evaluation of the continued fraction</returns>
        private static Double Lentz(Sequence a, Sequence b, Double tol = 1e-15)
        {
            Contracts.AssertValue(a);
            Contracts.AssertValue(b);
            Contracts.Assert(tol > 0);

            Double f = Unclamp(b(0));
            Double c = f;
            Double d = 0;
            const int iterMax = 100000;
            for (int i = 1; i < iterMax; ++i)
            {
                Double bi = b(i);
                Double ai = a(i);
                d = 1.0 / Unclamp(bi + ai * d);
                c = Unclamp(bi + ai / c);
                Double ratio = c * d;
                f *= ratio;
                if (Math.Abs(ratio - 1.0) < tol)
                    break;
            }
            return f;
        }

        private static Double Beta(Double a, Double b)
        {
            // REVIEW: LogGamma implementation precision is a concern, but
            // is minor, perhaps, compared to the instability of TtoP.
            return Math.Exp(LogGamma(a) + LogGamma(b) - LogGamma(a + b));
        }

        private static Double IncompleteBeta(Double x, Double a, Double b)
        {
            Contracts.Assert(0 <= x && x <= 1);
            Contracts.Assert(0 < a);
            Contracts.Assert(0 < b);
            if (x == 0 || x == 1)
                return x;

            // This implementation of the incomplete beta will converge quickly
            // only if the minimum of a and b is relatively small, that is, less than
            // few thousand or so -- otherwise it converges slowly. If this method
            // is made public, then this case of larger a and b should be handled
            // better.

            // If x > (a+1)/(a+b+2), exploit I_x(a,b) = 1 - I_{x-1}(b,a)...
            // REVIEW: Can we be sure some numerical imprecision "weirdness"
            // won't lead both the above test and (1-x) > (b+1)/(a+b+2) being true?
            // Should we build in some sort of protections?
            if (x * (a + b + 2) > a + 1)
                return 1 - IncompleteBeta(1 - x, b, a);

            // This is a fairly naive implementation of I_x(a,b), using a generic
            // non-purpose optimized application of continued fractions.
            Sequence adel = i =>
            {
                if (i == 1)
                    return 1;
                int m = (i - 1) >> 1;
                Double denom = ((a + i - 2) * (a + i - 1));
                if ((i & 1) == 0)
                    return -(a + m) * (a + b + m) * x / denom;
                return m * (b - m) * x / denom;
            };
            Sequence bdel = i => i == 0 ? 0 : 1;
            return Math.Pow(x, a) * Math.Pow(1 - x, b) / (a * Beta(a, b)) * Lentz(adel, bdel);
        }

        private static Double Unclamp(Double val)
        {
            const Double bound = 1e-30;
            if (!(-bound <= val && val <= bound))
                return val;
            return val < 0 ? -bound : bound;
        }

        /// <summary>
        /// The logistic sigmoid function: 1 / (1 + e^(-x)).
        /// </summary>
        public static Float Sigmoid(Float x)
        {
#if SLOW_EXP
            return SigmoidSlow(x);
#else
            return SigmoidFast(x);
#endif
        }

        /// <summary>
        /// Hyperbolic tangent.
        /// </summary>
        public static Float Tanh(Float x)
        {
#if SLOW_EXP
            return TanhSlow(x);
#else
            return TanhFast(x);
#endif
        }

        /// <summary>
        /// The logistic sigmoid function: 1 / (1 + e^(-x)).
        /// </summary>
        public static Float SigmoidSlow(Float x)
        {
            return 1 / (1 + ExpSlow(-x));
        }

        /// <summary>
        /// Hyperbolic tangent.
        /// </summary>
        public static Float TanhSlow(Float x)
        {
            return Math.Tanh(x).ToFloat();
        }

        /// <summary>
        /// The exponential function: e^(x).
        /// </summary>
        public static Float ExpSlow(Float x)
        {
            return Math.Exp(x).ToFloat();
        }

        private const int ExpInf = 128;
        private const Float Coef1 = (Float)0.013555747234814917704030793;
        private const Float Coef2 = (Float)0.065588116243247810171479524;
        private const Float Coef3 = (Float)0.3069678791803394491901401;

        // 1 / ln(2).
        private const Float RecipLn2 = (Float)1.44269504088896340735992468100;

        private static Float PowerOfTwo(int exp)
        {
            Contracts.Assert(0 <= exp && exp < ExpInf);
            return FloatUtils.GetPowerOfTwoSingle(exp);
        }

        /// <summary>
        /// The logistic sigmoid function: 1 / (1 + e^(-x)).
        /// </summary>
        public static Float SigmoidFast(Float x)
        {
            // This is a loose translation from SSE code
            if (Float.IsNaN(x))
                return x;

            bool neg = false;
            if (x < 0)
            {
                x = -x;
                neg = true;
            }

            // Multiply by 1/ln(2).
            x *= RecipLn2;
            if (x >= ExpInf)
                return neg ? (Float)0 : (Float)1;

            // Get the floor and fractional part.
            int n = (int)x;
            Contracts.Assert(0 <= n && n < ExpInf);
            Float f = x - n;
            Contracts.Assert(0 <= f && f < 1);

            // Get the integer power of two part.
            Float r = PowerOfTwo(n);
            Contracts.Assert(1 <= r && r < Float.PositiveInfinity);

            // This approximates 2^f for 0 <= f <= 1. Note that it is exact at the endpoints.
            Float res = 1 + f + (f - 1) * f * ((Coef1 * f + Coef2) * f + Coef3);

            res = 1 / (1 + r * res);
            if (!neg)
                res = 1 - res;
            return res;
        }

        /// <summary>
        /// The hyperbolic tangent function.
        /// </summary>
        public static Float TanhFast(Float x)
        {
            if (Float.IsNaN(x))
                return x;

            bool neg = false;
            if (x < 0)
            {
                x = -x;
                neg = true;
            }

            // Multiply by 2/ln(2).
            x *= 2 * RecipLn2;
            if (x >= ExpInf)
                return neg ? (Float)(-1) : (Float)1;

            // Get the floor and fractional part.
            int n = (int)x;
            Contracts.Assert(0 <= n && n < ExpInf);
            Float f = x - n;
            Contracts.Assert(0 <= f && f < 1);

            // Get the integer power of two part.
            Float r = PowerOfTwo(n);
            Contracts.Assert(1 <= r && r < Float.PositiveInfinity);

            // This approximates 2^f - 1 for 0 <= f <= 1. Note that it is exact at the endpoints.
            Float res = f + (f - 1) * f * ((Coef1 * f + Coef2) * f + Coef3);

            res *= r;
            res = (res + (r - 1)) / (res + (r + 1));
            if (neg)
                res = -res;
            return res;
        }

        /// <summary>
        /// The exponential function: e^(x).
        /// </summary>
        public static Float ExpFast(Float x)
        {
            if (Float.IsNaN(x))
                return x;

            bool neg = false;
            if (x < 0)
            {
                x = -x;
                neg = true;
            }

            // Multiply by 1/ln(2). Then we need to calculate 2^x.
            x *= RecipLn2;
            if (x >= ExpInf)
                return neg ? (Float)0 : Float.PositiveInfinity;

            // Get the floor and fractional part.
            int n = (int)x;
            Contracts.Assert(0 <= n && n < ExpInf);
            Float f = x - n;
            Contracts.Assert(0 <= f && f < 1);

            // Get the integer power of two part.
            Float r = PowerOfTwo(n);
            Contracts.Assert(1 <= r && r < Float.PositiveInfinity);

            // This approximates 2^f for 0 <= f <= 1. Note that it is exact at the endpoints.
            Float res = 1 + f + (f - 1) * f * ((Coef1 * f + Coef2) * f + Coef3);

            res *= r;
            if (neg)
                res = 1 / res;
            return res;
        }

        /// <summary>
        /// Apply a soft max on an array of Floats. Note that src and dst may be the same array.
        /// </summary>
        public static void ApplySoftMax(Float[] src, Float[] dst)
        {
            Contracts.Assert(src.Length == dst.Length);
            ApplySoftMax(src, dst, 0, src.Length);
        }

        /// <summary>
        /// Apply a soft max on a range within an array of Floats. Note that src and dst may be the same array.
        /// </summary>
        public static void ApplySoftMax(Float[] src, Float[] dst, int start, int end)
        {
            Contracts.Assert(src.Length == dst.Length);
            Contracts.Assert(0 <= start && start <= end && end <= src.Length);

            // Compute max output.
            Float maxOut = Float.NegativeInfinity;
            for (int i = start; i < end; i++)
                maxOut = Math.Max(maxOut, src[i]);

            // Compute exp and sum.
            Float sum = 0;
            for (int i = start; i < end; i++)
            {
                dst[i] = ExpFast(src[i] - maxOut);
                sum += dst[i];
            }

            // Normalize.
            for (int i = start; i < end; i++)
                dst[i] /= sum;
        }

        public static Float GetMedianInPlace(Float[] src, int count)
        {
            Contracts.Assert(count >= 0);
            Contracts.Assert(Utils.Size(src) >= count);

            if (count == 0)
                return Float.NaN;

            Array.Sort(src, 0, count);

            // Skip any NaNs. They sort to the low end.
            int ivMin = 0;
            int ivLim = count;
            while (ivMin < ivLim && Float.IsNaN(src[ivMin]))
                ivMin++;
            Contracts.Assert(ivMin <= ivLim);

            if (ivMin >= ivLim)
                return Float.NaN;

            // This assert will fire if Array.Sort changes to put NaNs at the high end.
            Contracts.Assert(!Float.IsNaN(src[ivLim - 1]));

            // If we're dealing with an odd number of things, just grab the middel item; otherwise,
            // average the two middle items.
            uint cv = (uint)ivMin + (uint)ivLim;
            int iv = (int)(cv / 2);
            if ((cv & 1) != 0)
                return src[iv];
            return (src[iv - 1] + src[iv]) / 2;
        }

        public static Double CosineSimilarity(Float[] a, Float[] b, int aIdx, int bIdx, int len)
        {
            const Double epsilon = 1e-12f;
            Contracts.Assert(len > 0);
            Contracts.Assert(aIdx >= 0 & aIdx <= a.Length - len);
            Contracts.Assert(bIdx >= 0 & bIdx <= b.Length - len);

            Double ab = 0;
            Double a2 = 0;
            Double b2 = 0;

            for (int lim = aIdx + len; aIdx < lim; aIdx++, bIdx++)
            {
                ab += (Double)a[aIdx] * b[bIdx];
                a2 += (Double)a[aIdx] * a[aIdx];
                b2 += (Double)b[bIdx] * b[bIdx];
            }

            Double similarity = ab / (Math.Sqrt(a2 * b2) + epsilon);
            Contracts.Assert(-1 - epsilon <= similarity & similarity <= 1 + epsilon);
            if (Math.Abs(similarity) > 1)
                return similarity > 1 ? 1 : -1;

            return similarity;
        }

        /// <summary>
        /// Entropy of a given probability
        /// </summary>
        public static Double Entropy(Double prob, bool useLnNotLog2 = false)
        {
            if (prob < 0 || prob > 1)
                return Double.NaN;
            if (prob == 0.0 || prob == 1.0)
                return 0.0;
            return
                useLnNotLog2
                ? -prob * Math.Log(prob) - (1 - prob) * Math.Log(1 - prob)
                : -prob * Math.Log(prob, 2) - (1 - prob) * Math.Log(1 - prob, 2);
        }

        /// <summary>
        /// Cross-entropy of two distributions
        /// </summary>
        public static Double CrossEntropy(Double probTrue, Double probPredicted, bool useLnNotLog2 = false)
        {
            if (probTrue < 0 || probTrue > 1 || probPredicted < 0 || probPredicted > 1)
                return Double.NaN;
            if ((probPredicted == 0.0 || probPredicted == 1.0) && (probPredicted == probTrue))
                return 0.0;
            return
                useLnNotLog2
                ? -probTrue * Math.Log(probPredicted) - (1 - probTrue) * Math.Log(1 - probPredicted)
                : -probTrue * Math.Log(probPredicted, 2) - (1 - probTrue) * Math.Log(1 - probPredicted, 2);
        }

        /// <summary>
        /// Given a set of values <c>Ln(a1), Ln(a2), ... Ln(an)</c>,
        /// return <c>Ln(a1+a2+...+an)</c>. This is especially useful
        /// when working with log probabilities and likelihoods.
        /// </summary>
        /// <param name="terms"></param>
        public static Float LnSum(IEnumerable<Float> terms)
        {
            // Two passes to find the overall max is a *lot* simpler,
            // but potentially more computationally intensive.
            Float max = Float.NegativeInfinity;
            Double soFar = 0;

            foreach (Float term in terms)
            {
                // At this point, all *prior* terms, Math.Exp(x - max).
                if (Float.IsNegativeInfinity(term))
                    continue;
                if (!(term > max))
                    soFar += Math.Exp(term - max);
                else
                {
                    soFar = Math.Exp(max - term) * soFar + 1;
                    max = term;
                }
            }
            return (Float)Math.Log(soFar) + max;
        }

        /// <summary>
        /// Math.Sin returns the input value for inputs with large magnitude. We return NaN instead, for consistency 
        /// with Math.Sin(infinity).
        /// </summary>
        public static double Sin(double a)
        {
            var res = Math.Sin(a);
            return Math.Abs(res) > 1 ? double.NaN : res;
        }

        /// <summary>
        /// Math.Cos returns the input value for inputs with large magnitude. We return NaN instead, for consistency 
        /// with Math.Cos(infinity).
        /// </summary>
        public static double Cos(double a)
        {
            var res = Math.Cos(a);
            return Math.Abs(res) > 1 ? double.NaN : res;
        }

        /// <summary>
        /// Returns the smallest integral value that is greater than or equal to the result of the division.
        /// </summary>
        /// <param name="numerator">Number to be divided.</param>
        /// <param name="denomenator">Number with which to divide the numerator.</param>
        /// <returns></returns>
        public static long DivisionCeiling(long numerator, long denomenator)
        {
            return (numerator + denomenator - 1) / denomenator;
        }
    }
}
