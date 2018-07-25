// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Security;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    public static class IntUtils
    {
        /// <summary>
        /// Add src to the 128 bits contained in dst. Ignores overflow, that is, the addition is done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ref ulong dstHi, ref ulong dstLo, ulong src)
        {
            if ((dstLo += src) < src)
                dstHi++;
        }

        /// <summary>
        /// Add src to dst. Ignores overflow, that is, the addition is done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ref ulong dstHi, ref ulong dstLo, ulong srcHi, ulong srcLo)
        {
            if ((dstLo += srcLo) < srcLo)
                dstHi++;
            dstHi += srcHi;
        }

        /// <summary>
        /// Subtract src from the 128 bits contained in dst. Ignores overflow, that is, the subtraction is
        /// done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sub(ref ulong dstHi, ref ulong dstLo, ulong src)
        {
            if (dstLo < src)
                dstHi--;
            dstLo -= src;
        }

        /// <summary>
        /// Subtract src from dst. Ignores overflow, that is, the subtraction is done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sub(ref ulong dstHi, ref ulong dstLo, ulong srcHi, ulong srcLo)
        {
            dstHi -= srcHi;
            if (dstLo < srcLo)
                dstHi--;
            dstLo -= srcLo;
        }

        /// <summary>
        /// Return true if a is less than b.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool LessThan(ulong a1, ulong a0, ulong b1, ulong b0)
        {
            return a1 < b1 || a1 == b1 && a0 < b0;
        }

        /// <summary>
        /// Divide the 128 bit value in <paramref name="lo"/> and <paramref name="hi"/> by <paramref name="den"/>.
        /// returning the quotient and placing the remainder in <paramref name="rem"/>. Throws on overflow.
        /// Note that <paramref name="lo"/> comes before <paramref name="hi"/>.
        /// </summary>
#if !CORECLR
        [DllImport(Thunk.NativePath), SuppressUnmanagedCodeSecurity]
        private static extern ulong Div64(ulong lo, ulong hi, ulong den, out ulong rem);
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong Div64(ulong lo, ulong hi, ulong den, out ulong rem)
        {
            if (den == 0)
                throw new DivideByZeroException();
            if (hi >= den)
                throw new OverflowException();
            return Div64Core(lo, hi, den, out rem);
        }

        // REVIEW: on Linux, the hardware divide-by-zero exception is not translated into
        // a managed exception properly by CoreCLR so the process will crash. This is a temporary fix
        // until CoreCLR addresses this issue.
        [DllImport(Thunk.NativePath, CharSet = CharSet.Unicode, EntryPoint = "Div64"), SuppressUnmanagedCodeSecurity]
        private static extern ulong Div64Core(ulong lo, ulong hi, ulong den, out ulong rem);
#endif

        /// <summary>
        /// Multiple the two 64-bit values to produce 128 bit result.
        /// </summary>
        [DllImport(Thunk.NativePath), SuppressUnmanagedCodeSecurity]
        private static extern ulong Mul64(ulong a, ulong b, out ulong hi);

        /// <summary>
        /// Divide and round to nearest using unbiased rounding. Throws on overflow.
        /// Note that <paramref name="lo"/> comes before <paramref name="hi"/>.
        /// </summary>
        public static ulong DivRound(ulong lo, ulong hi, ulong den)
        {
            // Divide and get the remainder.
            ulong rem;
            ulong quo = Div64(lo, hi, den, out rem);
            Contracts.Assert(rem < den);

            // Perform unbiased rounding, ie, tie goes to the even value.
            if (rem > den - rem || (rem == den - rem && (quo & 1) == 1))
                quo = checked(quo + 1);
            return quo;
        }

        /// <summary>
        /// Divide and round to nearest using unbiased rounding. Throws on overflow.
        /// Note that <paramref name="numLo"/> comes before <paramref name="numHi"/>.
        /// </summary>
        public static ulong DivRound(ulong numLo, ulong numHi, ulong denLo, ulong denHi)
        {
            // If the denominator fits in 64 bits use the simple overload above.
            if (denHi == 0)
                return DivRound(numLo, numHi, denLo);

            // At this point, we're guaranteed that the quotient doesn't overflow (since denHi > 0),
            // but rounding might still overflow.

            // Our goal is to set quo to the correct value and modify numXx to contain the remainder,
            // then fall through to the rounding code at the end.
            ulong quo;
            if (LessThan(numHi, numLo, denHi, denLo))
            {
                // Since num < den, quotient is zero and num is already the remainder.
                quo = 0;
            }
            else if ((long)denHi < 0)
            {
                // The high bit of den is set, so den <= num < den * 2. Thus, quotient is one and the
                // remainder is num - den.
                Contracts.Assert((long)numHi < 0);
                Sub(ref numHi, ref numLo, denHi, denLo);
                Contracts.Assert(LessThan(numHi, numLo, denHi, denLo));
                quo = 1;
            }
            else
            {
                // Shift num and den so that denHi has its high bit set. This requires 3 ulongs for num.
                int cbitShiftLeft = CbitHighZero(denHi);
                int cbitShiftRight = 64 - cbitShiftLeft;
                Contracts.Assert(0 < cbitShiftLeft & cbitShiftLeft < 64);

                denHi = (denHi << cbitShiftLeft) | (denLo >> cbitShiftRight);
                denLo <<= cbitShiftLeft;
                // The shifted numerator is (numEx, numHi, numLo). Note that the high bit of numEx must be zero,
                // since cbitShiftRight > 0.
                ulong numEx = numHi >> cbitShiftRight;
                numHi = (numHi << cbitShiftLeft) | (numLo >> cbitShiftRight);
                numLo <<= cbitShiftLeft;
                Contracts.Assert((long)numEx >= 0);
                Contracts.Assert((long)denHi < 0);
                Contracts.Assert(numEx < denHi);

                // Get the trial quotient and remainder by dividing (numEx, numHi) by denHi, and storing the
                // remainder in numHi.
                quo = Div64(numHi, numEx, denHi, out numHi);

                // Note that the quotient could be slightly too big, but never too small. To see this:
                //
                // * Use notation [ABC] as short-hand for A*X*X + B*X + C, where X = 2^64 is the "base"
                //   and A, B, C are the "digits", so 0 <= A < X, 0 <= B < X, and 0 <= C < X.
                //
                // * Given: numerator [ABC] and denominator [DE].
                // * The shifting above ensures X/2 <= D.
                // * Given: Q and R with [AB] = D*Q + R, and 0 <= R <= D - 1.
                // * Then [ABC] = [DE]*Q - E*Q + R*X + C = [DE]*Q + [RC] - E*Q.
                // * Thus [RC] - E*Q is the signed remainder when using quotient Q. Note that it is not
                //   necessarily normalized to be between 0 and [DE], so Q is not necessarily the correct quotient.
                // * However [RC] - E*Q <= [RC] < [DE] (since R < D), so Q is definitely NOT too small.
                // * However [RC] - E*Q can clearly be negative, implying that Q might be too big.
                // * Note that decreasing Q by x increases the remainder by x*[DE]. To get the correct quotient,
                //   we need the remainder r to satisfy 0 <= r < [DE].
                // * Since D >= X/2 (by construction), [DE] >= X*X/2.
                // * Trivially, [RC] - E*Q > -X*X >= -2*[DE]. This demonstrates that Q may need
                //   to decrease by at most two for the remainder to become non-negative.
                //
                // We can actually produce a tighter bound. Let k = cbitShiftLeft. Then 1 <= k <= 63.
                // We know that the low k bits of E are zero and only the low k bits of A can be non-zero.
                // Then A < 2^k and E = e*2^k where 0 <= e < 2^(64-k). We need to demonstrate that
                //    (1)   [RC] - E*Q + [DE] >= 0,
                // since that is equivalent to Q being too large by at most one.
                // Suppose (1) is false. That is, suppose [RC] - E*Q + [DE] < 0. Then
                //          [DE] < E*Q - [RC] <= E*Q,
                // so
                //    (2)   D*X < (Q-1)*E.
                // Note Q = (A*X+B-R) / D <= ((2^k-1)*X + (X-1)) / (X/2) < 2^(k+1). So Q < 2^(k+1). Since Q
                // is an integer, this implies
                //    (3)   Q <= 2^(k+1) - 1.
                // Then
                //    2^127 = X*X/2
                //          <= D*X                     since D >= X/2
                //          < (Q-1)*E                  by (2)
                //          <= (2^(k+1)-2)*e*2^k       by (3)
                //          <= 2*(2^k-1)*(2^(64-k)-1)*2^k
                //          = 2^(65+k) - 2^(2k+1) - 2^65 + 2^(k+1)
                // The only value of k that has a hope to make this true is k=63 (recall that k <= 63),
                // in which case the right hand side is:
                //          = 2^128 - 2^127 - 2^65 + 2^64
                //          = 2^127 - 2^64
                // Which is a contradiction, so our supposition that [RC] - E*Q + [DE] < 0 is impossible,
                // implying that (1) holds, implying that Q is too big by at most one. Also note that Q is
                // too big iff [RC] - E*Q < 0 iff [RC] < E*Q.

                // Compute E*Q = denLo * quo, stored in (p1, p0).
                ulong p1;
                ulong p0 = Mul64(denLo, quo, out p1);

                // See whether [RC] < E*Q, which is true iff [RC] - E*Q < 0 iff Q is too big.
                if (LessThan(numHi, numLo, p1, p0))
                {
                    // Need to decrement quo and add the denominator into the remainder.
                    Contracts.Assert(quo > 1);
                    quo--;
                    Add(ref numHi, ref numLo, denHi, denLo);
                }
                // Subtract E*Q from the remainder.
                Sub(ref numHi, ref numLo, p1, p0);
            }

            // At this point, num is the remainder, so num < den.
            Contracts.Assert(LessThan(numHi, numLo, denHi, denLo));

            // Set den = den - num and then compare to num, to determine whether the remainder is closer
            // to zero or to the denominator. If there is a tie, round to the even value. Note that the increment
            // of quo might overflow, hence the "checked".
            Sub(ref denHi, ref denLo, numHi, numLo);
            if (LessThan(denHi, denLo, numHi, numLo) || (quo & 1) == 1 && denHi == numHi && denLo == numLo)
                quo = checked(quo + 1);
            return quo;
        }

        /// <summary>
        /// Return the number of zero bits on the high end.
        /// </summary>
        private static int CbitHighZero(ulong u)
        {
            if (u == 0)
                return 64;

            int cbit = 0;
            if ((u & 0xFFFFFFFF00000000) == 0)
            {
                cbit += 32;
                u <<= 32;
            }
            if ((u & 0xFFFF000000000000) == 0)
            {
                cbit += 16;
                u <<= 16;
            }
            if ((u & 0xFF00000000000000) == 0)
            {
                cbit += 8;
                u <<= 8;
            }
            if ((u & 0xF000000000000000) == 0)
            {
                cbit += 4;
                u <<= 4;
            }
            if ((u & 0xC000000000000000) == 0)
            {
                cbit += 2;
                u <<= 2;
            }
            if ((u & 0x8000000000000000) == 0)
                cbit += 1;
            return cbit;
        }

        /// <summary>
        /// Multiply <paramref name="a"/> and <paramref name="b"/> and divide by <paramref name="den"/>,
        /// returning the quotient and placing the remainder in <paramref name="rem"/>. Throws on overflow.
        /// </summary>
#if !CORECLR
        [DllImport(Thunk.NativePath), SuppressUnmanagedCodeSecurity]
        private static extern ulong MulDiv64Core(ulong a, ulong b, ulong den, out ulong rem);

        public static ulong MulDiv64(ulong a, ulong b, ulong den, out ulong rem)
        {
            return MulDiv64Core(a, b, den, out rem);
        }
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MulDiv64(ulong a, ulong b, ulong den, out ulong rem)
        {
            if (den == 0)
                throw new DivideByZeroException();
            ulong quo;
            if (!TryMulDiv64(a, b, den, out quo, out rem))
                throw new OverflowException();
            return quo;
        }
#endif

        /// <summary>
        /// Multiply <paramref name="a"/> and <paramref name="b"/> and divide by <paramref name="den"/>,
        /// placing the quotient in <paramref name="quo"/> and the remainder in <paramref name="rem"/>.
        /// Returns true on success. On overflow, places zero in the out parameters and returns false.
        /// </summary>
        [DllImport(Thunk.NativePath), SuppressUnmanagedCodeSecurity]
        private static extern bool TryMulDiv64Core(ulong a, ulong b, ulong den, out ulong quo, out ulong rem);

        public static bool TryMulDiv64(ulong a, ulong b, ulong den, out ulong quo, out ulong rem)
        {
            return TryMulDiv64Core(a, b, den, out quo, out rem);
        }
    }
}
