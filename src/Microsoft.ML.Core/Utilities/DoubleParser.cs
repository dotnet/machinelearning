// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#undef COMPARE_BCL

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    public class DoubleParser
    {
        private const ulong TopBit = 0x8000000000000000UL;
        private const ulong TopTwoBits = 0xC000000000000000UL;
        private const ulong TopThreeBits = 0xE000000000000000UL;
        private const char InfinitySymbol = '\u221E';

        // REVIEW: casting ulong to Double doesn't always do the right thing, for example
        // with 0x84595161401484A0UL. Hence the gymnastics several places in this code. Note that
        // long to Double does work. The work around is:
        // if ((long)uu >= 0)
        //     dbl = (Double)(long)uu;
        // else
        // {
        //     dbl = (Double)(long)((uu >> 1) | (uu & 1));
        //     dbl += dbl;
        // }
        // Note that proper rounding to Double uses the "round to even" rule when there is a "tie".
        // That is, when the source is exactly half way between the two closest destination values,
        // the value with zero lowest bit is preferred. Oring (uu & 1) above ensures that we don't
        // let a non-tie become a tie. That is, if the low bit (that we're dropping) is 1, we capture
        // the fact that there are non-zero bits beyond the critical rounding bit, so the rounding
        // code knows that we don't have a tie.

        // When COMPARE_BCL is defined, we assert that we get the same result as the BCL. Unfortunately,
        // the BCL currently gets values wrong. For example, it maps 1.48e-323 to 0x02 instead of 0x03.
#if COMPARE_BCL
        // Whether we've reported a failure to match Double.TryParse. Only report the first failure.
        private volatile static bool _failed;
#endif

        /// <summary>
        /// Result codes from Parse methods. Note that the order matters.
        /// </summary>
        public enum Result
        {
            /// <summary>
            /// No issues.
            /// </summary>
            Good = 0,

            /// <summary>
            /// Empty or only whitespace
            /// </summary>
            Empty = 1,

            /// <summary>
            /// Extra non-whitespace characters after successful parse
            /// </summary>
            Extra = 2,

            /// <summary>
            /// Parsing error
            /// </summary>
            Error = 3
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(ReadOnlySpan<char> span, out Single value)
        {
            var res = Parse(span, out value);
            Contracts.Assert(res != Result.Empty || value == 0);
            return res <= Result.Empty;
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(ReadOnlySpan<char> span, out Double value)
        {
            var res = Parse(span, out value);
            Contracts.Assert(res != Result.Empty || value == 0);
            return res <= Result.Empty;
        }

        public static Result Parse(ReadOnlySpan<char> span, out Single value)
        {
            int ich = 0;
            for (; ; ich++)
            {
                if (ich >= span.Length)
                {
                    value = 0;
                    return Result.Empty;
                }
                if (!char.IsWhiteSpace(span[ich]))
                    break;
            }

            // Handle the common case of a single digit or ?
            if (span.Length - ich == 1)
            {
                char ch = span[ich];
                if (ch >= '0' && ch <= '9')
                {
                    value = ch - '0';
                    return Result.Good;
                }
                if (ch == '?')
                {
                    value = Single.NaN;
                    return Result.Good;
                }
            }

            int ichEnd;
            if (!DoubleParser.TryParse(span.Slice(ich, span.Length - ich), out ichEnd, out value))
            {
                value = default(Single);
                return Result.Error;
            }

            // Make sure everything was consumed.
            while (ichEnd < span.Length)
            {
                if (!char.IsWhiteSpace(span[ichEnd]))
                    return Result.Extra;
                ichEnd++;
            }

            return Result.Good;
        }

        public static Result Parse(ReadOnlySpan<char> span, out Double value)
        {
            int ich = 0;
            for (; ; ich++)
            {
                if (ich >= span.Length)
                {
                    value = 0;
                    return Result.Empty;
                }
                if (!char.IsWhiteSpace(span[ich]))
                    break;
            }

            // Handle the common case of a single digit or ?
            if (span.Length - ich == 1)
            {
                char ch = span[ich];
                if (ch >= '0' && ch <= '9')
                {
                    value = ch - '0';
                    return Result.Good;
                }
                if (ch == '?')
                {
                    value = Double.NaN;
                    return Result.Good;
                }
            }

            int ichEnd;
            if (!DoubleParser.TryParse(span.Slice(ich, span.Length - ich), out ichEnd, out value))
            {
                value = default(Double);
                return Result.Error;
            }

            // Make sure everything was consumed.
            while (ichEnd < span.Length)
            {
                if (!char.IsWhiteSpace(span[ichEnd]))
                    return Result.Extra;
                ichEnd++;
            }

            return Result.Good;
        }

        public static bool TryParse(ReadOnlySpan<char> span, out int ichEnd, out Single value)
        {
            bool neg = false;
            ulong num = 0;
            long exp = 0;

            ichEnd = 0;
            if (!TryParseCore(span, ref ichEnd, ref neg, ref num, ref exp))
                return TryParseSpecial(span, ref ichEnd, out value);

            if (num == 0)
            {
                value = 0;
                goto LDone;
            }

            // The Single version simply looks up the power of 10 in a table (as a Double), casts num
            // to Double, multiples, then casts to Single.
            Double res;
            if (exp >= 0)
            {
                if (exp == 0)
                    res = 1;
                else if (exp > _mpe10Dbl.Length)
                {
                    value = Single.PositiveInfinity;
                    goto LDone;
                }
                else
                    res = _mpe10Dbl[(int)exp - 1];
            }
            else
            {
                if (-exp > _mpne10Dbl.Length)
                {
                    value = 0;
                    goto LDone;
                }
                res = _mpne10Dbl[-(int)exp - 1];
            }

            // REVIEW: casting ulong to Double doesn't always get the answer correct.
            // Casting a non-negative long does work, though, so we jump through hoops to make
            // sure the top bit is clear and cast to long before casting to Double.
            Double tmp;
            if ((long)num >= 0)
                tmp = (Double)(long)num;
            else
            {
                tmp = (Double)(long)((num >> 1) | (num & 1));
                tmp += tmp;
            }

            res *= tmp;
            value = (Single)res;

        LDone:
            if (neg)
                value = -value;

#if COMPARE_BCL
            if (!_failed)
            {
                string str = span.ToString();
                Single x;
                if (!Single.TryParse(str, out x))
                {
                    // Single.TryParse doesn't gracefully handle overflow to infinity.
                    if (!Single.IsPositiveInfinity(value) && !Single.IsNegativeInfinity(value))
                    {
                        _failed = true;
                        Contracts.Assert(false, string.Format("Single.TryParse failed on: {0}", str));
                    }
                }
                else if (FloatUtils.GetBits(x) != FloatUtils.GetBits(value))
                {
                    // Double.TryParse gets negative zero wrong!
                    if (FloatUtils.GetBits(x) != 0 || FloatUtils.GetBits(value) != 0x80000000U || !neg)
                    {
                        _failed = true;
                        Contracts.Assert(false, string.Format("FloatParser disagrees with Single.TryParse on: {0} ({1} vs {2})", str, FloatUtils.GetBits(x), FloatUtils.GetBits(value)));
                    }
                }
            }
#endif

            return true;
        }

        public static bool TryParse(ReadOnlySpan<char> span, out int ichEnd, out Double value)
        {
            bool neg = false;
            ulong num = 0;
            long exp = 0;

            ichEnd = 0;
            if (!TryParseCore(span, ref ichEnd, ref neg, ref num, ref exp))
                return TryParseSpecial(span, ref ichEnd, out value);

            if (num == 0)
            {
                value = 0;
                goto LDone;
            }

            ulong mul;
            int e2;
            if (exp >= 0)
            {
                if (exp == 0)
                {
                    // REVIEW: casting ulong to Double doesn't always get the answer correct.
                    // Casting a non-negative long does work, though, so we jump through hoops to make
                    // sure the top bit is clear and cast to long before casting to Double.
                    if ((long)num >= 0)
                        value = (Double)(long)num;
                    else
                    {
                        value = (Double)(long)((num >> 1) | (num & 1));
                        value += value;
                    }
                    goto LDone;
                }
                if (exp <= 22 && num < (1UL << 53))
                {
                    // Can just use Double division, since both 10^|exp| and num can be exactly represented in Double.
                    // REVIEW: there are potential other "easy" cases, like when num isn't less than 2^54, but
                    // it ends with enough zeros that it can be made so by shifting.
                    Contracts.Assert((long)(Double)(long)num == (long)num);
                    value = (Double)(long)num * _mpe10Dbl[exp - 1];
                    goto LDone;
                }
                if (exp > _mpe10Man.Length)
                {
                    value = Double.PositiveInfinity;
                    goto LDone;
                }
                int index = (int)exp - 1;
                mul = _mpe10Man[index];
                e2 = _mpe10e2[index];
            }
            else
            {
                if (-exp <= 22 && num < (1UL << 53))
                {
                    // Can just use Double division, since both 10^|exp| and num can be exactly represented in Double.
                    // REVIEW: there are potential other "easy" cases, like when num isn't less than 2^54, but
                    // it ends with enough zeros that it can be made so by shifting.
                    Contracts.Assert((long)(Double)(long)num == (long)num);
                    value = (Double)(long)num / _mpe10Dbl[-exp - 1];
                    goto LDone;
                }
                if (-exp > _mpne10Man.Length)
                {
                    value = 0;
                    goto LDone;
                }
                int index = -(int)exp - 1;
                mul = _mpne10Man[index];
                e2 = -_mpne10ne2[index];
            }

            // Normalize the mantissa and initialize the base-2 (biased) exponent.
            // Ensure that the high bit is set.
            if ((num & 0xFFFFFFFF00000000UL) == 0) { num <<= 32; e2 -= 32; }
            if ((num & 0xFFFF000000000000UL) == 0) { num <<= 16; e2 -= 16; }
            if ((num & 0xFF00000000000000UL) == 0) { num <<= 8; e2 -= 8; }
            if ((num & 0xF000000000000000UL) == 0) { num <<= 4; e2 -= 4; }
            if ((num & 0xC000000000000000UL) == 0) { num <<= 2; e2 -= 2; }

            // Don't force the top bit to be non-zero, since we'll just have to clear it later....
            // if ((num & 0x8000000000000000UL) == 0) { num <<= 1; e2 -= 1; }

            // The high bit of mul is 1, and at least one of the top two bits of num is non-zero.
            // Taking the high 64 bits of the 128 bit result should give us enough bits to get the
            // right answer most of the time. Note, that it's not guaranteed that we always get the
            // right answer. Guaranteeing that takes much more work.... See the paper by David Gay at
            // https://www.ampl.com/REFS/rounding.pdf.
            Contracts.Assert((num & TopTwoBits) != 0);
            Contracts.Assert((mul & TopBit) != 0);

            // Multiply mul into num, keeping the high ulong.
            // REVIEW: Can this be made faster? It would be nice to use the _umul128 intrinsic.
            ulong hi1 = mul >> 32;
            ulong lo1 = mul & 0xFFFFFFFFUL;
            ulong hi2 = num >> 32;
            ulong lo2 = num & 0xFFFFFFFFUL;
            num = hi1 * hi2;
            hi1 *= lo2;
            hi2 *= lo1;
            num += hi1 >> 32;
            num += hi2 >> 32;
            hi1 <<= 32;
            hi2 <<= 32;
            if ((hi1 += hi2) < hi2)
                num++;
            lo1 *= lo2;
            if ((lo1 += hi1) < hi1)
                num++;

            // Cast to long first since ulong => Double is broken.
            Contracts.Assert((num & TopThreeBits) != 0);
            if ((long)num >= 0)
            {
                // Capture a non-zero lo to avoid an artificial tie.
                if (lo1 != 0)
                    num |= 1;
                value = (Double)(long)num;
            }
            else
            {
                if ((lo1 | (num & 1)) != 0)
                    num |= 2;
                value = (Double)(long)(num >> 1);
                e2++;
            }

            // Adjust the base-2 exponent.
            e2 += 0x3FF;
            if (e2 >= 0x7FF)
            {
                Contracts.Assert(exp > 0);
                value = Double.PositiveInfinity;
                goto LDone;
            }
            if (e2 <= 0)
            {
                // Break the exponent adjustment into two operations.
                Contracts.Assert(exp < 0);
                mul = 1UL << 52;
                unsafe { value *= *(Double*)&mul; }
                e2 += 0x3FF - 1;
                Contracts.Assert(e2 > 0);
            }

            // Multiply by the exponent adjustment.
            Contracts.Assert(0 < e2 & e2 < 0x7FF);
            mul = (ulong)e2 << 52;
            unsafe { value *= *(Double*)&mul; }

        LDone:
            if (neg)
                value = -value;

#if COMPARE_BCL
            string str = span.ToString();
            Double x;
            if (!Double.TryParse(str, out x))
            {
                // Double.TryParse doesn't gracefully handle overflow to infinity.
                if (!Double.IsPositiveInfinity(value) && !Double.IsNegativeInfinity(value))
                {
                    if (!_failed)
                    {
                        _failed = true;
                        Contracts.Assert(false, string.Format("Double.TryParse failed on: {0}", str));
                    }
                }
            }
            else if (FloatUtils.GetBits(x) != FloatUtils.GetBits(value))
            {
                // Double.TryParse gets negative zero wrong!
                if (FloatUtils.GetBits(x) != 0 || FloatUtils.GetBits(value) != TopBit || !neg)
                {
                    System.Diagnostics.Debug.WriteLine("*** FloatParser disagrees with Double.TryParse on: {0} ({1} vs {2})", str, FloatUtils.GetBits(x), FloatUtils.GetBits(value));
                }
            }
#endif

            return true;
        }

        private static bool TryParseSpecial(ReadOnlySpan<char> span, ref int ich, out Double value)
        {
            Single tmp;
            bool res = TryParseSpecial(span, ref ich, out tmp);
            value = tmp;
            return res;
        }

        private static bool TryParseSpecial(ReadOnlySpan<char> span, ref int ich, out Single value)
        {
            if (ich < span.Length)
            {
                switch (span[ich])
                {
                case '?':
                    // We also interpret ? to mean NaN.
                    value = Single.NaN;
                    ich += 1;
                    return true;

                case 'N':
                    if (ich + 3 <= span.Length && span[ich + 1] == 'a' && span[ich + 2] == 'N')
                    {
                        value = Single.NaN;
                        ich += 3;
                        return true;
                    }
                    break;

                case 'I':
                    if (ich + 8 <= span.Length && span[ich + 1] == 'n' && span[ich + 2] == 'f' && span[ich + 3] == 'i' && span[ich + 4] == 'n' && span[ich + 5] == 'i' && span[ich + 6] == 't' && span[ich + 7] == 'y')
                    {
                        value = Single.PositiveInfinity;
                        ich += 8;
                        return true;
                    }
                    break;

                case '-':
                    if (ich + 2 <= span.Length && span[ich + 1] == InfinitySymbol)
                    {
                        value = Single.NegativeInfinity;
                        ich += 2;
                        return true;
                    }

                    if (ich + 9 <= span.Length && span[ich + 1] == 'I' && span[ich + 2] == 'n' && span[ich + 3] == 'f' && span[ich + 4] == 'i' && span[ich + 5] == 'n' && span[ich + 6] == 'i' && span[ich + 7] == 't' && span[ich + 8] == 'y')
                    {
                        value = Single.NegativeInfinity;
                        ich += 9;
                        return true;
                    }
                    break;

                case InfinitySymbol:
                    value = Single.PositiveInfinity;
                    ich += 1;
                    return true;
                }
            }

            value = default(Single);
            return false;
        }

        private static bool TryParseCore(ReadOnlySpan<char> span, ref int ich, ref bool neg, ref ulong num, ref long exp)
        {
            Contracts.Assert(0 <= ich & ich <= span.Length);
            Contracts.Assert(!neg);
            Contracts.Assert(num == 0);
            Contracts.Assert(exp == 0);

            if (ich >= span.Length)
                return false;

            // If num gets bigger than this, we don't process additional digits.
            // REVIEW: Should we ensure that round off is always precise?
            const ulong numMax = (ulong.MaxValue - 9) / 10;

            bool digits = false;

            // Get started: handle sign
            int i = ich;
            switch (span[i])
            {
            default:
                return false;

            case '-':
                if (++i >= span.Length)
                    return false;
                neg = true;
                break;

            case '+':
                if (++i >= span.Length)
                    return false;
                break;

            case '.':
                goto LPoint;

            // The common cases.
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                break;
            }

            // Get digits before '.'
            uint d;
            for (; ; )
            {
                Contracts.Assert(i < span.Length);
                if ((d = (uint)span[i] - '0') > 9)
                    break;

                digits = true;
                if (num < numMax)
                    num = 10 * num + d;
                else
                    exp++;

                if (++i >= span.Length)
                {
                    ich = i;
                    return true;
                }
            }
            Contracts.Assert(i < span.Length);

            if (span[i] != '.')
                goto LAfterDigits;

        LPoint:
            Contracts.Assert(i < span.Length);
            Contracts.Assert(span[i] == '.');

            // Get the digits after '.'
            for (; ; )
            {
                if (++i >= span.Length)
                {
                    if (digits)
                        ich = i;
                    return digits;
                }

                Contracts.Assert(i < span.Length);
                if ((d = (uint)span[i] - '0') > 9)
                    break;

                digits = true;
                if (num < numMax)
                {
                    num = 10 * num + d;
                    exp--;
                }
            }

        LAfterDigits:
            Contracts.Assert(i < span.Length);
            if (!digits)
                return false;

            // Remember the current position.
            ich = i;

            // Check for an exponent.
            switch (span[i])
            {
            default:
                return true;

            case 'e':
            case 'E':
                if (++i >= span.Length)
                    return true;
                break;
            }

            // Handle the exponent sign.
            bool expNeg = false;
            Contracts.Assert(i < span.Length);
            switch (span[i])
            {
            case '-':
                if (++i >= span.Length)
                    return true;
                expNeg = true;
                break;
            case '+':
                if (++i >= span.Length)
                    return true;
                break;
            }

            // If the exponent exceeds this, the result will be infinite or zero
            // (depending on sign), so we clip it (to avoid overflow). Since exp is currently
            // bounded by a number of characters (digits before/after the decimal), adding
            // it to something bounded by eMax will not overflow.
            const long eMax = long.MaxValue / 100;
            Contracts.Assert(Math.Abs(exp) < int.MaxValue);

            digits = false;
            long e = 0;
            for (; ; )
            {
                Contracts.Assert(i < span.Length);
                if ((d = (uint)span[i] - '0') > 9)
                    break;

                digits = true;
                if (e < eMax)
                    e = 10 * e + (int)d;
                if (++i >= span.Length)
                    break;
            }

            if (digits)
            {
                if (expNeg)
                    e = -e;
                exp += e;
                ich = i;
            }

            return true;
        }

        // Map from base-10 exponent to 64-bit mantissa.
        // The approximation for 10^n is _mpe10man[n-1] * 2^(_mpe10e2[n-1]-64).
        private static ulong[] _mpe10Man = new ulong[] {
            0xA000000000000000UL, 0xC800000000000000UL, 0xFA00000000000000UL, 0x9C40000000000000UL, 0xC350000000000000UL, /*005*/
            0xF424000000000000UL, 0x9896800000000000UL, 0xBEBC200000000000UL, 0xEE6B280000000000UL, 0x9502F90000000000UL, /*010*/
            0xBA43B74000000000UL, 0xE8D4A51000000000UL, 0x9184E72A00000000UL, 0xB5E620F480000000UL, 0xE35FA931A0000000UL, /*015*/
            0x8E1BC9BF04000000UL, 0xB1A2BC2EC5000000UL, 0xDE0B6B3A76400000UL, 0x8AC7230489E80000UL, 0xAD78EBC5AC620000UL, /*020*/
            0xD8D726B7177A8000UL, 0x878678326EAC9000UL, 0xA968163F0A57B400UL, 0xD3C21BCECCEDA100UL, 0x84595161401484A0UL, /*025*/
            0xA56FA5B99019A5C8UL, 0xCECB8F27F4200F3AUL, 0x813F3978F8940984UL, 0xA18F07D736B90BE5UL, 0xC9F2C9CD04674EDEUL, /*030*/
            0xFC6F7C4045812296UL, 0x9DC5ADA82B70B59DUL, 0xC5371912364CE305UL, 0xF684DF56C3E01BC6UL, 0x9A130B963A6C115CUL, /*035*/
            0xC097CE7BC90715B3UL, 0xF0BDC21ABB48DB20UL, 0x96769950B50D88F4UL, 0xBC143FA4E250EB31UL, 0xEB194F8E1AE525FDUL, /*040*/
            0x92EFD1B8D0CF37BEUL, 0xB7ABC627050305ADUL, 0xE596B7B0C643C719UL, 0x8F7E32CE7BEA5C6FUL, 0xB35DBF821AE4F38BUL, /*045*/
            0xE0352F62A19E306EUL, 0x8C213D9DA502DE45UL, 0xAF298D050E4395D6UL, 0xDAF3F04651D47B4CUL, 0x88D8762BF324CD0FUL, /*050*/
            0xAB0E93B6EFEE0053UL, 0xD5D238A4ABE98068UL, 0x85A36366EB71F041UL, 0xA70C3C40A64E6C51UL, 0xD0CF4B50CFE20765UL, /*055*/
            0x82818F1281ED449FUL, 0xA321F2D7226895C7UL, 0xCBEA6F8CEB02BB39UL, 0xFEE50B7025C36A08UL, 0x9F4F2726179A2245UL, /*060*/
            0xC722F0EF9D80AAD6UL, 0xF8EBAD2B84E0D58BUL, 0x9B934C3B330C8577UL, 0xC2781F49FFCFA6D5UL, 0xF316271C7FC3908AUL, /*065*/
            0x97EDD871CFDA3A56UL, 0xBDE94E8E43D0C8ECUL, 0xED63A231D4C4FB27UL, 0x945E455F24FB1CF8UL, 0xB975D6B6EE39E436UL, /*070*/
            0xE7D34C64A9C85D44UL, 0x90E40FBEEA1D3A4AUL, 0xB51D13AEA4A488DDUL, 0xE264589A4DCDAB14UL, 0x8D7EB76070A08AECUL, /*075*/
            0xB0DE65388CC8ADA8UL, 0xDD15FE86AFFAD912UL, 0x8A2DBF142DFCC7ABUL, 0xACB92ED9397BF996UL, 0xD7E77A8F87DAF7FBUL, /*080*/
            0x86F0AC99B4E8DAFDUL, 0xA8ACD7C0222311BCUL, 0xD2D80DB02AABD62BUL, 0x83C7088E1AAB65DBUL, 0xA4B8CAB1A1563F52UL, /*085*/
            0xCDE6FD5E09ABCF26UL, 0x80B05E5AC60B6178UL, 0xA0DC75F1778E39D6UL, 0xC913936DD571C84CUL, 0xFB5878494ACE3A5FUL, /*090*/
            0x9D174B2DCEC0E47BUL, 0xC45D1DF942711D9AUL, 0xF5746577930D6500UL, 0x9968BF6ABBE85F20UL, 0xBFC2EF456AE276E8UL, /*095*/
            0xEFB3AB16C59B14A2UL, 0x95D04AEE3B80ECE5UL, 0xBB445DA9CA61281FUL, 0xEA1575143CF97226UL, 0x924D692CA61BE758UL, /*100*/
            0xB6E0C377CFA2E12EUL, 0xE498F455C38B997AUL, 0x8EDF98B59A373FECUL, 0xB2977EE300C50FE7UL, 0xDF3D5E9BC0F653E1UL, /*105*/
            0x8B865B215899F46CUL, 0xAE67F1E9AEC07187UL, 0xDA01EE641A708DE9UL, 0x884134FE908658B2UL, 0xAA51823E34A7EEDEUL, /*110*/
            0xD4E5E2CDC1D1EA96UL, 0x850FADC09923329EUL, 0xA6539930BF6BFF45UL, 0xCFE87F7CEF46FF16UL, 0x81F14FAE158C5F6EUL, /*115*/
            0xA26DA3999AEF7749UL, 0xCB090C8001AB551CUL, 0xFDCB4FA002162A63UL, 0x9E9F11C4014DDA7EUL, 0xC646D63501A1511DUL, /*120*/
            0xF7D88BC24209A565UL, 0x9AE757596946075FUL, 0xC1A12D2FC3978937UL, 0xF209787BB47D6B84UL, 0x9745EB4D50CE6332UL, /*125*/
            0xBD176620A501FBFFUL, 0xEC5D3FA8CE427AFFUL, 0x93BA47C980E98CDFUL, 0xB8A8D9BBE123F017UL, 0xE6D3102AD96CEC1DUL, /*130*/
            0x9043EA1AC7E41392UL, 0xB454E4A179DD1877UL, 0xE16A1DC9D8545E94UL, 0x8CE2529E2734BB1DUL, 0xB01AE745B101E9E4UL, /*135*/
            0xDC21A1171D42645DUL, 0x899504AE72497EBAUL, 0xABFA45DA0EDBDE69UL, 0xD6F8D7509292D603UL, 0x865B86925B9BC5C2UL, /*140*/
            0xA7F26836F282B732UL, 0xD1EF0244AF2364FFUL, 0x8335616AED761F1FUL, 0xA402B9C5A8D3A6E7UL, 0xCD036837130890A1UL, /*145*/
            0x802221226BE55A64UL, 0xA02AA96B06DEB0FDUL, 0xC83553C5C8965D3DUL, 0xFA42A8B73ABBF48CUL, 0x9C69A97284B578D7UL, /*150*/
            0xC38413CF25E2D70DUL, 0xF46518C2EF5B8CD1UL, 0x98BF2F79D5993802UL, 0xBEEEFB584AFF8603UL, 0xEEAABA2E5DBF6784UL, /*155*/
            0x952AB45CFA97A0B2UL, 0xBA756174393D88DFUL, 0xE912B9D1478CEB17UL, 0x91ABB422CCB812EEUL, 0xB616A12B7FE617AAUL, /*160*/
            0xE39C49765FDF9D94UL, 0x8E41ADE9FBEBC27DUL, 0xB1D219647AE6B31CUL, 0xDE469FBD99A05FE3UL, 0x8AEC23D680043BEEUL, /*165*/
            0xADA72CCC20054AE9UL, 0xD910F7FF28069DA4UL, 0x87AA9AFF79042286UL, 0xA99541BF57452B28UL, 0xD3FA922F2D1675F2UL, /*170*/
            0x847C9B5D7C2E09B7UL, 0xA59BC234DB398C25UL, 0xCF02B2C21207EF2EUL, 0x8161AFB94B44F57DUL, 0xA1BA1BA79E1632DCUL, /*175*/
            0xCA28A291859BBF93UL, 0xFCB2CB35E702AF78UL, 0x9DEFBF01B061ADABUL, 0xC56BAEC21C7A1916UL, 0xF6C69A72A3989F5BUL, /*180*/
            0x9A3C2087A63F6399UL, 0xC0CB28A98FCF3C7FUL, 0xF0FDF2D3F3C30B9FUL, 0x969EB7C47859E743UL, 0xBC4665B596706114UL, /*185*/
            0xEB57FF22FC0C7959UL, 0x9316FF75DD87CBD8UL, 0xB7DCBF5354E9BECEUL, 0xE5D3EF282A242E81UL, 0x8FA475791A569D10UL, /*190*/
            0xB38D92D760EC4455UL, 0xE070F78D3927556AUL, 0x8C469AB843B89562UL, 0xAF58416654A6BABBUL, 0xDB2E51BFE9D0696AUL, /*195*/
            0x88FCF317F22241E2UL, 0xAB3C2FDDEEAAD25AUL, 0xD60B3BD56A5586F1UL, 0x85C7056562757456UL, 0xA738C6BEBB12D16CUL, /*200*/
            0xD106F86E69D785C7UL, 0x82A45B450226B39CUL, 0xA34D721642B06084UL, 0xCC20CE9BD35C78A5UL, 0xFF290242C83396CEUL, /*205*/
            0x9F79A169BD203E41UL, 0xC75809C42C684DD1UL, 0xF92E0C3537826145UL, 0x9BBCC7A142B17CCBUL, 0xC2ABF989935DDBFEUL, /*210*/
            0xF356F7EBF83552FEUL, 0x98165AF37B2153DEUL, 0xBE1BF1B059E9A8D6UL, 0xEDA2EE1C7064130CUL, 0x9485D4D1C63E8BE7UL, /*215*/
            0xB9A74A0637CE2EE1UL, 0xE8111C87C5C1BA99UL, 0x910AB1D4DB9914A0UL, 0xB54D5E4A127F59C8UL, 0xE2A0B5DC971F303AUL, /*220*/
            0x8DA471A9DE737E24UL, 0xB10D8E1456105DADUL, 0xDD50F1996B947518UL, 0x8A5296FFE33CC92FUL, 0xACE73CBFDC0BFB7BUL, /*225*/
            0xD8210BEFD30EFA5AUL, 0x8714A775E3E95C78UL, 0xA8D9D1535CE3B396UL, 0xD31045A8341CA07CUL, 0x83EA2B892091E44DUL, /*230*/
            0xA4E4B66B68B65D60UL, 0xCE1DE40642E3F4B9UL, 0x80D2AE83E9CE78F3UL, 0xA1075A24E4421730UL, 0xC94930AE1D529CFCUL, /*235*/
            0xFB9B7CD9A4A7443CUL, 0x9D412E0806E88AA5UL, 0xC491798A08A2AD4EUL, 0xF5B5D7EC8ACB58A2UL, 0x9991A6F3D6BF1765UL, /*240*/
            0xBFF610B0CC6EDD3FUL, 0xEFF394DCFF8A948EUL, 0x95F83D0A1FB69CD9UL, 0xBB764C4CA7A4440FUL, 0xEA53DF5FD18D5513UL, /*245*/
            0x92746B9BE2F8552CUL, 0xB7118682DBB66A77UL, 0xE4D5E82392A40515UL, 0x8F05B1163BA6832DUL, 0xB2C71D5BCA9023F8UL, /*250*/
            0xDF78E4B2BD342CF6UL, 0x8BAB8EEFB6409C1AUL, 0xAE9672ABA3D0C320UL, 0xDA3C0F568CC4F3E8UL, 0x8865899617FB1871UL, /*255*/
            0xAA7EEBFB9DF9DE8DUL, 0xD51EA6FA85785631UL, 0x8533285C936B35DEUL, 0xA67FF273B8460356UL, 0xD01FEF10A657842CUL, /*260*/
            0x8213F56A67F6B29BUL, 0xA298F2C501F45F42UL, 0xCB3F2F7642717713UL, 0xFE0EFB53D30DD4D7UL, 0x9EC95D1463E8A506UL, /*265*/
            0xC67BB4597CE2CE48UL, 0xF81AA16FDC1B81DAUL, 0x9B10A4E5E9913128UL, 0xC1D4CE1F63F57D72UL, 0xF24A01A73CF2DCCFUL, /*270*/
            0x976E41088617CA01UL, 0xBD49D14AA79DBC82UL, 0xEC9C459D51852BA2UL, 0x93E1AB8252F33B45UL, 0xB8DA1662E7B00A17UL, /*275*/
            0xE7109BFBA19C0C9DUL, 0x906A617D450187E2UL, 0xB484F9DC9641E9DAUL, 0xE1A63853BBD26451UL, 0x8D07E33455637EB2UL, /*280*/
            0xB049DC016ABC5E5FUL, 0xDC5C5301C56B75F7UL, 0x89B9B3E11B6329BAUL, 0xAC2820D9623BF429UL, 0xD732290FBACAF133UL, /*285*/
            0x867F59A9D4BED6C0UL, 0xA81F301449EE8C70UL, 0xD226FC195C6A2F8CUL, 0x83585D8FD9C25DB7UL, 0xA42E74F3D032F525UL, /*290*/
            0xCD3A1230C43FB26FUL, 0x80444B5E7AA7CF85UL, 0xA0555E361951C366UL, 0xC86AB5C39FA63440UL, 0xFA856334878FC150UL, /*295*/
            0x9C935E00D4B9D8D2UL, 0xC3B8358109E84F07UL, 0xF4A642E14C6262C8UL, 0x98E7E9CCCFBD7DBDUL, 0xBF21E44003ACDD2CUL, /*300*/
            0xEEEA5D5004981478UL, 0x95527A5202DF0CCBUL, 0xBAA718E68396CFFDUL, 0xE950DF20247C83FDUL, 0x91D28B7416CDD27EUL, /*305*/
            0xB6472E511C81471DUL, 0xE3D8F9E563A198E5UL, 0x8E679C2F5E44FF8FUL, 0xB201833B35D63F73UL, 0xDE81E40A034BCF4FUL, /*310*/
            0x8B112E86420F6191UL, 0xADD57A27D29339F6UL, 0xD94AD8B1C7380874UL, 0x87CEC76F1C830548UL, 0xA9C2794AE3A3C69AUL, /*315*/
            0xD433179D9C8CB841UL, 0x849FEEC281D7F328UL, 0xA5C7EA73224DEFF3UL, 0xCF39E50FEAE16BEFUL, 0x81842F29F2CCE375UL, /*320*/
        };

        // Map from negative base-10 exponent to 64-bit mantissa. Note that the top bit of these is set.
        // The approximation for 10^-n is _mpne10man[n-1] * 2^(-_mpne10ne2[n-1]-64).
        private static ulong[] _mpne10Man = new ulong[] {
            0xCCCCCCCCCCCCCCCDUL, 0xA3D70A3D70A3D70AUL, 0x83126E978D4FDF3BUL, 0xD1B71758E219652CUL, 0xA7C5AC471B478423UL, /*005*/
            0x8637BD05AF6C69B6UL, 0xD6BF94D5E57A42BCUL, 0xABCC77118461CEFDUL, 0x89705F4136B4A597UL, 0xDBE6FECEBDEDD5BFUL, /*010*/
            0xAFEBFF0BCB24AAFFUL, 0x8CBCCC096F5088CCUL, 0xE12E13424BB40E13UL, 0xB424DC35095CD80FUL, 0x901D7CF73AB0ACD9UL, /*015*/
            0xE69594BEC44DE15BUL, 0xB877AA3236A4B449UL, 0x9392EE8E921D5D07UL, 0xEC1E4A7DB69561A5UL, 0xBCE5086492111AEBUL, /*020*/
            0x971DA05074DA7BEFUL, 0xF1C90080BAF72CB1UL, 0xC16D9A0095928A27UL, 0x9ABE14CD44753B53UL, 0xF79687AED3EEC551UL, /*025*/
            0xC612062576589DDBUL, 0x9E74D1B791E07E48UL, 0xFD87B5F28300CA0EUL, 0xCAD2F7F5359A3B3EUL, 0xA2425FF75E14FC32UL, /*030*/
            0x81CEB32C4B43FCF5UL, 0xCFB11EAD453994BAUL, 0xA6274BBDD0FADD62UL, 0x84EC3C97DA624AB5UL, 0xD4AD2DBFC3D07788UL, /*035*/
            0xAA242499697392D3UL, 0x881CEA14545C7575UL, 0xD9C7DCED53C72256UL, 0xAE397D8AA96C1B78UL, 0x8B61313BBABCE2C6UL, /*040*/
            0xDF01E85F912E37A3UL, 0xB267ED1940F1C61CUL, 0x8EB98A7A9A5B04E3UL, 0xE45C10C42A2B3B06UL, 0xB6B00D69BB55C8D1UL, /*045*/
            0x9226712162AB070EUL, 0xE9D71B689DDE71B0UL, 0xBB127C53B17EC159UL, 0x95A8637627989AAEUL, 0xEF73D256A5C0F77DUL, /*050*/
            0xBF8FDB78849A5F97UL, 0x993FE2C6D07B7FACUL, 0xF53304714D9265E0UL, 0xC428D05AA4751E4DUL, 0x9CED737BB6C4183DUL, /*055*/
            0xFB158592BE068D2FUL, 0xC8DE047564D20A8CUL, 0xA0B19D2AB70E6ED6UL, 0x808E17555F3EBF12UL, 0xCDB02555653131B6UL, /*060*/
            0xA48CEAAAB75A8E2BUL, 0x83A3EEEEF9153E89UL, 0xD29FE4B18E88640FUL, 0xA87FEA27A539E9A5UL, 0x86CCBB52EA94BAEBUL, /*065*/
            0xD7ADF884AA879177UL, 0xAC8B2D36EED2DAC6UL, 0x8A08F0F8BF0F156BUL, 0xDCDB1B2798182245UL, 0xB0AF48EC79ACE837UL, /*070*/
            0x8D590723948A535FUL, 0xE2280B6C20DD5232UL, 0xB4ECD5F01A4AA828UL, 0x90BD77F3483BB9BAUL, 0xE7958CB87392C2C3UL, /*075*/
            0xB94470938FA89BCFUL, 0x9436C0760C86E30CUL, 0xED246723473E3813UL, 0xBDB6B8E905CB600FUL, 0x97C560BA6B0919A6UL, /*080*/
            0xF2D56790AB41C2A3UL, 0xC24452DA229B021CUL, 0x9B69DBE1B548CE7DUL, 0xF8A95FCF88747D94UL, 0xC6EDE63FA05D3144UL, /*085*/
            0x9F24B832E6B0F436UL, 0xFEA126B7D78186BDUL, 0xCBB41EF979346BCAUL, 0xA2F67F2DFA90563BUL, 0x825ECC24C8737830UL, /*090*/
            0xD097AD07A71F26B2UL, 0xA6DFBD9FB8E5B88FUL, 0x857FCAE62D8493A5UL, 0xD59944A37C0752A2UL, 0xAAE103B5FCD2A882UL, /*095*/
            0x88B402F7FD75539BUL, 0xDAB99E59958885C5UL, 0xAEFAE51477A06B04UL, 0x8BFBEA76C619EF36UL, 0xDFF9772470297EBDUL, /*100*/
            0xB32DF8E9F3546564UL, 0x8F57FA54C2A9EAB7UL, 0xE55990879DDCAABEUL, 0xB77ADA0617E3BBCBUL, 0x92C8AE6B464FC96FUL, /*105*/
            0xEADAB0ABA3B2DBE5UL, 0xBBE226EFB628AFEBUL, 0x964E858C91BA2655UL, 0xF07DA27A82C37088UL, 0xC06481FB9BCF8D3AUL, /*110*/
            0x99EA0196163FA42EUL, 0xF64335BCF065D37DUL, 0xC5029163F384A931UL, 0x9D9BA7832936EDC1UL, 0xFC2C3F3841F17C68UL, /*115*/
            0xC9BCFF6034C13053UL, 0xA163FF802A3426A9UL, 0x811CCC668829B887UL, 0xCE947A3DA6A9273EUL, 0xA54394FE1EEDB8FFUL, /*120*/
            0x843610CB4BF160CCUL, 0xD389B47879823479UL, 0xA93AF6C6C79B5D2EUL, 0x87625F056C7C4A8BUL, 0xD89D64D57A607745UL, /*125*/
            0xAD4AB7112EB3929EUL, 0x8AA22C0DBEF60EE4UL, 0xDDD0467C64BCE4A1UL, 0xB1736B96B6FD83B4UL, 0x8DF5EFABC5979C90UL, /*130*/
            0xE3231912D5BF60E6UL, 0xB5B5ADA8AAFF80B8UL, 0x915E2486EF32CD60UL, 0xE896A0D7E51E1566UL, 0xBA121A4650E4DDECUL, /*135*/
            0x94DB483840B717F0UL, 0xEE2BA6C0678B597FUL, 0xBE89523386091466UL, 0x986DDB5C6B3A76B8UL, 0xF3E2F893DEC3F126UL, /*140*/
            0xC31BFA0FE5698DB8UL, 0x9C1661A651213E2DUL, 0xF9BD690A1B68637BUL, 0xC7CABA6E7C5382C9UL, 0x9FD561F1FD0F9BD4UL, /*145*/
            0xFFBBCFE994E5C620UL, 0xCC963FEE10B7D1B3UL, 0xA3AB66580D5FDAF6UL, 0x82EF85133DE648C5UL, 0xD17F3B51FCA3A7A1UL, /*150*/
            0xA798FC4196E952E7UL, 0x8613FD0145877586UL, 0xD686619BA27255A3UL, 0xAB9EB47C81F5114FUL, 0x894BC396CE5DA772UL, /*155*/
            0xDBAC6C247D62A584UL, 0xAFBD2350644EEAD0UL, 0x8C974F7383725573UL, 0xE0F218B8D25088B8UL, 0xB3F4E093DB73A093UL, /*160*/
            0x8FF71A0FE2C2E6DCUL, 0xE65829B3046B0AFAUL, 0xB84687C269EF3BFBUL, 0x936B9FCEBB25C996UL, 0xEBDF661791D60F56UL, /*165*/
            0xBCB2B812DB11A5DEUL, 0x96F5600F15A7B7E5UL, 0xF18899B1BC3F8CA2UL, 0xC13A148E3032D6E8UL, 0x9A94DD3E8CF578BAUL, /*170*/
            0xF7549530E188C129UL, 0xC5DD44271AD3CDBAUL, 0x9E4A9CEC15763E2FUL, 0xFD442E4688BD304BUL, 0xCA9CF1D206FDC03CUL, /*175*/
            0xA21727DB38CB0030UL, 0x81AC1FE293D599C0UL, 0xCF79CC9DB955C2CCUL, 0xA5FB0A17C777CF0AUL, 0x84C8D4DFD2C63F3BUL, /*180*/
            0xD47487CC8470652BUL, 0xA9F6D30A038D1DBCUL, 0x87F8A8D4CFA417CAUL, 0xD98DDAEE19068C76UL, 0xAE0B158B4738705FUL, /*185*/
            0x8B3C113C38F9F37FUL, 0xDEC681F9F4C31F31UL, 0xB23867FB2A35B28EUL, 0x8E938662882AF53EUL, 0xE41F3D6A7377EECAUL, /*190*/
            0xB67F6455292CBF08UL, 0x91FF83775423CC06UL, 0xE998D258869FACD7UL, 0xBAE0A846D2195713UL, 0x9580869F0E7AAC0FUL, /*195*/
            0xEF340A98172AACE5UL, 0xBF5CD54678EEF0B7UL, 0x991711052D8BF3C5UL, 0xF4F1B4D515ACB93CUL, 0xC3F490AA77BD60FDUL, /*200*/
            0x9CC3A6EEC6311A64UL, 0xFAD2A4B13D1B5D6CUL, 0xC8A883C0FDAF7DF0UL, 0xA086CFCD97BF97F4UL, 0x806BD9714632DFF6UL, /*205*/
            0xCD795BE870516656UL, 0xA46116538D0DEB78UL, 0x8380DEA93DA4BC60UL, 0xD267CAA862A12D67UL, 0xA8530886B54DBDECUL, /*210*/
            0x86A8D39EF77164BDUL, 0xD77485CB25823AC7UL, 0xAC5D37D5B79B6239UL, 0x89E42CAAF9491B61UL, 0xDCA04777F541C568UL, /*215*/
            0xB080392CC4349DEDUL, 0x8D3360F09CF6E4BDUL, 0xE1EBCE4DC7F16DFCUL, 0xB4BCA50B065ABE63UL, 0x9096EA6F3848984FUL, /*220*/
            0xE757DD7EC07426E5UL, 0xB913179899F68584UL, 0x940F4613AE5ED137UL, 0xECE53CEC4A314EBEUL, 0xBD8430BD08277231UL, /*225*/
            0x979CF3CA6CEC5B5BUL, 0xF294B943E17A2BC4UL, 0xC21094364DFB5637UL, 0x9B407691D7FC44F8UL, 0xF867241C8CC6D4C1UL, /*230*/
            0xC6B8E9B0709F109AUL, 0x9EFA548D26E5A6E2UL, 0xFE5D54150B090B03UL, 0xCB7DDCDDA26DA269UL, 0xA2CB1717B52481EDUL, /*235*/
            0x823C12795DB6CE57UL, 0xD0601D8EFC57B08CUL, 0xA6B34AD8C9DFC070UL, 0x855C3BE0A17FCD26UL, 0xD5605FCDCF32E1D7UL, /*240*/
            0xAAB37FD7D8F58179UL, 0x888F99797A5E012DUL, 0xDA7F5BF590966849UL, 0xAECC49914078536DUL, 0x8BD6A141006042BEUL, /*245*/
            0xDFBDCECE67006AC9UL, 0xB2FE3F0B8599EF08UL, 0x8F31CC0937AE58D3UL, 0xE51C79A85916F485UL, 0xB749FAED14125D37UL, /*250*/
            0x92A1958A7675175FUL, 0xEA9C227723EE8BCBUL, 0xBBB01B9283253CA3UL, 0x96267C7535B763B5UL, 0xF03D93EEBC589F88UL, /*255*/
            0xC0314325637A193AUL, 0x99C102844F94E0FBUL, 0xF6019DA07F549B2BUL, 0xC4CE17B399107C23UL, 0x9D71AC8FADA6C9B5UL, /*260*/
            0xFBE9141915D7A922UL, 0xC987434744AC874FUL, 0xA139029F6A239F72UL, 0x80FA687F881C7F8EUL, 0xCE5D73FF402D98E4UL, /*265*/
            0xA5178FFF668AE0B6UL, 0x8412D9991ED58092UL, 0xD3515C2831559A83UL, 0xA90DE3535AAAE202UL, 0x873E4F75E2224E68UL, /*270*/
            0xD863B256369D4A41UL, 0xAD1C8EAB5EE43B67UL, 0x8A7D3EEF7F1CFC52UL, 0xDD95317F31C7FA1DUL, 0xB1442798F49FFB4BUL, /*275*/
            0x8DD01FAD907FFC3CUL, 0xE2E69915B3FFF9F9UL, 0xB58547448FFFFB2EUL, 0x91376C36D99995BEUL, 0xE858AD248F5C22CAUL, /*280*/
            0xB9E08A83A5E34F08UL, 0x94B3A202EB1C3F39UL, 0xEDEC366B11C6CB8FUL, 0xBE5691EF416BD60CUL, 0x9845418C345644D7UL, /*285*/
            0xF3A20279ED56D48AUL, 0xC2E801FB244576D5UL, 0x9BECCE62836AC577UL, 0xF97AE3D0D2446F25UL, 0xC795830D75038C1EUL, /*290*/
            0x9FAACF3DF73609B1UL, 0xFF77B1FCBEBCDC4FUL, 0xCC5FC196FEFD7D0CUL, 0xA37FCE126597973DUL, 0x82CCA4DB847945CAUL, /*295*/
            0xD1476E2C07286FAAUL, 0xA76C582338ED2622UL, 0x85F0468293F0EB4EUL, 0xD64D3D9DB981787DUL, 0xAB70FE17C79AC6CAUL, /*300*/
            0x892731AC9FAF056FUL, 0xDB71E91432B1A24BUL, 0xAF8E5410288E1B6FUL, 0x8C71DCD9BA0B4926UL, 0xE0B62E2929ABA83CUL, /*305*/
            0xB3C4F1BA87BC8697UL, 0x8FD0C16206306BACUL, 0xE61ACF033D1A45DFUL, 0xB8157268FDAE9E4CUL, 0x93445B8731587EA3UL, /*310*/
            0xEBA09271E88D976CUL, 0xBC807527ED3E12BDUL, 0x96CD2A865764DBCAUL, 0xF148440A256E2C77UL, 0xC1069CD4EABE89F9UL, /*315*/
            0x9A6BB0AA55653B2DUL, 0xF712B443BBD52B7CUL, 0xC5A890362FDDBC63UL, 0x9E20735E8CB16382UL, 0xFD00B897478238D1UL, /*320*/
            0xCA66FA129F9B60A7UL, 0xA1EBFB4219491A1FUL, 0x818995CE7AA0E1B2UL, 0xCF42894A5DCE35EAUL, 0xA5CED43B7E3E9188UL, /*325*/
            0x84A57695FE98746DUL, 0xD43BF0EFFDC0BA48UL, 0xA9C98D8CCB009506UL, 0x87D4713D6F33AA6CUL, 0xD953E8624B85DD79UL, /*330*/
            0xADDCB9E83C6B1794UL, 0x8B16FB203055AC76UL, 0xDE8B2B66B3BC4724UL, 0xB208EF855C969F50UL, 0x8E6D8C6AB0787F73UL, /*335*/
            0xE3E27A444D8D98B8UL, 0xB64EC836A47146FAUL, 0x91D8A02BB6C10594UL, 0xE95A99DF8ACE6F54UL, 0xBAAEE17FA23EBF76UL, /*340*/
            0x9558B4661B6565F8UL, 0xEEF453D6923BD65AUL, 0xBF29DCABA82FDEAEUL, 0x98EE4A22ECF3188CUL, 0xF4B0769E47EB5A79UL, /*345*/
            0xC3C05EE50655E1FAUL, 0x9C99E58405118195UL, 0xFA8FD5A0081C0288UL, 0xC873114CD3499BA0UL, 0xA05C0DD70F6E161AUL, /*350*/
            0x8049A4AC0C5811AEUL, 0xCD42A11346F34F7DUL, 0xA4354DA9058F72CAUL, 0x835DD7BA6AD928A2UL, 0xD22FBF90AAF50DD0UL, /*355*/
            0xA82632DA225DA4A6UL, 0x8684F57B4EB15085UL, 0xD73B225EE44EE73BUL, 0xAC2F4EB2503F1F63UL, 0x89BF722840327F82UL, /*360*/
        };

        // Map from base-10 exponent to base-2 exponent.
        // The approximation for 10^n is _mpe10man[n-1] * 2^(_mpe10e2[n-1]-64).
        private static short[] _mpe10e2 = new short[] {
               4,    7,   10,   14,   17,   20,   24,   27,   30,   34,   37,   40,   44,   47,   50,   54,   57,   60,   64,   67, /*020*/
              70,   74,   77,   80,   84,   87,   90,   94,   97,  100,  103,  107,  110,  113,  117,  120,  123,  127,  130,  133, /*040*/
             137,  140,  143,  147,  150,  153,  157,  160,  163,  167,  170,  173,  177,  180,  183,  187,  190,  193,  196,  200, /*060*/
             203,  206,  210,  213,  216,  220,  223,  226,  230,  233,  236,  240,  243,  246,  250,  253,  256,  260,  263,  266, /*080*/
             270,  273,  276,  280,  283,  286,  290,  293,  296,  299,  303,  306,  309,  313,  316,  319,  323,  326,  329,  333, /*100*/
             336,  339,  343,  346,  349,  353,  356,  359,  363,  366,  369,  373,  376,  379,  383,  386,  389,  392,  396,  399, /*120*/
             402,  406,  409,  412,  416,  419,  422,  426,  429,  432,  436,  439,  442,  446,  449,  452,  456,  459,  462,  466, /*140*/
             469,  472,  476,  479,  482,  486,  489,  492,  495,  499,  502,  505,  509,  512,  515,  519,  522,  525,  529,  532, /*160*/
             535,  539,  542,  545,  549,  552,  555,  559,  562,  565,  569,  572,  575,  579,  582,  585,  588,  592,  595,  598, /*180*/
             602,  605,  608,  612,  615,  618,  622,  625,  628,  632,  635,  638,  642,  645,  648,  652,  655,  658,  662,  665, /*200*/
             668,  672,  675,  678,  681,  685,  688,  691,  695,  698,  701,  705,  708,  711,  715,  718,  721,  725,  728,  731, /*220*/
             735,  738,  741,  745,  748,  751,  755,  758,  761,  765,  768,  771,  775,  778,  781,  784,  788,  791,  794,  798, /*240*/
             801,  804,  808,  811,  814,  818,  821,  824,  828,  831,  834,  838,  841,  844,  848,  851,  854,  858,  861,  864, /*260*/
             868,  871,  874,  877,  881,  884,  887,  891,  894,  897,  901,  904,  907,  911,  914,  917,  921,  924,  927,  931, /*280*/
             934,  937,  941,  944,  947,  951,  954,  957,  961,  964,  967,  971,  974,  977,  980,  984,  987,  990,  994,  997, /*300*/
            1000, 1004, 1007, 1010, 1014, 1017, 1020, 1024, 1027, 1030, 1034, 1037, 1040, 1044, 1047, 1050, 1054, 1057, 1060, 1064, /*320*/
        };

        // Map from negative base-10 exponent to negative base-2 exponent.
        // The approximation for 10^-n is _mpne10man[n-1] * 2^(-_mpne10ne2[n-1]-64).
        private static readonly short[] _mpne10ne2 = new short[] {
               3,    6,    9,   13,   16,   19,   23,   26,   29,   33,   36,   39,   43,   46,   49,   53,   56,   59,   63,   66, /*020*/
              69,   73,   76,   79,   83,   86,   89,   93,   96,   99,  102,  106,  109,  112,  116,  119,  122,  126,  129,  132, /*040*/
             136,  139,  142,  146,  149,  152,  156,  159,  162,  166,  169,  172,  176,  179,  182,  186,  189,  192,  195,  199, /*060*/
             202,  205,  209,  212,  215,  219,  222,  225,  229,  232,  235,  239,  242,  245,  249,  252,  255,  259,  262,  265, /*080*/
             269,  272,  275,  279,  282,  285,  289,  292,  295,  298,  302,  305,  308,  312,  315,  318,  322,  325,  328,  332, /*100*/
             335,  338,  342,  345,  348,  352,  355,  358,  362,  365,  368,  372,  375,  378,  382,  385,  388,  391,  395,  398, /*120*/
             401,  405,  408,  411,  415,  418,  421,  425,  428,  431,  435,  438,  441,  445,  448,  451,  455,  458,  461,  465, /*140*/
             468,  471,  475,  478,  481,  485,  488,  491,  494,  498,  501,  504,  508,  511,  514,  518,  521,  524,  528,  531, /*160*/
             534,  538,  541,  544,  548,  551,  554,  558,  561,  564,  568,  571,  574,  578,  581,  584,  587,  591,  594,  597, /*180*/
             601,  604,  607,  611,  614,  617,  621,  624,  627,  631,  634,  637,  641,  644,  647,  651,  654,  657,  661,  664, /*200*/
             667,  671,  674,  677,  680,  684,  687,  690,  694,  697,  700,  704,  707,  710,  714,  717,  720,  724,  727,  730, /*220*/
             734,  737,  740,  744,  747,  750,  754,  757,  760,  764,  767,  770,  774,  777,  780,  783,  787,  790,  793,  797, /*240*/
             800,  803,  807,  810,  813,  817,  820,  823,  827,  830,  833,  837,  840,  843,  847,  850,  853,  857,  860,  863, /*260*/
             867,  870,  873,  876,  880,  883,  886,  890,  893,  896,  900,  903,  906,  910,  913,  916,  920,  923,  926,  930, /*280*/
             933,  936,  940,  943,  946,  950,  953,  956,  960,  963,  966,  970,  973,  976,  979,  983,  986,  989,  993,  996, /*300*/
             999, 1003, 1006, 1009, 1013, 1016, 1019, 1023, 1026, 1029, 1033, 1036, 1039, 1043, 1046, 1049, 1053, 1056, 1059, 1063, /*320*/
            1066, 1069, 1072, 1076, 1079, 1082, 1086, 1089, 1092, 1096, 1099, 1102, 1106, 1109, 1112, 1116, 1119, 1122, 1126, 1129, /*340*/
            1132, 1136, 1139, 1142, 1146, 1149, 1152, 1156, 1159, 1162, 1165, 1169, 1172, 1175, 1179, 1182, 1185, 1189, 1192, 1195, /*360*/
        };

        // Map from base-10 exponent to Double. Note that since this table is only used for the Single side,
        // we don't need all the values - only until the value cast to Single is infinite.
        private static readonly Double[] _mpe10Dbl;

        // Map from negative base-10 exponent to Double. Note that since this table is only used for the Single side,
        // we don't need all the values - only until the value times 2^64 cast to Single is zero.
        private static readonly Double[] _mpne10Dbl;

        // Build the Double tables from the mantissa/exponent tables.
        static DoubleParser()
        {
            Contracts.Assert(_mpe10Man.Length == _mpe10e2.Length);
            Contracts.Assert(_mpne10Man.Length == _mpne10ne2.Length);

            // Initialize the Double valued tables.
            _mpe10Dbl = new Double[39];
            Contracts.Assert(_mpe10Dbl.Length <= _mpe10Man.Length);
            for (int i = 0; i < _mpe10Dbl.Length; i++)
            {
                ulong man = _mpe10Man[i];

                // Adjust so the high bit is clear so we can use long => Double. For 10^27 and beyond
                // man will not have all the bits, hence we or in 1 (to make sure rounding is correct).
                man >>= 1;
                if (i >= 26)
                    man |= 1;

                Double dbl = (Double)(ulong)man;
                int e2 = _mpe10e2[i] + (0x3FF - 63);
                Contracts.Assert(0 < e2 & e2 < 0x7FF);
                ulong mul = (ulong)e2 << 52;
                unsafe { dbl *= *(Double*)&mul; }
                _mpe10Dbl[i] = dbl;
            }
            Contracts.Assert((Single)_mpe10Dbl[_mpe10Dbl.Length - 1] == Single.PositiveInfinity);
            Contracts.Assert((Single)_mpe10Dbl[_mpe10Dbl.Length - 2] < Single.PositiveInfinity);

            // Note that since this table is only used for the Single side, we don't need
            // any denormals - we go straight from a biased exponent of 1 to zero.
            _mpne10Dbl = new Double[65];
            Contracts.Assert(_mpne10Dbl.Length <= _mpne10Man.Length);
            for (int i = 0; i < _mpne10Dbl.Length; i++)
            {
                Double dbl = _mpne10Man[i];
                int e2 = -_mpne10ne2[i] + (0x3FF - 64);
                Contracts.Assert(0 < e2 & e2 < 0x7FF);
                ulong mul = (ulong)e2 << 52;
                unsafe { dbl *= *(Double*)&mul; }
                _mpne10Dbl[i] = dbl;
            }
#if DEBUG
            Double two64 = (Double)(1UL << 32) * (Double)(1UL << 32);
            Contracts.Assert((Single)(_mpne10Dbl[_mpne10Dbl.Length - 1] * two64) == 0);
            Contracts.Assert((Single)(_mpne10Dbl[_mpne10Dbl.Length - 2] * two64) > 0);
#endif
        }
    }
}
