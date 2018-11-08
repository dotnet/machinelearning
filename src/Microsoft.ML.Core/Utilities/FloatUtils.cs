// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    public static class FloatUtils
    {
        // This is used to read and write the bits of a Double.
        // Thanks to Vance Morrison for educating me about this excellent aliasing mechanism.
        [StructLayout(LayoutKind.Explicit)]
        private struct DoubleBits
        {
            // Masks for the portions of a Double: 1 sign bit, 11 exponent bits, 52 mantissa bits.
            public const ulong MaskSign = 0x8000000000000000UL;
            public const ulong MaskExp = 0x7FF0000000000000UL;
            public const ulong MaskMan = 0x000FFFFFFFFFFFFFUL;

            public const int RawExpInf = 0x7FF; // The raw exponent value for infinities and nan.
            public const int RawExpZero = 0x3FF; // The raw exponent value for "1", when the exponent is logically zero.
            public const int CbitExp = 11; // Number of exponent bits.
            public const int CbitMan = 52; // Number of mantissa bits.
            public const int ExpDenorm = -1074;
            public const int ExpOrigin = 1075;

            [FieldOffset(0)]
            public double Float; // used only for construction from Double
            [FieldOffset(0)]
            public ulong Bits; //overlay

            public int GetExp()
            {
                return ((int)(Bits >> CbitMan) & RawExpInf) - RawExpZero;
            }
            public int GetRawExp()
            {
                return (int)(Bits >> CbitMan) & RawExpInf;
            }
            public bool IsFinite()
            {
                return (Bits & MaskExp) < MaskExp;
            }
            public bool IsFiniteNonZero()
            {
                var bits = Bits & ~MaskSign;
                return 0 < bits && bits < MaskExp;
            }
            public bool IsFiniteNormal()
            {
                var exp = Bits & MaskExp;
                return 0 < exp && exp < MaskExp;
            }
            public bool IsDenormal()
            {
                return (Bits & MaskExp) == 0;
            }
            public void SetExponent(int exp)
            {
                Contracts.Assert(-RawExpZero < exp && exp < RawExpInf - RawExpZero);
                Bits = (Bits & ~MaskExp) | (((ulong)(exp + RawExpZero) << CbitMan) & MaskExp);
            }
            public void GetParts(out int sign, out int exp, out ulong man, out bool fFinite)
            {
                sign = 1 - ((int)(Bits >> 62) & 2);
                man = Bits & MaskMan;
                exp = GetRawExp();
                if (exp == 0)
                {
                    // Denormalized number.
                    fFinite = true;
                    if (man != 0)
                        exp = ExpDenorm;
                }
                else if (exp == RawExpInf)
                {
                    // NaN or infinite.
                    fFinite = false;
                    exp = int.MaxValue;
                }
                else
                {
                    fFinite = true;
                    man += MaskMan + 1;
                    exp -= ExpOrigin;
                }
            }
            public void SetFromParts(int sign, int exp, ulong man)
            {
                if (man == 0)
                    Bits = 0;
                else
                {
                    // Normalize so that 0x0010 0000 0000 0000 is the highest bit set.
                    int cbitShift = Utils.CbitHighZero(man) - CbitExp;
                    if (cbitShift < 0)
                    {
                        // REVIEW: Should this round?
                        man >>= -cbitShift;
                    }
                    else
                        man <<= cbitShift;
                    exp -= cbitShift;
                    Contracts.Assert((man & ~MaskMan) == MaskMan + 1);

                    // Move the point to just behind the leading 1: 0x001.0 0000 0000 0000
                    // (52 bits) and skew the exponent (by 0x3FF == 1023).
                    exp += ExpOrigin;

                    if (exp >= RawExpInf)
                    {
                        // Infinity.
                        Bits = MaskExp;
                    }
                    else if (exp <= 0)
                    {
                        // Denormalized.
                        exp--;
                        if (exp < -CbitMan)
                        {
                            // Underflow to zero.
                            Bits = 0;
                        }
                        else
                        {
                            Bits = man >> (int)(-exp);
                            Contracts.Assert(Bits != 0);
                        }
                    }
                    else
                    {
                        // Mask off the implicit high bit.
                        Bits = (man & MaskMan) | ((ulong)exp << CbitMan);
                    }
                }

                if (sign < 0)
                    Bits |= MaskSign;
            }
            public void SetPowerOfTwo(int exp)
            {
                if (exp >= RawExpInf || (exp += RawExpZero) >= RawExpInf)
                    Bits = MaskExp; // Overflow to infinity.
                else if (exp > 0)
                    Bits = (ulong)exp << CbitMan;
                else if ((exp += CbitMan - 1) < 0)
                    Bits = 0; // Underflow to zero.
                else
                    Bits = 1UL << exp; // Denormal.
            }
            public void NormalizeExponent()
            {
                // This doesn't work on denormalized numbers or non-finite values.
                Contracts.Assert(IsFiniteNormal());
                Bits = (Bits & ~MaskExp) | ((ulong)RawExpZero << CbitMan);
            }
            public void TruncateMantissaToSingleBit()
            {
                if (Bits == 0)
                    return;
                int exp = GetRawExp();
                if (exp == RawExpInf)
                    return;
                if (exp != 0)
                {
                    // Mantissa has an implicit high bit.
                    Bits &= ~MaskMan;
                }
                else
                {
                    // Denormalized number.
                    // Clear all but the high bit in the mantissa.
                    ulong tmp;
                    ulong man = Bits & MaskMan;
                    while ((tmp = man & (man - 1)) != 0)
                        man = tmp;
                    Bits = (Bits & MaskSign) | man;
                }
            }
            public void Truncate()
            {
                int exp = GetRawExp();
                if (exp == RawExpInf) // NaN or infinite.
                    return;

                exp -= ExpOrigin;
                if (exp >= 0)
                    return;

                if (exp <= -CbitMan - 1)
                {
                    // Preserve only the sign. The rest is zero.
                    Bits &= MaskSign;
                }
                else
                {
                    // Clear the low (-exp) bits.
                    Bits &= ~Utils.UuMaskBelow(-exp);
                }
            }
        }

        // This is used to read and write the bits of a Single.
        // Thanks to Vance Morrison for educating me about this excellent aliasing mechanism.
        [StructLayout(LayoutKind.Explicit)]
        private struct SingleBits
        {
            // Masks for the portions of a Single: 1 sign bit, 8 exponent bits, 23 mantissa bits.
            public const uint MaskSign = 0x80000000U;
            public const uint MaskExp = 0x7F800000U;
            public const uint MaskMan = 0x007FFFFFU;

            public const int RawExpInf = 0xFF; // The raw exponent value for infinities and nan.
            public const int RawExpZero = 0x7F; // The raw exponent value for "1", when the exponent is logically zero.
            public const int CbitExp = 8; // Number of exponent bits.
            public const int CbitMan = 23; // Number of mantissa bits.
            public const int ExpDenorm = -126;
            public const int ExpOrigin = 127;

            [FieldOffset(0)]
            public float Float; // used only for construction from Single
            [FieldOffset(0)]
            public uint Bits; //overlay

            public int GetExp()
            {
                return ((int)(Bits >> CbitMan) & RawExpInf) - RawExpZero;
            }
            public int GetRawExp()
            {
                return (int)(Bits >> CbitMan) & RawExpInf;
            }
            public bool IsFinite()
            {
                return (Bits & MaskExp) < MaskExp;
            }
            public bool IsFiniteNonZero()
            {
                var bits = Bits & ~MaskSign;
                return 0 < bits && bits < MaskExp;
            }
            public bool IsFiniteNormal()
            {
                var exp = Bits & MaskExp;
                return 0 < exp && exp < MaskExp;
            }
            public bool IsDenormal()
            {
                return (Bits & MaskExp) == 0;
            }
            public void SetExponent(int exp)
            {
                Contracts.Assert(-RawExpZero < exp && exp < RawExpInf - RawExpZero);
                Bits = (Bits & ~MaskExp) | (((uint)(exp + RawExpZero) << CbitMan) & MaskExp);
            }
            public void GetParts(out int sign, out int exp, out uint man, out bool fFinite)
            {
                sign = 1 - ((int)(Bits >> 30) & 2);
                man = Bits & MaskMan;
                exp = GetRawExp();
                if (exp == 0)
                {
                    // Denormalized number.
                    fFinite = true;
                    if (man != 0)
                        exp = ExpDenorm;
                }
                else if (exp == RawExpInf)
                {
                    // NaN or infinite.
                    fFinite = false;
                    exp = int.MaxValue;
                }
                else
                {
                    fFinite = true;
                    man += MaskMan + 1;
                    exp += ExpDenorm - 1;
                }
            }
            public void SetFromParts(int sign, int exp, uint man)
            {
                if (man == 0)
                    Bits = 0;
                else
                {
                    // Normalize so that 0x0010 0000 0000 0000 is the highest bit set.
                    int cbitShift = Utils.CbitHighZero(man) - CbitExp;
                    if (cbitShift < 0)
                    {
                        // REVIEW: Should this round?
                        man >>= -cbitShift;
                    }
                    else
                        man <<= cbitShift;
                    exp -= cbitShift;
                    Contracts.Assert((man & ~MaskMan) == MaskMan + 1);

                    // Move the point to just behind the leading 1: 0x001.0 0000 0000 0000
                    // (52 bits) and skew the exponent (by 0x3FF == 1023).
                    exp += ExpOrigin;

                    if (exp >= RawExpInf)
                    {
                        // Infinity.
                        Bits = MaskExp;
                    }
                    else if (exp <= 0)
                    {
                        // Denormalized.
                        exp--;
                        if (exp < -CbitMan)
                        {
                            // Underflow to zero.
                            Bits = 0;
                        }
                        else
                        {
                            Bits = man >> (int)(-exp);
                            Contracts.Assert(Bits != 0);
                        }
                    }
                    else
                    {
                        // Mask off the implicit high bit.
                        Bits = (man & MaskMan) | ((uint)exp << CbitMan);
                    }
                }

                if (sign < 0)
                    Bits |= MaskSign;
            }
            public void SetPowerOfTwo(int exp)
            {
                if (exp >= RawExpInf || (exp += RawExpZero) >= RawExpInf)
                    Bits = MaskExp; // Overflow to infinity.
                else if (exp > 0)
                    Bits = (uint)exp << CbitMan;
                else if ((exp += CbitMan - 1) < 0)
                    Bits = 0; // Underflow to zero.
                else
                    Bits = 1U << exp; // Denormal.
            }
            public void NormalizeExponent()
            {
                // This doesn't work on denormalized numbers or non-finite values.
                Contracts.Assert(IsFiniteNormal());
                Bits = (Bits & ~MaskExp) | ((uint)RawExpZero << CbitMan);
            }
            public void TruncateMantissaToSingleBit()
            {
                if (Bits == 0)
                    return;
                int exp = GetRawExp();
                if (exp == RawExpInf)
                    return;
                if (exp != 0)
                {
                    // Mantissa has an implicit high bit.
                    Bits &= ~MaskMan;
                }
                else
                {
                    // Denormalized number.
                    // Clear all but the high bit in the mantissa.
                    uint tmp;
                    uint man = Bits & MaskMan;
                    while ((tmp = man & (man - 1)) != 0)
                        man = tmp;
                    Bits = (Bits & MaskSign) | man;
                }
            }
            public void Truncate()
            {
                int exp = GetRawExp();
                if (exp == RawExpInf) // NaN or infinite.
                    return;

                exp -= ExpOrigin;
                if (exp >= 0)
                    return;

                if (exp <= -CbitMan - 1)
                {
                    // Preserve only the sign. The rest is zero.
                    Bits &= MaskSign;
                }
                else
                {
                    // Clear the low (-exp) bits.
                    Bits &= ~Utils.UMaskBelow(-exp);
                }
            }
        }

        public static ulong GetBits(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            return bits.Bits;
        }

        public static uint GetBits(float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            return bits.Bits;
        }

        public static double FromBits(ulong bits)
        {
            var sb = default(DoubleBits);
            sb.Bits = bits;
            return sb.Float;
        }

        public static float FromBits(uint bits)
        {
            var sb = default(SingleBits);
            sb.Bits = bits;
            return sb.Float;
        }

        public static bool IsFinite(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            return bits.IsFinite();
        }

        public static bool IsFinite(float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            return bits.IsFinite();
        }

        public static bool IsFinite(ReadOnlySpan<double> values)
        {
            // Assuming that non-finites are rare, this is faster than testing on each item.
            double sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                var v = values[i];
                sum += v - v;
            }
            return sum == 0;
        }

        // REVIEW: Consider implementing using SSE.
        public static bool IsFinite(ReadOnlySpan<float> values)
        {
            // Assuming that non-finites are rare, this is faster than testing on each item.
            float sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                var v = values[i];
                sum += v - v;
            }
            return sum == 0;
        }

        public static bool IsFiniteNonZero(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            return bits.IsFiniteNonZero();
        }

        public static bool IsFiniteNonZero(float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            return bits.IsFiniteNonZero();
        }

        public static bool IsFiniteNormal(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            return bits.IsFiniteNormal();
        }

        public static bool IsFiniteNormal(float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            return bits.IsFiniteNormal();
        }

        public static bool IsDenormal(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            return bits.IsDenormal();
        }

        public static bool IsDenormal(float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            return bits.IsDenormal();
        }

        public static int GetExponent(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            return bits.GetExp();
        }

        public static int GetExponent(float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            return bits.GetExp();
        }

        public static double SetExponent(double x, int exp)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            bits.SetExponent(exp);
            return bits.Float;
        }

        public static float SetExponent(float x, int exp)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            bits.SetExponent(exp);
            return bits.Float;
        }

        public static void GetParts(double x, out int sign, out int exp, out ulong man, out bool fFinite)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            bits.GetParts(out sign, out exp, out man, out fFinite);
        }

        public static void GetParts(float x, out int sign, out int exp, out uint man, out bool fFinite)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            bits.GetParts(out sign, out exp, out man, out fFinite);
        }

        public static double GetFromPartsDouble(int sign, int exp, ulong man)
        {
            var bits = default(DoubleBits);
            bits.SetFromParts(sign, exp, man);
            return bits.Float;
        }

        public static float GetFromPartsSingle(int sign, int exp, uint man)
        {
            var bits = default(SingleBits);
            bits.SetFromParts(sign, exp, man);
            return bits.Float;
        }

        public static double GetPowerOfTwoDouble(int exp)
        {
            var bits = default(DoubleBits);
            bits.SetPowerOfTwo(exp);
            return bits.Float;
        }

        public static float GetPowerOfTwoSingle(int exp)
        {
            var bits = default(SingleBits);
            bits.SetPowerOfTwo(exp);
            return bits.Float;
        }

        // Returns the previous exponent and sets the exponent to zero. Asserts that
        // the original value is finite and not a denormal.
        public static int NormalizeExponent(ref double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            int expTmp = bits.GetExp();
            bits.NormalizeExponent();
            x = bits.Float;
            return expTmp;
        }

        // Returns the previous exponent and sets the exponent to zero. Asserts that
        // the original value is finite and not a denormal.
        public static int NormalizeExponent(ref float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            int expTmp = bits.GetExp();
            bits.NormalizeExponent();
            x = bits.Float;
            return expTmp;
        }

        public static double TruncateMantissaToSingleBit(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            bits.TruncateMantissaToSingleBit();
            return bits.Float;
        }

        public static float TruncateMantissaToSingleBit(float x)
        {
            var bits = default(SingleBits);
            bits.Float = x;
            bits.TruncateMantissaToSingleBit();
            return bits.Float;
        }

        public static double Truncate(double x)
        {
            var bits = default(DoubleBits);
            bits.Float = x;
            bits.Truncate();
            return bits.Float;
        }

        public static string ToRoundTripString(float x)
        {
            return x.ToString("R", CultureInfo.InvariantCulture);
        }

        public static string ToRoundTripString(double x)
        {
            return x.ToString("G17", CultureInfo.InvariantCulture);
        }
    }
}
