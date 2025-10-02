// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif

namespace Microsoft.ML.Internal.Utilities
{
    /// <summary>
    /// Highly optimized SIMD-enabled Mersenne Twister implementation.
    /// </summary>
    internal sealed class MersenneTwister
    {
        private const int N = 624;
        private const int M = 397;
        private const uint MatrixA = 0x9908B0DFU;
        private const uint UpperMask = 0x80000000U;
        private const uint LowerMask = 0x7FFFFFFFU;

        private readonly uint[] _mt = new uint[N];
        private int _mti = N + 1;

        private readonly uint[] _buf = new uint[N];
        private uint _carry;
        private bool _hasCarry;

        public MersenneTwister(uint seed)
        {
            InitGenrand(seed);
        }

        private void InitGenrand(uint s)
        {
            unchecked
            {
                _mt[0] = s;
                for (_mti = 1; _mti < N; _mti++)
                {
                    var x = _mt[_mti - 1];
                    _mt[_mti] = 1812433253U * (x ^ (x >> 30)) + (uint)_mti;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Twist()
        {
            var kk = 0;
            uint y;

            for (; kk < N - M; kk++)
            {
                y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                _mt[kk] = _mt[kk + M] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            }

            for (; kk < N - 1; kk++)
            {
                y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                _mt[kk] = _mt[kk - (N - M)] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            }

            y = (_mt[N - 1] & UpperMask) | (_mt[0] & LowerMask);
            _mt[N - 1] = _mt[M - 1] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            _mti = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void TemperScalar(ReadOnlySpan<uint> src, Span<uint> dst)
        {
            for (var i = 0; i < src.Length; i++)
            {
                var y = src[i];
                y ^= (y >> 11);
                y ^= (y << 7) & 0x9D2C5680U;
                y ^= (y << 15) & 0xEFC60000U;
                y ^= (y >> 18);
                dst[i] = y;
            }
        }

#if NET8_0_OR_GREATER
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void TemperAvx2(ReadOnlySpan<uint> src, Span<uint> dst)
        {
            var len = src.Length;
            var i = 0;
            var c7 = Vector256.Create(0x9D2C5680u);
            var c15 = Vector256.Create(0xEFC60000u);

            fixed (uint* pSrc = src)
            fixed (uint* pDst = dst)
            {
                for (; i + 8 <= len; i += 8)
                {
                    var y = Avx.LoadVector256(pSrc + i);
                    y = Avx2.Xor(y, Avx2.ShiftRightLogical(y, 11));

                    var t = Avx2.And(Avx2.ShiftLeftLogical(y, 7), c7);
                    y = Avx2.Xor(y, t);

                    t = Avx2.And(Avx2.ShiftLeftLogical(y, 15), c15);
                    y = Avx2.Xor(y, t);

                    y = Avx2.Xor(y, Avx2.ShiftRightLogical(y, 18));
                    Avx.Store(pDst + i, y);
                }
            }

            for (; i < len; i++)
            {
                var y = src[i];
                y ^= (y >> 11);
                y ^= (y << 7) & 0x9D2C5680u;
                y ^= (y << 15) & 0xEFC60000u;
                y ^= (y >> 18);
                dst[i] = y;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void TemperAdvSimd(ReadOnlySpan<uint> src, Span<uint> dst)
        {
            if (!AdvSimd.IsSupported)
                throw new PlatformNotSupportedException("AdvSimd is not supported on this CPU.");

            var len = src.Length;
            var i = 0;
            var c7 = Vector128.Create(0x9D2C5680u);
            var c15 = Vector128.Create(0xEFC60000u);

            fixed (uint* pSrc = src)
            fixed (uint* pDst = dst)
            {
                for (; i + 4 <= len; i += 4)
                {
                    var y = Unsafe.ReadUnaligned<Vector128<uint>>((byte*)(pSrc + i));
                    y = AdvSimd.Xor(y, AdvSimd.ShiftRightLogical(y, 11));

                    var t = AdvSimd.And(AdvSimd.ShiftLeftLogical(y, 7), c7);
                    y = AdvSimd.Xor(y, t);

                    t = AdvSimd.And(AdvSimd.ShiftLeftLogical(y, 15), c15);
                    y = AdvSimd.Xor(y, t);

                    y = AdvSimd.Xor(y, AdvSimd.ShiftRightLogical(y, 18));
                    Unsafe.WriteUnaligned((byte*)(pDst + i), y);
                }
            }

            for (; i < len; i++)
            {
                var y = src[i];
                y ^= (y >> 11);
                y ^= (y << 7) & 0x9D2C5680u;
                y ^= (y << 15) & 0xEFC60000u;
                y ^= (y >> 18);
                dst[i] = y;
            }
        }
#endif

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Temper(ReadOnlySpan<uint> src, Span<uint> dst)
        {
#if NET8_0_OR_GREATER
            if (src.Length >= 8 && Avx2.IsSupported)
            {
                TemperAvx2(src, dst);
                return;
            }

            if (src.Length >= 4 && AdvSimd.IsSupported)
            {
                TemperAdvSimd(src, dst);
                return;
            }
#endif
            TemperScalar(src, dst);
        }

        private const double DoubleDivisor = 1.0 / 9007199254740992.0; // 1 / 2^53

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double DoubleFromMant53(ulong mant53)
        {
            return mant53 * DoubleDivisor;
        }

        public double NextDouble()
        {
            Span<double> buffer = stackalloc double[1];
            NextDoubles(buffer);
            return buffer[0];
        }

        public unsafe void NextDoubles(Span<double> destination)
        {
            var n = destination.Length;
            var filled = 0;

            fixed (uint* pMt = _mt)
            fixed (uint* pBuf = _buf)
            fixed (double* pDst = destination)
            {
                while (filled < n)
                {
                    if (!_hasCarry)
                    {
                        if (_mti >= N)
                            Twist();

                        Temper(new ReadOnlySpan<uint>(pMt + _mti, 1), new Span<uint>(pBuf, 1));
                        _carry = pBuf[0];
                        _mti += 1;
                        _hasCarry = true;
                    }

                    if (_mti >= N)
                        Twist();

                    var pairsRemaining = n - filled;
                    var availInts = N - _mti;

                    if (availInts == 0)
                    {
                        Twist();
                        continue;
                    }

                    var maxPairsFromAvail = 1 + ((availInts - 1) >> 1);
                    var makePairs = Math.Min(pairsRemaining, maxPairsFromAvail);
                    var wantInts = (makePairs << 1) - 1;

                    Temper(new ReadOnlySpan<uint>(pMt + _mti, wantInts), new Span<uint>(pBuf, wantInts));
                    _mti += wantInts;

                    var j = 0;
                    var a = (ulong)(_carry >> 5);
                    var b = (ulong)(pBuf[j++] >> 6);
                    pDst[filled++] = DoubleFromMant53((a << 26) | b);
                    _hasCarry = false;

                    var remainingPairs = makePairs - 1;
                    for (var p = 0; p < remainingPairs; p++)
                    {
                        a = (ulong)(pBuf[j++] >> 5);
                        b = (ulong)(pBuf[j++] >> 6);
                        pDst[filled++] = DoubleFromMant53((a << 26) | b);
                    }

                    if (filled < n)
                    {
                        var intsLeftBeforeTwist = N - _mti;
                        if (intsLeftBeforeTwist == 1)
                        {
                            Temper(new ReadOnlySpan<uint>(pMt + _mti, 1), new Span<uint>(pBuf, 1));
                            _carry = pBuf[0];
                            _mti += 1;
                            _hasCarry = true;
                        }
                    }
                }
            }
        }

        public unsafe void NextTemperedUInt32(Span<uint> destination)
        {
            var n = destination.Length;
            var filled = 0;

            if (_hasCarry && n != 0)
            {
                destination[filled++] = _carry;
                _hasCarry = false;
            }

            if (filled >= n)
                return;

            fixed (uint* pMt = _mt)
            fixed (uint* pBuf = _buf)
            fixed (uint* pDst = destination)
            {
                while (filled < n)
                {
                    if (_mti >= N)
                        Twist();

                    var avail = N - _mti;

                    if (avail == 0)
                    {
                        Twist();
                        continue;
                    }

                    var toProduce = Math.Min(avail, n - filled);

                    Temper(new ReadOnlySpan<uint>(pMt + _mti, toProduce), new Span<uint>(pBuf, toProduce));
                    _mti += toProduce;

                    new ReadOnlySpan<uint>(pBuf, toProduce).CopyTo(new Span<uint>(pDst + filled, toProduce));
                    filled += toProduce;
                }
            }
        }

        public uint NextTemperedUInt32()
        {
            if (_hasCarry)
            {
                _hasCarry = false;
                return _carry;
            }

            if (_mti >= N)
                Twist();

            var y = _mt[_mti++];
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9D2C5680u;
            y ^= (y << 15) & 0xEFC60000u;
            y ^= (y >> 18);
            return y;
        }
    }
}
