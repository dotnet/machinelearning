// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime
{
    public interface IRandom
    {
        /// <summary>
        /// Generates a Single in the range [0, 1).
        /// </summary>
        Single NextSingle();

        /// <summary>
        /// Generates a Double in the range [0, 1).
        /// </summary>
        Double NextDouble();

        /// <summary>
        /// Generates an int in the range [0, int.MaxValue]. Note that this differs
        /// from the contract for System.Random.Next, which claims to never return
        /// int.MaxValue.
        /// </summary>
        int Next();

        /// <summary>
        /// Generates an int in the range [int.MinValue, int.MaxValue].
        /// </summary>
        int NextSigned();

        /// <summary>
        /// Generates an int in the range [0, limit), unless limit == 0, in which case this advances the generator
        /// and returns 0.
        /// Throws if limit is less than 0.
        /// </summary>
        int Next(int limit);
    }

    public static class RandomUtils
    {
        public static TauswortheHybrid Create()
        {
            // Seed from a system random.
            return new TauswortheHybrid(new SysRandom());
        }

        public static TauswortheHybrid Create(int? seed)
        {
            if (seed == null)
                return Create();
            return Create(seed.GetValueOrDefault());
        }

        public static TauswortheHybrid Create(int seed)
        {
            var state = new TauswortheHybrid.State((uint)seed);
            return new TauswortheHybrid(state);
        }

        public static TauswortheHybrid Create(uint? seed)
        {
            if (seed == null)
                return Create();
            return Create(seed.GetValueOrDefault());
        }

        public static TauswortheHybrid Create(uint seed)
        {
            var state = new TauswortheHybrid.State(seed);
            return new TauswortheHybrid(state);
        }

        public static TauswortheHybrid Create(IRandom seed)
        {
            return new TauswortheHybrid(seed);
        }
    }

    public sealed class SysRandom : IRandom
    {
        private readonly Random _rnd;

        public SysRandom()
        {
            _rnd = new Random();
        }

        public SysRandom(int seed)
        {
            _rnd = new Random(seed);
        }

        public static SysRandom Wrap(Random rnd)
        {
            if (rnd != null)
                return new SysRandom(rnd);
            return null;
        }

        private SysRandom(Random rnd)
        {
            Contracts.AssertValue(rnd);
            _rnd = rnd;
        }

        public Single NextSingle()
        {
            // Since the largest value that NextDouble() can return rounds to 1 when cast to Single,
            // we need to protect against returning 1.
            for (;;)
            {
                var res = (Single)_rnd.NextDouble();
                if (res < 1.0f)
                    return res;
            }
        }

        public Double NextDouble()
        {
            return _rnd.NextDouble();
        }

        public int Next()
        {
            // Note that, according to the documentation for System.Random,
            // this won't ever achieve int.MaxValue, but oh well.
            return _rnd.Next();
        }

        public int Next(int limit)
        {
            Contracts.CheckParam(limit >= 0, nameof(limit), "limit must be non-negative");
            return _rnd.Next(limit);
        }

        public int NextSigned()
        {
            // Note that, according to the documentation for System.Random,
            // this won't ever achieve int.MaxValue, but oh well.
            return _rnd.Next(int.MinValue, int.MaxValue);
        }
    }

    /// <summary>
    /// Tausworthe hybrid random number generator.
    /// </summary>
    public sealed class TauswortheHybrid : IRandom
    {
        public struct State
        {
            public readonly uint U1;
            public readonly uint U2;
            public readonly uint U3;
            public readonly uint U4;

            public State(uint seed)
            {
                U1 = seed;
                U2 = Hashing.MurmurRound(U1, U1);
                U3 = Hashing.MurmurRound(U2, U1);
                U4 = Hashing.MurmurRound(U3, U1);
            }

            public State(uint u1, uint u2, uint u3, uint u4)
            {
                U1 = u1;
                U2 = u2;
                U3 = u3;
                U4 = u4;
            }

            public void Save(BinaryWriter writer)
            {
                writer.Write(U1);
                writer.Write(U2);
                writer.Write(U3);
                writer.Write(U4);
            }

            public static State Load(BinaryReader reader)
            {
                var u1 = reader.ReadUInt32();
                var u2 = reader.ReadUInt32();
                var u3 = reader.ReadUInt32();
                var u4 = reader.ReadUInt32();
                return new State(u1, u2, u3, u4);
            }
        }

        private uint _z1;
        private uint _z2;
        private uint _z3;
        private uint _z4;

        public TauswortheHybrid(State state)
        {
            _z1 = state.U1;
            _z2 = state.U2;
            _z3 = state.U3;
            _z4 = state.U4;
        }

        public TauswortheHybrid(IRandom rng)
        {
            _z1 = GetSeed(rng);
            _z2 = GetSeed(rng);
            _z3 = GetSeed(rng);
            _z4 = GetU(rng);
        }

        private static uint GetU(IRandom rng)
        {
            return ((uint)rng.Next(0x00010000) << 16) | ((uint)rng.Next(0x00010000));
        }

        private static uint GetSeed(IRandom rng)
        {
            for (;;)
            {
                uint u = GetU(rng);
                if (u >= 128)
                    return u;
            }
        }

        public Single NextSingle()
        {
            NextState();
            return GetSingle();
        }

        public Double NextDouble()
        {
            NextState();
            return GetDouble();
        }

        public int Next()
        {
            NextState();
            uint u = GetUint();
            int n = (int)(u + (u & 0x80000000U));
            Contracts.Assert(n >= 0);
            return n;
        }

        public int Next(int limit)
        {
            Contracts.CheckParam(limit >= 0, nameof(limit), "limit must be non-negative");
            NextState();
            uint u = GetUint();
            ulong uu = (ulong)u * (ulong)limit;
            int res = (int)(uu >> 32);
            Contracts.Assert(0 <= res && (res < limit || res == 0));
            return res;
        }

        public int NextSigned()
        {
            NextState();
            return (int)GetUint();
        }

        private uint GetUint()
        {
            return _z1 ^ _z2 ^ _z3 ^ _z4;
        }

        private Single GetSingle()
        {
            const Single scale = (Single)1 / (1 << 23);

            // Drop the low 9 bits so the conversion to Single is exact. Allowing rounding would cause
            // issues with biasing values and, worse, the possibility of returning exactly 1.
            uint u = GetUint() >> 9;
            Contracts.Assert((uint)(Single)u == u);

            return (Single)u * scale;
        }

        private Double GetDouble()
        {
            const Double scale = (Double)1 / (1 << 16) / (1 << 16);
            uint u = GetUint();
            return (Double)u * scale;
        }

        private void NextState()
        {
            TauswortheStateChange(ref _z1, 13, 19, 12, ~0x1U);
            TauswortheStateChange(ref _z2, 2, 25, 4, ~0x7U);
            TauswortheStateChange(ref _z3, 3, 11, 17, ~0xfU);
            LcgStateChange(ref _z4, 1664525, 1013904223);
        }

        private static void TauswortheStateChange(ref uint z, int s1, int s2, int s3, uint m)
        {
            z = ((z & m) << s3) ^ (((z << s1) ^ z) >> s2);
        }

        private static void LcgStateChange(ref uint z, uint a, uint c)
        {
            z = a * z + c;
        }

        // When creating a new TauswortheHybrid instance by using the constructor taking 4 uints, it is guaranteed that
        // the new instance will produce the same sequence of values (without the prefix that was already generated by this).
        // To get the same sequence from the start, call this method before any calls to NextSingle, NextDouble or Next.
        public State GetState()
        {
            return new State(_z1, _z2, _z3, _z4);
        }
    }

#if false // REVIEW: This was written for NN drop out but turned out to be too slow, so I inlined it instead.
    public sealed class BooleanSampler
    {
        public const int CbitRand = 25;

        private readonly IRandom _rand;
        private readonly uint _k; // probability of "true" is _k / (1U << _qlog).
        private readonly int _qlog; // Number of bits consumed by each call to Sample().
        private readonly int _cv; // Number of calls to Sample() covered by a call to _rand.Next(...).
        private readonly uint _mask; // (1U << _qlog) - 1

        // Mutable state.
        private int _c;
        private uint _v;

        /// <summary>
        /// Create a boolean sampler using the given random number generator, quantizing the true rate
        /// to cbitQuant bits, assuming that sampling the random number generator is capable of producing
        /// cbitRand good bits.
        ///
        /// For example, new BooleanSampler(0.5f, 1, 25, new Random()) will produce a reasonable fair coin flipper.
        /// Note that this reduces the parameters, so new BooleanSampler(0.5f, 6, 25, new Random()) will produce
        /// the same flipper. In other words, since 0.5 quantized to 6 bits can be reduced to only needing one
        /// bit, it reduces cbitQuant to 1.
        /// </summary>
        public static BooleanSampler Create(Single rate, int cbitQuant, IRandom rand)
        {
            Contracts.Assert(0 < rate && rate < 1);
            Contracts.Assert(0 < cbitQuant && cbitQuant <= CbitRand / 2);

            int qlog = cbitQuant;
            uint k = (uint)(rate * (1 << qlog));
            if (k == 0)
                k = 1;
            Contracts.Assert(0 <= k && k < (1U << qlog));

            while ((k & 1) == 0 && k > 0)
            {
                qlog--;
                k >>= 1;
            }
            Contracts.Assert(qlog > 0);
            uint q = 1U << qlog;
            Contracts.Assert(0 < k && k < q);

            int cv = CbitRand / qlog;
            Contracts.Assert(cv > 1);
            return new BooleanSampler(qlog, k, rand);
        }

        private BooleanSampler(int qlog, uint k, IRandom rand)
        {
            _qlog = qlog;
            _k = k;
            _rand = rand;
            _qlog = qlog;
            _cv = CbitRand / _qlog;
            _mask = (1U << _qlog) - 1;
        }

        public bool Sample()
        {
            _v >>= _qlog;
            if (--_c <= 0)
            {
                _v = (uint)_rand.Next(1 << (_cv * _qlog));
                _c = _cv;
            }
            return (_v & _mask) < _k;
        }

        public void SampleMany(out uint bits, out int count)
        {
            uint u = (uint)_rand.Next(1 << (_cv * _qlog));
            count = _cv;
            if (_qlog == 1)
            {
                bits = u;
                return;
            }

            bits = 0;
            for (int i = 0; i < count; i++)
            {
                bits <<= 1;
                if ((u & _mask) < _k)
                    bits |= 1;
                u >>= _qlog;
            }
        }
    }
#endif
}