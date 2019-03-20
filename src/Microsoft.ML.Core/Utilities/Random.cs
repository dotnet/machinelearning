// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    [BestFriend]
    internal static class RandomUtils
    {
        public static float NextSingle(this Random random)
        {
            if (random is TauswortheHybrid tauswortheHybrd)
            {
                return tauswortheHybrd.NextSingle();
            }

            for (; ; )
            {
                // Since the largest value that NextDouble() can return rounds to 1 when cast to float,
                // we need to protect against returning 1.
                var res = (float)random.NextDouble();
                if (res < 1.0f)
                    return res;
            }
        }

        public static int NextSigned(this Random random)
        {
            if (random is TauswortheHybrid tauswortheHybrd)
            {
                return tauswortheHybrd.NextSigned();
            }

            // Note that, according to the documentation for System.Random,
            // this won't ever achieve int.MaxValue, but oh well.
            return random.Next(int.MinValue, int.MaxValue);
        }

        public static TauswortheHybrid Create()
        {
            // Seed from a system random.
            return new TauswortheHybrid(new Random());
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

        public static TauswortheHybrid Create(Random seed)
        {
            return new TauswortheHybrid(seed);
        }
    }

    /// <summary>
    /// Tausworthe hybrid random number generator.
    /// </summary>
    [BestFriend]
    internal sealed class TauswortheHybrid : Random
    {
        public readonly struct State
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

        public TauswortheHybrid(Random rng)
        {
            _z1 = GetSeed(rng);
            _z2 = GetSeed(rng);
            _z3 = GetSeed(rng);
            _z4 = GetU(rng);
        }

        private static uint GetU(Random rng)
        {
            return ((uint)rng.Next(0x00010000) << 16) | ((uint)rng.Next(0x00010000));
        }

        private static uint GetSeed(Random rng)
        {
            for (; ; )
            {
                uint u = GetU(rng);
                if (u >= 128)
                    return u;
            }
        }

        public float NextSingle()
        {
            NextState();
            return GetSingle();
        }

        public override double NextDouble()
        {
            NextState();
            return GetDouble();
        }

        public override int Next()
        {
            NextState();
            uint u = GetUint();
            int n = (int)(u + (u & 0x80000000U));
            Contracts.Assert(n >= 0);
            return n;
        }

        public override int Next(int maxValue)
        {
            Contracts.CheckParam(maxValue >= 0, nameof(maxValue), "maxValue must be non-negative");
            NextState();
            uint u = GetUint();
            ulong uu = (ulong)u * (ulong)maxValue;
            int res = (int)(uu >> 32);
            Contracts.Assert(0 <= res && (res < maxValue || res == 0));
            return res;
        }

        public override int Next(int minValue, int maxValue)
        {
            Contracts.CheckParam(minValue <= maxValue, nameof(minValue), "minValue must be less than or equal to maxValue.");

            long range = (long)maxValue - minValue;
            return (int)((long)(NextDouble() * range) + minValue);
        }

        public override void NextBytes(byte[] buffer)
        {
            Contracts.CheckValue(buffer, nameof(buffer));

            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = (byte)Next();
            }
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

        private float GetSingle()
        {
            const float scale = (float)1 / (1 << 23);

            // Drop the low 9 bits so the conversion to Single is exact. Allowing rounding would cause
            // issues with biasing values and, worse, the possibility of returning exactly 1.
            uint u = GetUint() >> 9;
            Contracts.Assert((uint)(float)u == u);

            return (float)u * scale;
        }

        private double GetDouble()
        {
            const double scale = (double)1 / (1 << 16) / (1 << 16);
            uint u = GetUint();
            return (double)u * scale;
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
}