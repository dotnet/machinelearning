﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Internal.CpuMath;

namespace Microsoft.ML.CpuMath.PerformanceTests
{
    public abstract class PerformanceTests
    {
        private const int ExponentMax = 127;
        private const int ExponentMin = 0;
        private const int ExponentRange = ExponentMax / 8;

        protected const int IndexLength = 1000003;
        protected const int Length = 1000003;
        protected const int MatrixIndexLength = 1000;

        private const int DefaultSeed = 253421;
        protected const float DefaultScale = 1.11f;
        protected int matrixLength = 1000;
        protected virtual int align { get; set; } = 16;

        internal AlignedArray testMatrixAligned;
        internal AlignedArray testSrcVectorAligned;
        internal AlignedArray testDstVectorAligned;

        protected float[] src;
        protected float[] dst;
        protected float[] original;
        protected float[] src1;
        protected float[] src2;
        protected float[] result;
        protected int[] idx;
        protected int[] matrixIdx;

        private int _seed = DefaultSeed;

        private float NextFloat(Random rand, int expRange)
        {
            double mantissa = (rand.NextDouble() * 2.0) - 1.0;
            double exponent = Math.Pow(2.0, rand.Next(-expRange + 1, expRange + 1));
            return (float)(mantissa * exponent);
        }

        private int GetSeed()
        {
            int seed = DefaultSeed;
            string cpumathSeed = Environment.GetEnvironmentVariable("CPUMATH_SEED");

            if (cpumathSeed != null)
            {
                if (!int.TryParse(cpumathSeed, out seed))
                {
                    if (string.Equals(cpumathSeed, "random", StringComparison.OrdinalIgnoreCase))
                    {
                        seed = new Random().Next();
                    }
                    else
                    {
                        seed = DefaultSeed;
                    }
                }
            }

            Console.WriteLine("Random seed: " + seed + "; set environment variable CPUMATH_SEED to this value to reproduce results");
            return seed;
        }

        [GlobalSetup]
        public void Setup()
        {
            src = new float[Length];
            dst = new float[Length];
            src1 = new float[Length];
            src2 = new float[Length];
            original = new float[Length];
            result = new float[Length];
            idx = new int[IndexLength];
            matrixIdx = new int[MatrixIndexLength];

            _seed = GetSeed();
            Random rand = new Random(_seed);

            for (int i = 0; i < Length; i++)
            {
                src[i] = NextFloat(rand, ExponentRange);
                dst[i] = NextFloat(rand, ExponentRange);
                original[i] = dst[i];
                result[i] = dst[i];
                src1[i] = NextFloat(rand, ExponentRange);
                src2[i] = NextFloat(rand, ExponentRange);
            }

            for (int i = 0; i < IndexLength; i++)
            {
                idx[i] = rand.Next(0, Length);
            }

            for (int i = 0; i < MatrixIndexLength; i++)
            {
                matrixIdx[i] = rand.Next(0, 1000);
            }

            testMatrixAligned = new AlignedArray(matrixLength * matrixLength, align);
            testMatrixAligned.CopyFrom(src.AsSpan(0, (matrixLength - 1) * ( matrixLength - 1)));

            testSrcVectorAligned = new AlignedArray(matrixLength, align);
            testSrcVectorAligned.CopyFrom(src1.AsSpan(0, matrixLength - 1)); // odd input

            testDstVectorAligned = new AlignedArray(matrixLength, align);
            testDstVectorAligned.CopyFrom(dst.AsSpan(0, matrixLength));
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            original.CopyTo(dst, 0);
            original.CopyTo(result, 0);
        }
    }
}