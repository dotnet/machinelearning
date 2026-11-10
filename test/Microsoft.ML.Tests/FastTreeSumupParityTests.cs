// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    /// <summary>
    /// Verifies the optimized managed <c>Sumup</c> implementations added for platforms without the
    /// native FastTree library (e.g. arm64). Two properties are checked for Dense 4/8/16/32-bit and
    /// Segment arrays, in both the root (no doc indices) and leaf (with doc indices) cases:
    /// <list type="bullet">
    /// <item>the managed histogram matches an independent brute-force reference (runs everywhere), and</item>
    /// <item>the managed histogram is bit-identical to the native histogram (runs where FastTreeNative exists,
    /// i.e. x64 CI legs), giving the "native and managed side by side" coverage requested in the PR review.</item>
    /// </list>
    /// </summary>
    public sealed class FastTreeSumupParityTests : BaseTestClass
    {
        private const int Length = 2000;

        public FastTreeSumupParityTests(ITestOutputHelper output) : base(output)
        {
        }

        // kind, useWeights, useIndices (leaf case).
        public static IEnumerable<object[]> Cases()
        {
            foreach (var kind in new[] { "Dense4", "Dense8", "Dense16", "Dense32", "Segment" })
                foreach (var useWeights in new[] { false, true })
                    foreach (var useIndices in new[] { false, true })
                        yield return new object[] { kind, useWeights, useIndices };
        }

        [Theory]
        [MemberData(nameof(Cases))]
        public void ManagedSumupMatchesReference(string kind, bool useWeights, bool useIndices)
        {
            var arr = CreateIntArray(kind, seed: 1, out int numBins);
            var input = CreateInput(seed: 2, useWeights, useIndices, out double[] outputs, out double[] weights, out int[] docIndices, out int count);

            var managed = new FeatureHistogram(arr, numBins, useWeights);
            CallManaged(arr, input, managed);

            ComputeReference(arr, numBins, outputs, weights, docIndices, count,
                out double[] refTargets, out double[] refWeights, out int[] refCounts);

            AssertHistogramEqual(refCounts, refTargets, refWeights, managed, useWeights);
        }

        [NativeDependencyTheory("FastTreeNative")]
        [MemberData(nameof(Cases))]
        public void ManagedSumupMatchesNative(string kind, bool useWeights, bool useIndices)
        {
            // This attribute guarantees the native FastTree library is available, so the native
            // handlers below run and are compared against the managed handler.
            Assert.True(IntArray.UseFastTreeNative);

            var arr = CreateIntArray(kind, seed: 1, out int numBins);
            var input = CreateInput(seed: 2, useWeights, useIndices, out double[] outputs, out double[] weights, out int[] docIndices, out int count);

            var native = new FeatureHistogram(arr, numBins, useWeights);
            CallNative(arr, input, native);

            var managed = new FeatureHistogram(arr, numBins, useWeights);
            CallManaged(arr, input, managed);

            // Managed must match native exactly, and both must match the independent reference so a
            // shared decode mistake can't hide behind an equal-but-wrong comparison.
            ComputeReference(arr, numBins, outputs, weights, docIndices, count,
                out double[] refTargets, out double[] refWeights, out int[] refCounts);
            AssertHistogramEqual(refCounts, refTargets, refWeights, native, useWeights);
            AssertHistogramEqual(native.CountByBin, native.SumTargetsByBin, native.SumWeightsByBin, managed, useWeights);
        }

        // Segment shapes that exercise the hard part of the segment decoder: multiple segments of
        // different bit widths and different run lengths, transitions across 32-bit packed-word
        // boundaries, and degenerate corner cases (all-zero, all-same, single element, long runs).
        public static IEnumerable<object[]> SegmentShapeCases()
        {
            foreach (var shape in new[]
            {
                "AllZero", "AllSame", "Single", "Two", "IncreasingWidths",
                "DecreasingWidths", "AlternatingExtremes", "WordBoundaryWidths",
                "LongSingleWidth", "PowersOfTwo", "ManyShortSegments"
            })
                foreach (var useWeights in new[] { false, true })
                    foreach (var useIndices in new[] { false, true })
                        yield return new object[] { shape, useWeights, useIndices };
        }

        [Theory]
        [MemberData(nameof(SegmentShapeCases))]
        public void ManagedSegmentSumupHandlesVariedShapes(string shape, bool useWeights, bool useIndices)
        {
            var values = BuildSegmentValues(shape, out int numBins);
            var arr = CreateManagedSegment(values);
            var input = CreateInputForLength(values.Length, seed: 3, useWeights, useIndices,
                out double[] outputs, out double[] weights, out int[] docIndices, out int count);

            var managed = new FeatureHistogram(arr, numBins, useWeights);
            CallManaged(arr, input, managed);

            ComputeReference(arr, numBins, outputs, weights, docIndices, count,
                out double[] refTargets, out double[] refWeights, out int[] refCounts);
            AssertHistogramEqual(refCounts, refTargets, refWeights, managed, useWeights);

            // Where the native FastTree library is present (x64), the managed decoder must also be
            // bit-identical to the native one on these same varied-segment shapes.
            if (IntArray.UseFastTreeNative)
            {
                var native = new FeatureHistogram(arr, numBins, useWeights);
                CallNative(arr, input, native);
                AssertHistogramEqual(native.CountByBin, native.SumTargetsByBin, native.SumWeightsByBin, managed, useWeights);
            }
        }

        [Fact]
        public void SumupDispatchMatchesArchitecture()
        {
            // The whole point of the PR: x64/x86 must dispatch Sumup to the native handler and every
            // other architecture (e.g. arm64) to the new managed handler.
            bool expectNative = RuntimeInformation.ProcessArchitecture == Architecture.X64 ||
                                RuntimeInformation.ProcessArchitecture == Architecture.X86;
            Assert.Equal(expectNative, IntArray.UseFastTreeNative);

            // Verify the delegate actually wired into a real array matches that decision, not just the
            // UseFastTreeNative flag. A dense array is used because its constructor sets up the handler
            // without invoking the (separately tested) segment encoder.
            var values = new int[Length];
            var rand = new Random(7);
            for (int i = 0; i < Length; i++)
                values[i] = rand.Next(256);
            var dense = IntArray.New(Length, IntArrayType.Dense, IntArrayBits.Bits8, values);

            var handlerProp = typeof(IntArray).GetProperty("SumupHandler", BindingFlags.NonPublic | BindingFlags.Instance);
            var handler = (Delegate)handlerProp.GetValue(dense);
            Assert.NotNull(handler);
            Assert.Equal(expectNative ? "SumupNative" : "SumupManaged", handler.Method.Name);
        }

        // Builds a value array with the given segment "shape". Segments are runs of values that share
        // the same bit width, so varying the magnitude and run length of blocks yields multiple
        // segments of different widths/lengths.
        private static int[] BuildSegmentValues(string shape, out int numBins)
        {
            var values = new List<int>();
            var rand = new Random(101);

            // Appends 'count' values drawn from [0, maxExclusive), i.e. a run of a given bit width.
            void Block(int maxExclusive, int count)
            {
                for (int i = 0; i < count; i++)
                    values.Add(maxExclusive <= 1 ? 0 : rand.Next(maxExclusive));
            }

            switch (shape)
            {
                case "AllZero":
                    Block(1, 500);
                    break;
                case "AllSame":
                    for (int i = 0; i < 500; i++)
                        values.Add(42);
                    break;
                case "Single":
                    values.Add(123);
                    break;
                case "Two":
                    values.Add(1);
                    values.Add(200);
                    break;
                case "IncreasingWidths":
                    Block(2, 1); Block(16, 3); Block(256, 17); Block(4, 100); Block(64, 33); Block(1024, 7);
                    break;
                case "DecreasingWidths":
                    Block(1024, 7); Block(64, 33); Block(4, 100); Block(256, 17); Block(16, 3); Block(2, 1);
                    break;
                case "AlternatingExtremes":
                    for (int i = 0; i < 200; i++)
                    {
                        values.Add(0);
                        values.Add(1500 + (i % 100));
                    }
                    break;
                case "WordBoundaryWidths":
                    // Bit widths 3,5,7,3,1 with run lengths chosen so bit offsets cross 32-bit words.
                    Block(8, 5); Block(32, 7); Block(128, 11); Block(8, 13); Block(2, 64); Block(128, 3);
                    break;
                case "LongSingleWidth":
                    Block(256, 3000);
                    break;
                case "PowersOfTwo":
                    for (int p = 0; p <= 20; p++)
                        values.Add(1 << p);
                    break;
                case "ManyShortSegments":
                    for (int i = 0; i < 300; i++)
                        Block((i % 6) switch { 0 => 2, 1 => 16, 2 => 4, 3 => 256, 4 => 8, _ => 64 }, 1 + (i % 3));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(shape), shape, null);
            }

            int max = 0;
            foreach (int v in values)
            {
                if (v > max)
                    max = v;
            }
            numBins = max + 1;
            return values.ToArray();
        }

        private static IntArray CreateIntArray(string kind, int seed, out int numBins)
        {
            IntArrayBits bits;
            switch (kind)
            {
                case "Dense4": bits = IntArrayBits.Bits4; numBins = 16; break;
                case "Dense8": bits = IntArrayBits.Bits8; numBins = 256; break;
                case "Dense16": bits = IntArrayBits.Bits16; numBins = 2048; break;
                case "Dense32": bits = IntArrayBits.Bits32; numBins = 5000; break;
                case "Segment": bits = IntArrayBits.Bits8; numBins = 64; break;
                default: throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
            }

            var rand = new Random(seed);
            var values = new int[Length];
            for (int i = 0; i < Length; i++)
                values[i] = rand.Next(numBins);

            if (kind == "Segment")
                return CreateManagedSegment(values);

            return IntArray.New(Length, IntArrayType.Dense, bits, values);
        }

        // Builds a SegmentIntArray using the managed segment encoder explicitly. The public
        // IntArray.New(..., Segmented, ...) path would pick the native encoder on x64, and the native
        // C_SegmentFindOptimalPath declares its buffer as `unsigned long*`, which is 64-bit on LP64
        // (Linux/macOS x64) while the managed array is 32-bit — a pre-existing native buffer overrun
        // that is unrelated to the Sumup decoders under test here. Encoding managed-side avoids it while
        // still producing an array that both the managed and native Sumup decoders read identically.
        private static SegmentIntArray CreateManagedSegment(int[] values)
        {
            var work = new uint[values.Length];
            uint max = 0;
            for (int i = 0; i < values.Length; i++)
            {
                work[i] = (uint)values[i];
                if (work[i] > max)
                    max = work[i];
            }
            int maxBits = SegmentIntArray.BitsForValue(max);
            SegmentIntArray.ManagedSegmentFindOptimalPath(work, work.Length, maxBits, out long bits, out int transitions);
            return SegmentIntArray.FromWorkArray(work, work.Length, bits, transitions);
        }

        private static SumupInputData CreateInput(int seed, bool useWeights, bool useIndices,
            out double[] outputs, out double[] weights, out int[] docIndices, out int count)
        {
            return CreateInputForLength(Length, seed, useWeights, useIndices, out outputs, out weights, out docIndices, out count);
        }

        private static SumupInputData CreateInputForLength(int length, int seed, bool useWeights, bool useIndices,
            out double[] outputs, out double[] weights, out int[] docIndices, out int count)
        {
            var rand = new Random(seed);

            outputs = new double[length];
            for (int i = 0; i < length; i++)
                outputs[i] = rand.NextDouble() * 2 - 1;

            weights = null;
            if (useWeights)
            {
                weights = new double[length];
                for (int i = 0; i < length; i++)
                    weights[i] = rand.NextDouble();
            }

            docIndices = null;
            if (useIndices)
            {
                // Leaf case: a strictly increasing subset of document indices, as required by the
                // segment decoder (it walks segments forward assuming ascending indices).
                var list = new List<int>();
                for (int i = 0; i < length; i++)
                {
                    if (rand.Next(2) == 0)
                        list.Add(i);
                }
                docIndices = list.ToArray();
            }

            count = useIndices ? docIndices.Length : length;

            double sumTargets = 0;
            double sumWeights = 0;
            for (int i = 0; i < count; i++)
            {
                sumTargets += outputs[i];
                if (useWeights)
                    sumWeights += weights[i];
            }

            return new SumupInputData(count, sumTargets, sumWeights, outputs, weights, docIndices);
        }

        private static void ComputeReference(IntArray arr, int numBins, double[] outputs, double[] weights,
            int[] docIndices, int count, out double[] sumTargets, out double[] sumWeights, out int[] counts)
        {
            sumTargets = new double[numBins];
            sumWeights = weights == null ? null : new double[numBins];
            counts = new int[numBins];

            var indexer = arr.GetIndexer();
            for (int i = 0; i < count; i++)
            {
                int doc = docIndices == null ? i : docIndices[i];
                int bin = indexer[doc];
                sumTargets[bin] += outputs[i];
                if (sumWeights != null)
                    sumWeights[bin] += weights[i];
                counts[bin]++;
            }
        }

        private static void CallManaged(IntArray arr, SumupInputData input, FeatureHistogram histogram)
        {
            switch (arr)
            {
                case Dense4BitIntArray a: a.SumupManaged(input, histogram); break;
                case Dense8BitIntArray a: a.SumupManaged(input, histogram); break;
                case Dense16BitIntArray a: a.SumupManaged(input, histogram); break;
                case Dense32BitIntArray a: a.SumupManaged(input, histogram); break;
                case SegmentIntArray a: a.SumupManaged(input, histogram); break;
                default: throw new InvalidOperationException($"Unexpected IntArray type {arr.GetType().Name}");
            }
        }

        private static void CallNative(IntArray arr, SumupInputData input, FeatureHistogram histogram)
        {
            // For SegmentIntArray we call the native decoder (SumupCPlusPlus) directly rather than via
            // arr.Sumup: this array is built with the managed encoder through FromWorkArray, whose
            // constructor does not wire up SumupHandler, so arr.Sumup would NullReference. Dense arrays
            // set up their handler in their constructor, so arr.Sumup dispatches to the native handler.
            if (arr is SegmentIntArray seg)
                seg.SumupCPlusPlus(input, histogram);
            else
                arr.Sumup(input, histogram);
        }

        private static void AssertHistogramEqual(int[] expectedCounts, double[] expectedTargets, double[] expectedWeights,
            FeatureHistogram actual, bool useWeights)
        {
            for (int bin = 0; bin < expectedCounts.Length; bin++)
            {
                Assert.Equal(expectedCounts[bin], actual.CountByBin[bin]);
                // Accumulation order is mirrored between the implementations, so the sums are bit-identical.
                Assert.Equal(expectedTargets[bin], actual.SumTargetsByBin[bin]);
                if (useWeights)
                    Assert.Equal(expectedWeights[bin], actual.SumWeightsByBin[bin]);
            }
        }
    }
}
