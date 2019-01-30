// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal static class SweepableParams
    {
        private static IEnumerable<SweepableParam> BuildAveragedLinearArgsParams()
        {
            return new SweepableParam[]
            {
                new SweepableDiscreteParam("LearningRate", new object[] { 0.01, 0.1, 0.5, 1.0 }),
                new SweepableDiscreteParam("DecreaseLearningRate", new object[] { false, true }),
                new SweepableFloatParam("L2RegularizerWeight", 0.0f, 0.4f),
            };
        }

        private static IEnumerable<SweepableParam> BuildOnlineLinearArgsParams()
        {
            return new SweepableParam[]
            {
                new SweepableLongParam("NumIterations", 1, 100, stepSize: 10, isLogScale: true),
                new SweepableFloatParam("InitWtsDiameter", 0.0f, 1.0f, numSteps: 5),
                new SweepableDiscreteParam("Shuffle", new object[] { false, true }),
            };
        }

        private static IEnumerable<SweepableParam> BuildTreeArgsParams()
        {
            return new SweepableParam[]
            {
                new SweepableLongParam("NumLeaves", 2, 128, isLogScale: true, stepSize: 4),
                new SweepableDiscreteParam("MinDocumentsInLeafs", new object[] { 1, 10, 50 }),
                new SweepableDiscreteParam("NumTrees", new object[] { 20, 100, 500 }),
                new SweepableFloatParam("LearningRates", 0.025f, 0.4f, isLogScale: true),
                new SweepableFloatParam("Shrinkage", 0.025f, 4f, isLogScale: true),
            };
        }

        private static IEnumerable<SweepableParam> BuildLbfgsArgsParams()
        {
            return new SweepableParam[] {
                new SweepableFloatParam("L2Weight", 0.0f, 1.0f, numSteps: 4),
                new SweepableFloatParam("L1Weight", 0.0f, 1.0f, numSteps: 4),
                new SweepableDiscreteParam("OptTol", new object[] { 1e-4f, 1e-7f }),
                new SweepableDiscreteParam("MemorySize", new object[] { 5, 20, 50 }),
                new SweepableLongParam("MaxIterations", 1, int.MaxValue),
                new SweepableFloatParam("InitWtsDiameter", 0.0f, 1.0f, numSteps: 5),
                new SweepableDiscreteParam("DenseOptimizer", new object[] { false, true }),
            };
        }

        public static IEnumerable<SweepableParam> BuildAveragePerceptronParams()
        {
            return BuildAveragedLinearArgsParams().Concat(BuildOnlineLinearArgsParams());
        }

        public static IEnumerable<SweepableParam> BuildFastForestParams()
        {
            return BuildTreeArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildFastTreeParams()
        {
            return BuildTreeArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildFastTreeTweedieParams()
        {
            return BuildTreeArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildLightGbmParams()
        {
            return new SweepableParam[]
            {
                new SweepableDiscreteParam("NumBoostRound", new object[] { 10, 20, 50, 100, 150, 200 }),
                new SweepableFloatParam("LearningRate", 0.025f, 0.4f, isLogScale: true),
                new SweepableLongParam("NumLeaves", 2, 128, isLogScale: true, stepSize: 4),
                new SweepableDiscreteParam("MinDataPerLeaf", new object[] { 1, 10, 20, 50 }),
                new SweepableDiscreteParam("UseSoftmax", new object[] { true, false }),
                new SweepableDiscreteParam("UseCat", new object[] { true, false }),
                new SweepableDiscreteParam("UseMissing", new object[] { true, false }),
                new SweepableDiscreteParam("MinDataPerGroup", new object[] { 10, 50, 100, 200 }),
                new SweepableDiscreteParam("MaxCatThreshold", new object[] { 8, 16, 32, 64 }),
                new SweepableDiscreteParam("CatSmooth", new object[] { 1, 10, 20 }),
                new SweepableDiscreteParam("CatL2", new object[] { 0.1, 0.5, 1, 5, 10 }),

                // TreeBooster params
                new SweepableDiscreteParam("RegLambda", new object[] { 0f, 0.5f, 1f }),
                new SweepableDiscreteParam("RegAlpha", new object[] { 0f, 0.5f, 1f })
            };
        }

        public static IEnumerable<SweepableParam> BuildLinearSvmParams()
        {
            return new SweepableParam[] {
                new SweepableFloatParam("Lambda", 0.00001f, 0.1f, 10, isLogScale: true),
                new SweepableDiscreteParam("PerformProjection", null, isBool: true),
                new SweepableDiscreteParam("NoBias", null, isBool: true)
            }.Concat(BuildOnlineLinearArgsParams());
        }

        public static IEnumerable<SweepableParam> BuildLogisticRegressionParams()
        {
            return BuildLbfgsArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildOnlineGradientDescentParams()
        {
            return BuildAveragedLinearArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildPoissonRegressionParams()
        {
            return BuildLbfgsArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildSdcaParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("L2Const", new object[] { null, 1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f }),
                new SweepableDiscreteParam("L1Threshold", new object[] { null, 0f, 0.25f, 0.5f, 0.75f, 1f }),
                new SweepableDiscreteParam("ConvergenceTolerance", new object[] { 0.001f, 0.01f, 0.1f, 0.2f }),
                new SweepableDiscreteParam("MaxIterations", new object[] { null, 10, 20, 100 }),
                new SweepableDiscreteParam("Shuffle", null, isBool: true),
                new SweepableDiscreteParam("BiasLearningRate", new object[] { 0.0f, 0.01f, 0.1f, 1f })
            };
        }

        public static IEnumerable<SweepableParam> BuildOrdinaryLeastSquaresParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("L2Weight", new object[] { 1e-6f, 0.1f, 1f })
            };
        }

        public static IEnumerable<SweepableParam> BuildSgdParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("L2Weight", new object[] { 1e-7f, 5e-7f, 1e-6f, 5e-6f, 1e-5f }),
                new SweepableDiscreteParam("ConvergenceTolerance", new object[] { 1e-2f, 1e-3f, 1e-4f, 1e-5f }),
                new SweepableDiscreteParam("MaxIterations", new object[] { 1, 5, 10, 20 }),
                new SweepableDiscreteParam("Shuffle", null, isBool: true),
            };
        }

        public static IEnumerable<SweepableParam> BuildSymSgdParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("NumberOfIterations", new object[] { 1, 5, 10, 20, 30, 40, 50 }),
                new SweepableDiscreteParam("LearningRate", new object[] { null, 1e1f, 1e0f, 1e-1f, 1e-2f, 1e-3f }),
                new SweepableDiscreteParam("L2Regularization", new object[] { 0.0f, 1e-5f, 1e-5f, 1e-6f, 1e-7f }),
                new SweepableDiscreteParam("UpdateFrequency", new object[] { null, 5, 20 })
            };
        }
    }
}
