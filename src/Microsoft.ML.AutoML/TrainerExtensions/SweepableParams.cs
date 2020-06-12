// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.AutoML
{
    internal static class SweepableParams
    {
        private static IEnumerable<SweepableParam> BuildAveragedLinearArgsParams()
        {
            return new SweepableParam[]
            {
                new SweepableDiscreteParam("LearningRate", new object[] { 0.01f, 0.1f, 0.5f, 1.0f}),
                new SweepableDiscreteParam("DecreaseLearningRate", new object[] { false, true }),
                new SweepableFloatParam("L2Regularization", 0.0f, 0.4f),
            };
        }

        private static IEnumerable<SweepableParam> BuildOnlineLinearArgsParams()
        {
            return new SweepableParam[]
            {
                new SweepableLongParam("NumberOfIterations", 1, 100, stepSize: 10, isLogScale: true),
                new SweepableFloatParam("InitialWeightsDiameter", 0.0f, 1.0f, numSteps: 5),
                new SweepableDiscreteParam("Shuffle", new object[] { false, true }),
            };
        }

        private static IEnumerable<SweepableParam> BuildTreeArgsParams()
        {
            return new SweepableParam[]
            {
                new SweepableLongParam("NumberOfLeaves", 2, 128, isLogScale: true, stepSize: 4),
                new SweepableDiscreteParam("MinimumExampleCountPerLeaf", new object[] { 1, 10, 50 }),
                new SweepableDiscreteParam("NumberOfTrees", new object[] { 20, 100, 500 }),
            };
        }

        private static IEnumerable<SweepableParam> BuildBoostedTreeArgsParams()
        {
            return BuildTreeArgsParams().Concat(new List<SweepableParam>()
            {
                new SweepableFloatParam("LearningRate", 0.025f, 0.4f, isLogScale: true),
                new SweepableFloatParam("Shrinkage", 0.025f, 4f, isLogScale: true),
            });
        }

        private static IEnumerable<SweepableParam> BuildLbfgsArgsParams()
        {
            return new SweepableParam[] {
                new SweepableFloatParam("L2Regularization", 0.0f, 1.0f, numSteps: 4),
                new SweepableFloatParam("L1Regularization", 0.0f, 1.0f, numSteps: 4),
                new SweepableDiscreteParam("OptimizationTolerance", new object[] { 1e-4f, 1e-7f }),
                new SweepableDiscreteParam("HistorySize", new object[] { 5, 20, 50 }),
                new SweepableLongParam("MaximumNumberOfIterations", 1, int.MaxValue),
                new SweepableFloatParam("InitialWeightsDiameter", 0.0f, 1.0f, numSteps: 5),
                new SweepableDiscreteParam("DenseOptimizer", new object[] { false, true }),
            };
        }

        /// <summary>
        /// The names of every hyperparameter swept across all trainers.
        /// </summary>
        public static ISet<string> AllHyperparameterNames = GetAllSweepableParameterNames();

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
            return BuildBoostedTreeArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildFastTreeTweedieParams()
        {
            return BuildBoostedTreeArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildLightGbmParamsMulticlass()
        {
            return BuildLightGbmParams().Union(new SweepableParam[]
            {
                new SweepableDiscreteParam("UseSoftmax", new object[] { true, false }),
            });
        }

        public static IEnumerable<SweepableParam> BuildLightGbmParams()
        {
            return new SweepableParam[]
            {
                new SweepableDiscreteParam("NumberOfIterations", new object[] { 10, 20, 50, 100, 150, 200 }),
                new SweepableFloatParam("LearningRate", 0.025f, 0.4f, isLogScale: true),
                new SweepableLongParam("NumberOfLeaves", 2, 128, isLogScale: true, stepSize: 4),
                new SweepableDiscreteParam("MinimumExampleCountPerLeaf", new object[] { 1, 10, 20, 50 }),
                new SweepableDiscreteParam("UseCategoricalSplit", new object[] { true, false }),
                new SweepableDiscreteParam("HandleMissingValue", new object[] { true, false }),
                new SweepableDiscreteParam("UseZeroAsMissingValue", new object[] { true, false }),
                new SweepableDiscreteParam("MinimumExampleCountPerGroup", new object[] { 10, 50, 100, 200 }),
                new SweepableDiscreteParam("MaximumCategoricalSplitPointCount", new object[] { 8, 16, 32, 64 }),
                new SweepableDiscreteParam("CategoricalSmoothing", new object[] { 1, 10, 20 }),
                new SweepableDiscreteParam("L2CategoricalRegularization", new object[] { 0.1, 0.5, 1, 5, 10 }),

                // Booster params
                new SweepableDiscreteParam("L2Regularization", new object[] { 0f, 0.5f, 1f }),
                new SweepableDiscreteParam("L1Regularization", new object[] { 0f, 0.5f, 1f })
            };
        }

        public static IEnumerable<SweepableParam> BuildMatrixFactorizationParams()
        {
            return new SweepableParam[]
            {
                new SweepableDiscreteParam(nameof(MatrixFactorizationTrainer.Options.NumberOfIterations), new object[] { 10, 20, 40 }),
                new SweepableDiscreteParam(nameof(MatrixFactorizationTrainer.Options.LearningRate), new object[] { 0.001f, 0.01f, 0.1f }),
                new SweepableDiscreteParam(nameof(MatrixFactorizationTrainer.Options.ApproximationRank), new object[] { 8, 16, 64, 128 }),
                new SweepableDiscreteParam(nameof(MatrixFactorizationTrainer.Options.Lambda), new object[] { 0.01f, 0.05f, 0.1f, 0.5f, 1f }),
                new SweepableDiscreteParam(nameof(MatrixFactorizationTrainer.Options.LossFunction), new object[] { MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression, MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass }),
                new SweepableDiscreteParam(nameof(MatrixFactorizationTrainer.Options.Alpha), new object[] { 1f, 0.01f, 0.0001f, 0.000001f }),
                new SweepableDiscreteParam(nameof(MatrixFactorizationTrainer.Options.C), new object[] { 0.000001f, 0.0001f, 0.01f }),
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

        public static IEnumerable<SweepableParam> BuildLbfgsLogisticRegressionParams()
        {
            return BuildLbfgsArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildOnlineGradientDescentParams()
        {
            return BuildAveragedLinearArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildLbfgsPoissonRegressionParams()
        {
            return BuildLbfgsArgsParams();
        }

        public static IEnumerable<SweepableParam> BuildSdcaParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("L2Regularization", new object[] { "<Auto>", 1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f }),
                new SweepableDiscreteParam("L1Regularization", new object[] { "<Auto>", 0f, 0.25f, 0.5f, 0.75f, 1f }),
                new SweepableDiscreteParam("ConvergenceTolerance", new object[] { 0.001f, 0.01f, 0.1f, 0.2f }),
                new SweepableDiscreteParam("MaximumNumberOfIterations", new object[] { "<Auto>", 10, 20, 100 }),
                new SweepableDiscreteParam("Shuffle", null, isBool: true),
                new SweepableDiscreteParam("BiasLearningRate", new object[] { 0.0f, 0.01f, 0.1f, 1f })
            };
        }

        public static IEnumerable<SweepableParam> BuildOlsParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("L2Regularization", new object[] { 1e-6f, 0.1f, 1f })
            };
        }

        public static IEnumerable<SweepableParam> BuildSgdParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("L2Regularization", new object[] { 1e-7f, 5e-7f, 1e-6f, 5e-6f, 1e-5f }),
                new SweepableDiscreteParam("ConvergenceTolerance", new object[] { 1e-2f, 1e-3f, 1e-4f, 1e-5f }),
                new SweepableDiscreteParam("NumberOfIterations", new object[] { 1, 5, 10, 20 }),
                new SweepableDiscreteParam("Shuffle", null, isBool: true),
            };
        }

        public static IEnumerable<SweepableParam> BuildSymSgdLogisticRegressionParams()
        {
            return new SweepableParam[] {
                new SweepableDiscreteParam("NumberOfIterations", new object[] { 1, 5, 10, 20, 30, 40, 50 }),
                new SweepableDiscreteParam("LearningRate", new object[] { "<Auto>", 1e1f, 1e0f, 1e-1f, 1e-2f, 1e-3f }),
                new SweepableDiscreteParam("L2Regularization", new object[] { 0.0f, 1e-5f, 1e-5f, 1e-6f, 1e-7f }),
                new SweepableDiscreteParam("UpdateFrequency", new object[] { "<Auto>", 5, 20 })
            };
        }

        /// <summary>
        /// Gets the name of every hyperparameter swept across all trainers.
        /// </summary>
        public static ISet<string> GetAllSweepableParameterNames()
        {
            var sweepableParams = new List<SweepableParam>();
            sweepableParams.AddRange(BuildAveragePerceptronParams());
            sweepableParams.AddRange(BuildAveragePerceptronParams());
            sweepableParams.AddRange(BuildFastForestParams());
            sweepableParams.AddRange(BuildFastTreeParams());
            sweepableParams.AddRange(BuildFastTreeTweedieParams());
            sweepableParams.AddRange(BuildLightGbmParamsMulticlass());
            sweepableParams.AddRange(BuildLightGbmParams());
            sweepableParams.AddRange(BuildLinearSvmParams());
            sweepableParams.AddRange(BuildLbfgsLogisticRegressionParams());
            sweepableParams.AddRange(BuildOnlineGradientDescentParams());
            sweepableParams.AddRange(BuildLbfgsPoissonRegressionParams());
            sweepableParams.AddRange(BuildSdcaParams());
            sweepableParams.AddRange(BuildOlsParams());
            sweepableParams.AddRange(BuildSgdParams());
            sweepableParams.AddRange(BuildSymSgdLogisticRegressionParams());
            return new HashSet<string>(sweepableParams.Select(p => p.Name));
        }
    }
}
