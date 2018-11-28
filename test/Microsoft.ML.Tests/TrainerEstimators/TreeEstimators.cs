// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Conversions;
using System;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeBinaryEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = new FastTreeBinaryClassificationTrainer(Env, "Label", "Features", numTrees: 10, numLeaves: 5, advancedSettings: s =>
            {
                s.NumThreads = 1;
            });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMBinaryEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = new LightGbmBinaryTrainer(Env, "Label", "Features", advancedSettings: s =>
            {
                s.NumLeaves = 10;
                s.NThread = 1;
                s.MinDataPerLeaf = 2;
            });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }


        [Fact]
        public void GAMClassificationEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = new BinaryClassificationGamTrainer(Env, "Label", "Features", advancedSettings: s =>
            {
                s.GainConfidenceLevel = 0;
                s.NumIterations = 15;
            });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }


        [Fact]
        public void FastForestClassificationEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = new FastForestClassification(Env, "Label", "Features", advancedSettings: s =>
            {
                s.NumLeaves = 10;
                s.NumTrees = 20;
            });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// FastTreeRankingTrainer TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRankerEstimator()
        {
            var (pipe, dataView) = GetRankingPipeline();

            var trainer = new FastTreeRankingTrainer(Env, "Label0", "NumericFeatures", "Group",
                                advancedSettings: s => { s.NumTrees = 10; });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// LightGbmRankingTrainer TrainerEstimator test 
        /// </summary>
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMRankerEstimator()
        {
            var (pipe, dataView) = GetRankingPipeline();

            var trainer = new LightGbmRankingTrainer(Env, "Label0", "NumericFeatures", "Group",
                                advancedSettings: s => { s.LearningRate = 0.4; });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new FastTreeRegressionTrainer(Env, "Label", "Features", advancedSettings: s =>
            {
                s.NumTrees = 10;
                s.NumThreads = 1;
                s.NumLeaves = 5;
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmRegressorTrainer TrainerEstimator test 
        /// </summary>
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new LightGbmRegressorTrainer(Env, "Label", "Features", advancedSettings: s =>
            {
                s.NThread = 1;
                s.NormalizeFeatures = NormalizeOption.Warn;
                s.CatL2 = 5;
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }


        /// <summary>
        /// RegressionGamTrainer TrainerEstimator test 
        /// </summary>
        [Fact]
        public void GAMRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new RegressionGamTrainer(Env, "Label", "Features", advancedSettings: s =>
            {
                s.EnablePruning = false;
                s.NumIterations = 15;
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// FastTreeTweedieTrainer TrainerEstimator test 
        /// </summary>
        [Fact]
        public void TweedieRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new FastTreeTweedieTrainer(Env, "Label", "Features", advancedSettings: s =>
            {
                s.EntropyCoefficient = 0.3;
                s.OptimizationAlgorithm = BoostedTreeArgs.OptimizationAlgorithmType.AcceleratedGradientDescent;
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// FastForestRegression TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastForestRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new FastForestRegression(Env, "Label", "Features", advancedSettings: s =>
            {
                s.BaggingSize = 2;
                s.NumTrees = 10;
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmMulticlass TrainerEstimator test 
        /// </summary>
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmMultiClassEstimator()
        {
            var (pipeline, dataView) = GetMultiClassPipeline();
            var trainer = new LightGbmMulticlassTrainer(Env, "Label", "Features", advancedSettings: s => { s.LearningRate = 0.4; });
            var pipe = pipeline.Append(trainer)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
