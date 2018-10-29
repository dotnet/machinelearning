// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms.CategoricalTransforms;
using System;
using System.Linq;
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
            var (pipeline, data) = GetBinaryClassificationPipeline();

            pipeline.Append(new FastTreeBinaryClassificationTrainer(Env, "Label", "Features", numTrees:10, numLeaves:5,  advancedSettings: s => {
                    s.NumThreads = 1;
                }));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [Fact]
        public void LightGBMBinaryEstimator()
        {
            var (pipeline, data) = GetBinaryClassificationPipeline();

            pipeline.Append(new LightGbmBinaryTrainer(Env, "Label", "Features", advancedSettings: s => {
                s.NumLeaves = 10;
                s.NThread = 1;
                s.MinDataPerLeaf = 2;
            }));

            TestEstimatorCore(pipeline, data);
            Done();
        }


        [Fact]
        public void GAMClassificationEstimator()
        {
            var (pipeline, data) = GetBinaryClassificationPipeline();

            pipeline.Append(new BinaryClassificationGamTrainer(Env, "Label", "Features", advancedSettings: s => {
                s.GainConfidenceLevel = 0;
                s.NumIterations = 15;
            }));

            TestEstimatorCore(pipeline, data);
            Done();
        }


        [Fact]
        public void FastForestClassificationEstimator()
        {
            var (pipeline, data) = GetBinaryClassificationPipeline();

            pipeline.Append(new FastForestClassification(Env, "Label", "Features", advancedSettings: s => {
                s.NumLeaves = 10;
                s.NumTrees = 20;
            }));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRankerEstimator()
        {
            var (pipeline, data) = GetRankingPipeline();

            pipeline.Append(new FastTreeRankingTrainer(Env, "Label0", "NumericFeatures", "Group", 
                                advancedSettings: s => { s.NumTrees = 10; }));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void LightGBMRankerEstimator()
        {
            var (pipeline, data) = GetRankingPipeline();

            pipeline.Append(new LightGbmRankingTrainer(Env, "Label0", "NumericFeatures", "Group",
                                advancedSettings: s => { s.LearningRate = 0.4; }));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRegressorEstimator()
        {

            // Pipeline.
            var pipeline = new FastTreeRegressionTrainer(Env, "Label", "Features", advancedSettings: s => {
                    s.NumTrees = 10;
                    s.NumThreads = 1;
                    s.NumLeaves = 5;
                });

            TestEstimatorCore(pipeline, GetRegressionPipeline());
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMRegressorEstimator()
        {

            // Pipeline.
            var pipeline = new LightGbmRegressorTrainer(Env, "Label", "Features", advancedSettings: s => {
                s.NThread = 1;
                s.NormalizeFeatures = NormalizeOption.Warn;
                s.CatL2 = 5; 
            });

            TestEstimatorCore(pipeline, GetRegressionPipeline());
            Done();
        }


        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void GAMRegressorEstimator()
        {

            // Pipeline.
            var pipeline = new RegressionGamTrainer(Env, "Label", "Features", advancedSettings: s => {
                s.EnablePruning = false;
                s.NumIterations = 15;
            });

            TestEstimatorCore(pipeline, GetRegressionPipeline());
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void TweedieRegressorEstimator()
        {

            // Pipeline.
            var pipeline = new FastTreeTweedieTrainer(Env, "Label", "Features", advancedSettings: s => {
                s.EntropyCoefficient = 0.3;
                s.OptimizationAlgorithm = BoostedTreeArgs.OptimizationAlgorithmType.AcceleratedGradientDescent;
            });

            TestEstimatorCore(pipeline, GetRegressionPipeline());
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastForestRegressorEstimator()
        {

            // Pipeline.
            var pipeline = new FastForestRegression(Env, "Label", "Features", advancedSettings: s => {
                s.BaggingSize = 2;
                s.NumTrees = 10;
            });

            TestEstimatorCore(pipeline, GetRegressionPipeline());
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void LightGbmMultiClassEstimator()
        {
            var (pipeline, data) = GetMultiClassPipeline();

            pipeline.Append(new LightGbmMulticlassTrainer(Env, "Label", "Features", advancedSettings: s => { s.LearningRate = 0.4; }))
                    .Append(new KeyToValueEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
