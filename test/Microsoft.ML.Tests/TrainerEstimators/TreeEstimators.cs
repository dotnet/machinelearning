// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.RunTests;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TreeEstimators : TestDataPipeBase
    {

        public TreeEstimators(ITestOutputHelper output) : base(output)
        {
        }


        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeBinaryEstimator()
        {
            var (pipeline, data) = GetBinaryClassificationPipeline();

            pipeline.Append(new FastTreeBinaryClassificationTrainer(Env, "Label", "Features", advancedSettings: s => {
                    s.NumTrees = 10;
                    s.NumThreads = 1;
                    s.NumLeaves = 5;
                }));

            TestEstimatorCore(pipeline, data);
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
        }

        private (IEstimator<ITransformer>, IDataView) GetBinaryClassificationPipeline()
        {
            var data = new TextLoader(Env,
                    new TextLoader.Arguments()
                    {
                        Separator = "\t",
                        HasHeader = true,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("SentimentText", DataKind.Text, 1)
                        }
                    }).Read(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

            // Pipeline.
            var pipeline = new TextTransform(Env, "SentimentText", "Features");

            return (pipeline, data);
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
        }

        private (IEstimator<ITransformer>, IDataView) GetRankingPipeline()
        {
            var data = new TextLoader(Env, new TextLoader.Arguments
            {
                HasHeader = true,
                Separator = "\t",
                Column = new[]
                     {
                        new TextLoader.Column("Label", DataKind.R4, 0),
                        new TextLoader.Column("Workclass", DataKind.Text, 1),
                        new TextLoader.Column("NumericFeatures", DataKind.R4, new [] { new TextLoader.Range(9, 14) })
                    }
            }).Read(new MultiFileSource(GetDataPath(TestDatasets.adultRanking.trainFilename)));

            // Pipeline.
            var pipeline = new TermEstimator(Env, new[]{
                                    new TermTransform.ColumnInfo("Workclass", "Group"),
                                    new TermTransform.ColumnInfo("Label", "Label0") });

            return (pipeline, data);
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

            TestEstimatorCore(pipeline, GetRegressionData());
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void LightGBMRegressorEstimator()
        {

            // Pipeline.
            var pipeline = new LightGbmRegressorTrainer(Env, "Label", "Features", advancedSettings: s => {
                s.NThread = 1;
                s.NormalizeFeatures = NormalizeOption.Warn;
                s.CatL2 = 5; 
            });

            TestEstimatorCore(pipeline, GetRegressionData());
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

            TestEstimatorCore(pipeline, GetRegressionData());
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

            TestEstimatorCore(pipeline, GetRegressionData());
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

            TestEstimatorCore(pipeline, GetRegressionData());
        }

        private IDataView GetRegressionData()
        {
            return new TextLoader(Env,
                    new TextLoader.Arguments()
                    {
                        Separator = ";",
                        HasHeader = true,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 11),
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(0, 10) } )
                        }
                    }).Read(new MultiFileSource(GetDataPath(TestDatasets.generatedRegressionDatasetmacro.trainFilename)));
        }
    }
}
