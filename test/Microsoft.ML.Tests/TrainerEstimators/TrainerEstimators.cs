// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        public TrainerEstimators(ITestOutputHelper helper) : base(helper)
        {
        }

        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void PCATrainerEstimator()
        {
            string featureColumn = "NumericFeatures";

            var reader = new TextLoader(Env, new TextLoader.Options()
            {
                HasHeader = true,
                Separator = "\t",
                Columns = new[]
                {
                    new TextLoader.Column(featureColumn, DataKind.Single, new [] { new TextLoader.Range(1, 784) })
                },
                AllowSparse = true
            });
            var data = reader.Load(GetDataPath(TestDatasets.mnistOneClass.trainFilename));


            // Pipeline.
            var pipeline = new RandomizedPcaTrainer(Env, featureColumn, rank: 10);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// KMeans TrainerEstimator test 
        /// </summary>
        [Fact]
        public void KMeansEstimator()
        {
            string featureColumn = "NumericFeatures";
            string weights = "Weights";

            var reader = new TextLoader(Env, new TextLoader.Options
            {
                HasHeader = true,
                Separator = "\t",
                Columns = new[]
                {
                    new TextLoader.Column(featureColumn, DataKind.Single, new [] { new TextLoader.Range(1, 784) }),
                    new TextLoader.Column(weights, DataKind.Single, 0),
                },
                AllowSparse = true
            });
            var data = reader.Load(GetDataPath(TestDatasets.mnistTiny28.trainFilename));


            // Pipeline.
            var pipeline = new KMeansTrainer(Env, new KMeansTrainer.Options
            {
                FeatureColumnName = featureColumn,
                ExampleWeightColumnName = weights,
                InitializationAlgorithm = KMeansTrainer.InitializationAlgorithm.KMeansYinyang,
            });

            TestEstimatorCore(pipeline, data);

            Done();
        }

        /// <summary>
        /// HogwildSGD TrainerEstimator test (logistic regression).
        /// </summary>
        [Fact]
        public void TestEstimatorHogwildSGD()
        {
            var trainers = new[] { ML.BinaryClassification.Trainers.SgdCalibrated(l2Regularization: 0, numberOfIterations: 80),
                ML.BinaryClassification.Trainers.SgdCalibrated(new Trainers.SgdCalibratedTrainer.Options(){ L2Regularization = 0, NumberOfIterations = 80})};

            foreach (var trainer in trainers)
            {
                (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

                var pipeWithTrainer = pipe.AppendCacheCheckpoint(Env).Append(trainer);
                TestEstimatorCore(pipeWithTrainer, dataView);

                var transformedDataView = pipe.Fit(dataView).Transform(dataView);
                var model = trainer.Fit(transformedDataView);
                trainer.Fit(transformedDataView, model.Model.SubModel);
                TestEstimatorCore(pipe, dataView);

                var result = model.Transform(transformedDataView);
                var metrics = ML.BinaryClassification.Evaluate(result);

                Assert.InRange(metrics.Accuracy, 0.8, 1);
                Assert.InRange(metrics.AreaUnderRocCurve, 0.9, 1);
                Assert.InRange(metrics.LogLoss, 0, 0.6);
            }

            Done();
        }

        /// <summary>
        /// HogwildSGD TrainerEstimator test (support vector machine)
        /// </summary>
        [Fact]
        public void TestEstimatorHogwildSGDNonCalibrated()
        {
            var trainers = new[] { ML.BinaryClassification.Trainers.SgdNonCalibrated(loss : new SmoothedHingeLoss()),
                ML.BinaryClassification.Trainers.SgdNonCalibrated(new Trainers.SgdNonCalibratedTrainer.Options() { Loss = new HingeLoss() }) };

            foreach (var trainer in trainers)
            {
                (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();
                var pipeWithTrainer = pipe.AppendCacheCheckpoint(Env).Append(trainer);
                TestEstimatorCore(pipeWithTrainer, dataView);

                var transformedDataView = pipe.Fit(dataView).Transform(dataView);
                var model = trainer.Fit(transformedDataView);
                trainer.Fit(transformedDataView, model.Model);
                TestEstimatorCore(pipe, dataView);

                var result = model.Transform(transformedDataView);
                var metrics = ML.BinaryClassification.EvaluateNonCalibrated(result);

                Assert.InRange(metrics.Accuracy, 0.7, 1);
                Assert.InRange(metrics.AreaUnderRocCurve, 0.9, 1);
            }

            Done();
        }

        /// <summary>
        /// MulticlassNaiveBayes TrainerEstimator test 
        /// </summary>
        [Fact]
        public void TestEstimatorMulticlassNaiveBayesTrainer()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMulticlassPipeline();
            pipe = pipe.Append(ML.MulticlassClassification.Trainers.NaiveBayes("Label", "Features"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        private (IEstimator<ITransformer>, IDataView) GetBinaryClassificationPipeline()
        {
            var data = new TextLoader(Env,
                    new TextLoader.Options()
                    {
                        AllowQuoting = true,
                        Separator = "\t",
                        HasHeader = true,
                        Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Boolean, 0),
                            new TextLoader.Column("SentimentText", DataKind.String, 1)
                        }
                    }).Load(GetDataPath(TestDatasets.Sentiment.trainFilename));

            // Pipeline.
            var pipeline = new TextFeaturizingEstimator(Env, "Features", "SentimentText");

            return (pipeline, data);
        }


        private (IEstimator<ITransformer>, IDataView) GetRankingPipeline()
        {
            var data = new TextLoader(Env, new TextLoader.Options
            {
                HasHeader = true,
                Separator = "\t",
                Columns = new[]
                     {
                        new TextLoader.Column("Label", DataKind.Single, 0),
                        new TextLoader.Column("Workclass", DataKind.String, 1),
                        new TextLoader.Column("NumericFeatures", DataKind.Single, new [] { new TextLoader.Range(9, 14) })
                    }
            }).Load(GetDataPath(TestDatasets.adultRanking.trainFilename));

            // Pipeline.
            var pipeline = new ValueToKeyMappingEstimator(Env, new[]{
                                    new ValueToKeyMappingEstimator.ColumnOptions("Group", "Workclass"),
                                    new ValueToKeyMappingEstimator.ColumnOptions("Label0", "Label") });

            return (pipeline, data);
        }

        private IDataView GetRegressionPipeline()
        {
            return new TextLoader(Env,
                    new TextLoader.Options()
                    {
                        Separator = ";",
                        HasHeader = true,
                        Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 11),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(0, 10) } )
                        }
                    }).Load(GetDataPath(TestDatasets.generatedRegressionDatasetmacro.trainFilename));
        }

        private TextLoader.Options GetIrisLoaderArgs()
        {
            return new TextLoader.Options()
            {
                Separator = "comma",
                HasHeader = true,
                Columns = new[]
                        {
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(0, 3) }),
                            new TextLoader.Column("Label", DataKind.String, 4)
                        }
            };
        }

        private (IEstimator<ITransformer>, IDataView) GetMulticlassPipeline()
        {
            var data = new TextLoader(Env, new TextLoader.Options()
            {
                Separator = "comma",
                Columns = new[]
                        {
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(0, 3) }),
                            new TextLoader.Column("Label", DataKind.String, 4)
                        }
            }).Load(GetDataPath(IrisDataPath));

            var pipeline = new ValueToKeyMappingEstimator(Env, "Label");

            return (pipeline, data);
        }
    }
}
