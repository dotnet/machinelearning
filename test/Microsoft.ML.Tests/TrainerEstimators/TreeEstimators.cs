// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
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

            var trainer = ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfTrees = 10,
                    NumberOfLeaves = 5,
                });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);
            Done();
        }

        [LightGBMFact]
        public void LightGBMBinaryEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            // Attention: Do not set NumberOfThreads here, left this to use default value to avoid test crash.
            // Details can be found here: https://github.com/dotnet/machinelearning/pull/4918
            var trainer = ML.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
            {
                NumberOfLeaves = 10,
                MinimumExampleCountPerLeaf = 2,
                UnbalancedSets = false, // default value
            });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);
            Done();
        }

        [LightGBMFact]
        public void LightGBMBinaryEstimatorUnbalanced()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            // Attention: Do not set NumberOfThreads here, left this to use default value to avoid test crash.
            // Details can be found here: https://github.com/dotnet/machinelearning/pull/4918
            var trainer = ML.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
            {
                NumberOfLeaves = 10,
                MinimumExampleCountPerLeaf = 2,
                UnbalancedSets = true,
            });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// LightGBMBinaryTrainer CorrectSigmoid test
        /// </summary>
        [LightGBMFact]
        public void LightGBMBinaryEstimatorCorrectSigmoid()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();
            var sigmoid = .789;

            // Attention: Do not set NumberOfThreads here, left this to use default value to avoid test crash.
            // Details can be found here: https://github.com/dotnet/machinelearning/pull/4918
            var trainer = ML.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
            {
                NumberOfLeaves = 10,
                MinimumExampleCountPerLeaf = 2,
                Sigmoid = sigmoid
            });

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);

            // The slope in the model calibrator should be equal to the negative of the sigmoid passed into the trainer.
            Assert.Equal(sigmoid, -model.Model.Calibrator.Slope);
            Done();
        }


        [Fact]
        public void GAMClassificationEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = new GamBinaryTrainer(Env, new GamBinaryTrainer.Options
            {
                GainConfidenceLevel = 0,
                NumberOfIterations = 15,
            });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);
            Done();
        }


        [Fact]
        public void FastForestClassificationEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = ML.BinaryClassification.Trainers.FastForest(
                new FastForestBinaryTrainer.Options
                {
                    NumberOfLeaves = 10,
                    NumberOfTrees = 20,
                });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// FastTreeRankingTrainer TrainerEstimator test
        /// </summary>
        [Fact]
        public void FastTreeRankerEstimator()
        {
            var (pipe, dataView) = GetRankingPipeline();

            var trainer = ML.Ranking.Trainers.FastTree(
                new FastTreeRankingTrainer.Options
                {
                    FeatureColumnName = "NumericFeatures",
                    NumberOfTrees = 10,
                    RowGroupColumnName = "Group"
                });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// LightGbmRankingTrainer TrainerEstimator test
        /// </summary>
        [LightGBMFact]
        public void LightGBMRankerEstimator()
        {
            var (pipe, dataView) = GetRankingPipeline();

            var trainer = ML.Ranking.Trainers.LightGbm(new LightGbmRankingTrainer.Options() { LabelColumnName = "Label0", FeatureColumnName = "NumericFeatures", RowGroupColumnName = "Group", LearningRate = 0.4 });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test
        /// </summary>
        [Fact]
        public void FastTreeRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastTree(
                new FastTreeRegressionTrainer.Options { NumberOfTrees = 10, NumberOfThreads = 1, NumberOfLeaves = 5 });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Fit(dataView, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmRegressionTrainer TrainerEstimator test
        /// </summary>
        [LightGBMFact]
        public void LightGBMRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();

            // Attention: Do not set NumberOfThreads here, left this to use default value to avoid test crash.
            // Details can be found here: https://github.com/dotnet/machinelearning/pull/4918
            var trainer = ML.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                NormalizeFeatures = NormalizeOption.Warn,
                L2CategoricalRegularization = 5,
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Fit(dataView, dataView);
            Done();
        }


        /// <summary>
        /// RegressionGamTrainer TrainerEstimator test
        /// </summary>
        [Fact]
        public void GAMRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new GamRegressionTrainer(Env, new GamRegressionTrainer.Options
            {
                EnablePruning = false,
                NumberOfIterations = 15,
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Fit(dataView, dataView);
            Done();
        }

        /// <summary>
        /// FastTreeTweedieTrainer TrainerEstimator test
        /// </summary>
        [Fact]
        public void TweedieRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastTreeTweedie(
                new FastTreeTweedieTrainer.Options
                {
                    EntropyCoefficient = 0.3,
                    OptimizationAlgorithm = BoostedTreeOptions.OptimizationAlgorithmType.AcceleratedGradientDescent,
                });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Fit(dataView, dataView);
            Done();
        }

        /// <summary>
        /// FastForestRegression TrainerEstimator test
        /// </summary>
        [Fact]
        public void FastForestRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastForest(
                new FastForestRegressionTrainer.Options
                {
                    BaggingSize = 2,
                    NumberOfTrees = 10,
                });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Fit(dataView, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmMulticlass TrainerEstimator test
        /// </summary>
        [LightGBMFact]
        public void LightGbmMulticlassEstimator()
        {
            var (pipeline, dataView) = GetMulticlassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LightGbm(learningRate: 0.4);
            var pipe = pipeline.Append(trainer)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmMulticlass TrainerEstimator test with options
        /// </summary>
        [LightGBMTheory]
        [InlineData(true)]
        [InlineData(false)]
        public void LightGbmMulticlassEstimatorWithOptions(bool softMax)
        {
            var options = new LightGbmMulticlassTrainer.Options
            {
                EvaluationMetric = LightGbmMulticlassTrainer.Options.EvaluateMetricType.Default,
                UseSoftmax = softMax
            };

            var (pipeline, dataView) = GetMulticlassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LightGbm(options);
            var pipe = pipeline.Append(trainer)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmMulticlass CorrectSigmoid test
        /// </summary>
        [LightGBMFact]
        public void LightGbmMulticlassEstimatorCorrectSigmoid()
        {
            var (pipeline, dataView) = GetMulticlassPipeline();
            var sigmoid = .789;

            var trainer = ML.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                Sigmoid = sigmoid
            });

            var pipe = pipeline.Append(trainer)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView, transformedDataView);

            // The slope in the all the calibrators should be equal to the negative of the sigmoid passed into the trainer.
            Assert.True(model.Model.SubModelParameters.All(predictor =>
            ((FeatureWeightsCalibratedModelParameters<LightGbmBinaryModelParameters, PlattCalibrator>)predictor).Calibrator.Slope == -sigmoid));
            Done();
        }

        /// <summary>
        /// LightGbmMulticlass Test of Balanced Data
        /// </summary>
        [LightGBMFact]
        public void LightGbmMulticlassEstimatorBalanced()
        {
            var (pipeline, dataView) = GetMulticlassPipeline();

            var trainer = ML.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                UnbalancedSets = false
            });

            var pipe = pipeline.Append(trainer)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmMulticlass Test of Unbalanced Data
        /// </summary>
        [LightGBMFact]
        public void LightGbmMulticlassEstimatorUnbalanced()
        {
            var (pipeline, dataView) = GetMulticlassPipeline();

            var trainer = ML.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                UnbalancedSets = true
            });

            var pipe = pipeline.Append(trainer)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        // Number of examples
        private const int _rowNumber = 1000;
        // Number of features
        private const int _columnNumber = 5;
        // Number of classes
        private const int _classNumber = 3;
        private class GbmExample
        {
            [VectorType(_columnNumber)]
            public float[] Features;
            [KeyType(_classNumber)]
            public uint Label;
            [VectorType(_classNumber)]
            public float[] Score;
        }

        private void LightGbmHelper(bool useSoftmax, double sigmoid, out string modelString, out List<GbmExample> mlnetPredictions, out double[] lgbmRawScores, out double[] lgbmProbabilities, bool unbalancedSets = false)
        {
            // Prepare data and train LightGBM model via ML.NET
            // Training matrix. It contains all feature vectors.
            var dataMatrix = new float[_rowNumber * _columnNumber];
            // Labels for multi-class classification
            var labels = new uint[_rowNumber];
            // Training list, which is equivalent to the training matrix above.
            var dataList = new List<GbmExample>();
            for (/*row index*/ int i = 0; i < _rowNumber; ++i)
            {
                int featureSum = 0;
                var featureVector = new float[_columnNumber];
                for (/*column index*/ int j = 0; j < _columnNumber; ++j)
                {
                    int featureValue = (j + i * _columnNumber) % 10;
                    featureSum += featureValue;
                    dataMatrix[j + i * _columnNumber] = featureValue;
                    featureVector[j] = featureValue;
                }
                labels[i] = (uint)featureSum % _classNumber;
                dataList.Add(new GbmExample { Features = featureVector, Label = labels[i], Score = new float[_classNumber] });
            }

            var mlContext = new MLContext(seed: 0);
            var dataView = mlContext.Data.LoadFromEnumerable(dataList);
            int numberOfTrainingIterations = 3;
            var gbmTrainer = new LightGbmMulticlassTrainer(mlContext,
                new LightGbmMulticlassTrainer.Options
                {
                    NumberOfIterations = numberOfTrainingIterations,
                    MinimumExampleCountPerGroup = 1,
                    MinimumExampleCountPerLeaf = 1,
                    UseSoftmax = useSoftmax,
                    Sigmoid = sigmoid, // Custom sigmoid value.
                    UnbalancedSets = unbalancedSets // false by default
                });

            var gbm = gbmTrainer.Fit(dataView);
            var predicted = gbm.Transform(dataView);
            mlnetPredictions = mlContext.Data.CreateEnumerable<GbmExample>(predicted, false).ToList();

            // Convert training to LightGBM's native format and train LightGBM model via its APIs
            // Convert the whole training matrix to CSC format required by LightGBM interface. Notice that the training matrix
            // is dense so this conversion is simply a matrix transpose.
            double[][] sampleValueGroupedByColumn = new double[_columnNumber][];
            int[][] sampleIndicesGroupedByColumn = new int[_columnNumber][];
            int[] sampleNonZeroCntPerColumn = new int[_columnNumber];
            for (int j = 0; j < _columnNumber; ++j)
            {
                // Allocate memory for the j-th column in the training matrix
                sampleValueGroupedByColumn[j] = new double[_rowNumber];
                sampleIndicesGroupedByColumn[j] = new int[_rowNumber];
                sampleNonZeroCntPerColumn[j] = _rowNumber;
                // Copy the j-th column in training matrix
                for (int i = 0; i < _rowNumber; ++i)
                {
                    // data[j + i * _columnNumber] is the value at the j-th column and the i-th row.
                    sampleValueGroupedByColumn[j][i] = dataMatrix[j + i * _columnNumber];
                    // Row index of the assigned value.
                    sampleIndicesGroupedByColumn[j][i] = i;
                }
            }

            // LightGBM only accepts float labels.
            float[] floatLabels = new float[_rowNumber];
            for (int i = 0; i < _rowNumber; ++i)
                floatLabels[i] = labels[i];

            // Allocate LightGBM data container (called Dataset in LightGBM world).
            var gbmDataSet = new Trainers.LightGbm.Dataset(sampleValueGroupedByColumn, sampleIndicesGroupedByColumn, _columnNumber, sampleNonZeroCntPerColumn, _rowNumber, _rowNumber, "", floatLabels);

            // Push training examples into LightGBM data container.
            gbmDataSet.PushRows(dataMatrix, _rowNumber, _columnNumber, 0);

            // Probability output.
            lgbmProbabilities = new double[_rowNumber * _classNumber];
            // Raw score.
            lgbmRawScores = new double[_rowNumber * _classNumber];

            // Get parameters used in ML.NET's LightGBM
            var gbmParams = gbmTrainer.GetGbmParameters();

            // Call LightGBM C-style APIs to do prediction.
            modelString = null;
            using (var ch = (mlContext as IChannelProvider).Start("Training LightGBM..."))
            using (var pch = (mlContext as IProgressChannelProvider).StartProgressChannel("Training LightGBM..."))
            {
                var host = (mlContext as IHostEnvironment).Register("Training LightGBM...");

                using (var gbmNative = WrappedLightGbmTraining.Train(ch, pch, gbmParams, gbmDataSet, numIteration: numberOfTrainingIterations))
                {
                    int nativeLength = 0;
                    unsafe
                    {
                        fixed (float* data = dataMatrix)
                        fixed (double* result0 = lgbmProbabilities)
                        fixed (double* result1 = lgbmRawScores)
                        {
                            WrappedLightGbmInterface.BoosterPredictForMat(gbmNative.Handle, (IntPtr)data, WrappedLightGbmInterface.CApiDType.Float32,
                                _rowNumber, _columnNumber, 1, (int)WrappedLightGbmInterface.CApiPredictType.Normal, numberOfTrainingIterations, "", ref nativeLength, result0);
                            WrappedLightGbmInterface.BoosterPredictForMat(gbmNative.Handle, (IntPtr)data, WrappedLightGbmInterface.CApiDType.Float32,
                                _rowNumber, _columnNumber, 1, (int)WrappedLightGbmInterface.CApiPredictType.Raw, numberOfTrainingIterations, "", ref nativeLength, result1);
                        }
                        modelString = gbmNative.GetModelString();
                    }
                }
            }
        }

        [LightGBMFact]
        public void LightGbmMulticlassEstimatorCompareOva()
        {
            float sigmoidScale = 0.5f; // Constant used train LightGBM. See gbmParams["sigmoid"] in the helper function.

            // Train ML.NET LightGBM and native LightGBM and apply the trained models to the training set.
            LightGbmHelper(useSoftmax: false, sigmoid: sigmoidScale, out string modelString, out List<GbmExample> mlnetPredictions, out double[] nativeResult1, out double[] nativeResult0);

            // The i-th predictor returned by LightGBM produces the raw score, denoted by z_i, of the i-th class.
            // Assume that we have n classes in total. The i-th class probability can be computed via
            // p_i = sigmoid(sigmoidScale * z_i) / (sigmoid(sigmoidScale * z_1) + ... + sigmoid(sigmoidScale * z_n)).
            Assert.True(modelString != null);
            // Compare native LightGBM's and ML.NET's LightGBM results example by example
            for (int i = 0; i < _rowNumber; ++i)
            {
                double sum = 0;
                for (int j = 0; j < _classNumber; ++j)
                {
                    Assert.Equal(nativeResult0[j + i * _classNumber], mlnetPredictions[i].Score[j], 6);
                    if (float.IsNaN((float)nativeResult1[j + i * _classNumber]))
                        continue;
                    sum += MathUtils.SigmoidSlow(sigmoidScale * (float)nativeResult1[j + i * _classNumber]);
                }
                for (int j = 0; j < _classNumber; ++j)
                {
                    double prob = MathUtils.SigmoidSlow(sigmoidScale * (float)nativeResult1[j + i * _classNumber]);
                    Assert.Equal(prob / sum, mlnetPredictions[i].Score[j], 6);
                }
            }

            Done();
        }

        /// <summary>
        /// Test LightGBM's sigmoid parameter with a custom value. This test checks if ML.NET and LightGBM produce the same result.
        /// </summary>
        [LightGBMFact]
        public void LightGbmMulticlassEstimatorCompareOvaUsingSigmoids()
        {
            var sigmoidScale = .790;
            // Train ML.NET LightGBM and native LightGBM and apply the trained models to the training set.
            LightGbmHelper(useSoftmax: false, sigmoid: sigmoidScale, out string modelString, out List<GbmExample> mlnetPredictions, out double[] nativeResult1, out double[] nativeResult0);

            // The i-th predictor returned by LightGBM produces the raw score, denoted by z_i, of the i-th class.
            // Assume that we have n classes in total. The i-th class probability can be computed via
            // p_i = sigmoid(sigmoidScale * z_i) / (sigmoid(sigmoidScale * z_1) + ... + sigmoid(sigmoidScale * z_n)).
            Assert.True(modelString != null);

            // Compare native LightGBM's and ML.NET's LightGBM results example by example
            for (int i = 0; i < _rowNumber; ++i)
            {
                double sum = 0;
                for (int j = 0; j < _classNumber; ++j)
                {
                    Assert.Equal(nativeResult0[j + i * _classNumber], mlnetPredictions[i].Score[j], 6);
                    if (float.IsNaN((float)nativeResult1[j + i * _classNumber]))
                        continue;
                    sum += MathUtils.SigmoidSlow((float)sigmoidScale * (float)nativeResult1[j + i * _classNumber]);
                }
                for (int j = 0; j < _classNumber; ++j)
                {
                    double prob = MathUtils.SigmoidSlow((float)sigmoidScale * (float)nativeResult1[j + i * _classNumber]);
                    Assert.Equal(prob / sum, mlnetPredictions[i].Score[j], 6);
                }
            }

            Done();
        }

        /// <summary>
        /// Make sure different sigmoid parameters produce different scores. In this test, two LightGBM models are trained with two different sigmoid values.
        /// </summary>
        [LightGBMFact]
        public void LightGbmMulticlassEstimatorCompareOvaUsingDifferentSigmoids()
        {
            // Run native implementation twice, see that results are different with different sigmoid values.
            var firstSigmoidScale = .790;
            var secondSigmoidScale = .2;

            // Train native LightGBM with both sigmoid values and apply the trained models to the training set.
            LightGbmHelper(useSoftmax: false, sigmoid: firstSigmoidScale, out string firstModelString, out List<GbmExample> firstMlnetPredictions, out double[] firstNativeResult1, out double[] firstNativeResult0);
            LightGbmHelper(useSoftmax: false, sigmoid: secondSigmoidScale, out string secondModelString, out List<GbmExample> secondMlnetPredictions, out double[] secondNativeResult1, out double[] secondNativeResult0);

            // Compare native LightGBM's results when 2 different sigmoid values are used.
            for (int i = 0; i < _rowNumber; ++i)
            {
                var areEqual = true;
                for (int j = 0; j < _classNumber; ++j)
                {
                    if (float.IsNaN((float)firstNativeResult1[j + i * _classNumber]))
                        continue;
                    if (float.IsNaN((float)secondNativeResult1[j + i * _classNumber]))
                        continue;

                    // Testing to make sure that at least 1 value is different. This avoids false positives when values are 0
                    // even for the same sigmoid value.
                    areEqual &= firstMlnetPredictions[i].Score[j].Equals(secondMlnetPredictions[i].Score[j]);

                    // Testing that the native result is different before we apply the sigmoid.
                    Assert.NotEqual((float)firstNativeResult1[j + i * _classNumber], (float)secondNativeResult1[j + i * _classNumber], 6);
                }

                // There should be at least 1 value that is different in the row.
                Assert.False(areEqual);
            }

            Done();
        }

        [LightGBMFact]
        public void LightGbmMulticlassEstimatorCompareSoftMax()
        {
            // Train ML.NET LightGBM and native LightGBM and apply the trained models to the training set.
            LightGbmHelper(useSoftmax: true, sigmoid: .5, out string modelString, out List<GbmExample> mlnetPredictions, out double[] nativeResult1, out double[] nativeResult0);

            // The i-th predictor returned by LightGBM produces the raw score, denoted by z_i, of the i-th class.
            // Assume that we have n classes in total. The i-th class probability can be computed via
            // p_i = exp(z_i) / (exp(z_1) + ... + exp(z_n)).
            Assert.True(modelString != null);
            // Compare native LightGBM's and ML.NET's LightGBM results example by example
            for (int i = 0; i < _rowNumber; ++i)
            {
                double sum = 0;
                for (int j = 0; j < _classNumber; ++j)
                {
                    Assert.Equal(nativeResult0[j + i * _classNumber], mlnetPredictions[i].Score[j], 6);
                    sum += Math.Exp((float)nativeResult1[j + i * _classNumber]);
                }
                for (int j = 0; j < _classNumber; ++j)
                {
                    double prob = Math.Exp(nativeResult1[j + i * _classNumber]);
                    Assert.Equal(prob / sum, mlnetPredictions[i].Score[j], 6);
                }
            }

            Done();
        }

        [LightGBMFact]
        public void LightGbmMulticlassEstimatorCompareUnbalanced()
        {
            // Train ML.NET LightGBM and native LightGBM and apply the trained models to the training set.
            LightGbmHelper(useSoftmax: true, sigmoid: .5, out string modelString, out List<GbmExample> mlnetPredictions, out double[] nativeResult1, out double[] nativeResult0, unbalancedSets: true);

            // The i-th predictor returned by LightGBM produces the raw score, denoted by z_i, of the i-th class.
            // Assume that we have n classes in total. The i-th class probability can be computed via
            // p_i = exp(z_i) / (exp(z_1) + ... + exp(z_n)).
            Assert.True(modelString != null);
            // Compare native LightGBM's and ML.NET's LightGBM results example by example
            for (int i = 0; i < _rowNumber; ++i)
            {
                double sum = 0;
                for (int j = 0; j < _classNumber; ++j)
                {
                    Assert.Equal(nativeResult0[j + i * _classNumber], mlnetPredictions[i].Score[j], 6);
                    sum += Math.Exp((float)nativeResult1[j + i * _classNumber]);
                }
                for (int j = 0; j < _classNumber; ++j)
                {
                    double prob = Math.Exp(nativeResult1[j + i * _classNumber]);
                    Assert.Equal(prob / sum, mlnetPredictions[i].Score[j], 6);
                }
            }

            Done();
        }

        private class DataPoint
        {
            public uint Label { get; set; }

            [VectorType(20)]
            public float[] Features { get; set; }
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0, int numClasses = 3)

        {
            var random = new Random(seed);
            float randomFloat() => (float)(random.NextDouble() - 0.5);
            for (int i = 0; i < count; i++)
            {
                var label = random.Next(1, numClasses + 1);
                yield return new DataPoint
                {
                    Label = (uint)label,
                    Features = Enumerable.Repeat(label, 20)
                        .Select(x => randomFloat() + label * 0.2f).ToArray()
                };
            }
        }

        [LightGBMFact]
        public void LightGbmFitMoreThanOnce()
        {
            var mlContext = new MLContext(seed: 0);

            var pipeline =
                mlContext.Transforms.Conversion
                .MapValueToKey(nameof(DataPoint.Label))
                .Append(mlContext.MulticlassClassification.Trainers
                .LightGbm());

            var numClasses = 3;
            var dataPoints = GenerateRandomDataPoints(100, numClasses: numClasses);
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);
            var model = pipeline.Fit(trainingData);
            var numOfSubParameters = (model.LastTransformer.Model as OneVersusAllModelParameters).SubModelParameters.Length;
            Assert.Equal(numClasses, numOfSubParameters);

            numClasses = 4;
            dataPoints = GenerateRandomDataPoints(100, numClasses: numClasses);
            trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);
            model = pipeline.Fit(trainingData);
            numOfSubParameters = (model.LastTransformer.Model as OneVersusAllModelParameters).SubModelParameters.Length;
            Assert.Equal(numClasses, numOfSubParameters);

            numClasses = 2;
            dataPoints = GenerateRandomDataPoints(100, numClasses: numClasses);
            trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);
            model = pipeline.Fit(trainingData);
            numOfSubParameters = (model.LastTransformer.Model as OneVersusAllModelParameters).SubModelParameters.Length;
            Assert.Equal(numClasses, numOfSubParameters);

            Done();
        }

        [LightGBMFact]
        public void LightGbmInDifferentCulture()
        {
            var currentCulture = Thread.CurrentThread.CurrentCulture;
            Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
            var (pipeline, dataView) = GetMulticlassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LightGbm(learningRate: 0.4);
            var pipe = pipeline.Append(trainer)
                    .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var model = pipe.Fit(dataView);
            var metrics = ML.MulticlassClassification.Evaluate(model.Transform(dataView));
            Assert.True(metrics.MacroAccuracy > 0.8);
            Thread.CurrentThread.CurrentCulture = currentCulture;
        }

        private class SummaryDataRow
        {
            public double Bias { get; set; }
            public double TreeWeights { get; set; }
            public int TreeID { get; set; }
            public string IsLeaf { get; set; }
            public int LeftChild { get; set; }
            public int RightChild { get; set; }
            public int NumericalSplitFeatureIndexes { get; set; }
            public float NumericalSplitThresholds { get; set; }
            public bool CategoricalSplitFlags { get; set; }
            public double LeafValues { get; set; }
            public double SplitGains { get; set; }
            [VectorType(0)]
            public int[] CategoricalSplitFeatures { get; set; }
            [VectorType(0)]
            public int[] CategoricalCategoricalSplitFeatureRange { get; set; }
        }

        private class QuantileTestSummaryDataRow : SummaryDataRow
        {
            [VectorType(0)]
            public double[] LeafSamples { get; set; }
            [VectorType(0)]
            public double[] LeafSampleWeights { get; set; }
        }

        private static void CheckSummaryRowTreeNode(SummaryDataRow row, int treeIndex, double bias, double treeWeight, RegressionTreeBase tree, int nodeId)
        {
            Assert.Equal(row.TreeID, treeIndex);
            Assert.Equal(row.Bias, bias);
            Assert.Equal(row.TreeWeights, treeWeight);
            Assert.Equal("Tree node", row.IsLeaf);
            Assert.Equal(row.LeftChild, tree.LeftChild[nodeId]);
            Assert.Equal(row.RightChild, tree.RightChild[nodeId]);
            Assert.Equal(row.NumericalSplitFeatureIndexes, tree.NumericalSplitFeatureIndexes[nodeId]);
            Assert.Equal(row.NumericalSplitThresholds, tree.NumericalSplitThresholds[nodeId]);
            Assert.Equal(row.CategoricalSplitFlags, tree.CategoricalSplitFlags[nodeId]);
            Assert.Equal(0, row.LeafValues);
            Assert.Equal(row.SplitGains, tree.SplitGains[nodeId]);
            if (tree.GetCategoricalSplitFeaturesAt(nodeId).Count() > 0)
                Assert.Equal(row.CategoricalSplitFeatures, tree.GetCategoricalSplitFeaturesAt(nodeId).ToArray());
            else
                Assert.Null(row.CategoricalSplitFeatures);
            if (tree.GetCategoricalCategoricalSplitFeatureRangeAt(nodeId).Count() > 0)
                Assert.Equal(row.CategoricalCategoricalSplitFeatureRange, tree.GetCategoricalCategoricalSplitFeatureRangeAt(nodeId).ToArray());
            else
                Assert.Null(row.CategoricalCategoricalSplitFeatureRange);
        }

        private static void CheckSummaryRowLeafNode(SummaryDataRow row, int treeIndex, double bias, double treeWeight, RegressionTreeBase tree, int nodeId)
        {
            Assert.Equal(row.TreeID, treeIndex);
            Assert.Equal(row.Bias, bias);
            Assert.Equal(row.TreeWeights, treeWeight);
            Assert.Equal("Leaf node", row.IsLeaf);
            Assert.Equal(0, row.LeftChild);
            Assert.Equal(0, row.RightChild);
            Assert.Equal(0, row.NumericalSplitFeatureIndexes);
            Assert.Equal(0, row.NumericalSplitThresholds);
            Assert.False(row.CategoricalSplitFlags);
            Assert.Equal(tree.LeafValues[nodeId], row.LeafValues);
            Assert.Equal(0d, row.SplitGains);
            Assert.Null(row.CategoricalSplitFeatures);
            Assert.Null(row.CategoricalCategoricalSplitFeatureRange);
        }

        private static void CheckSummaryRowLeafNodeQuantileTree(QuantileTestSummaryDataRow row, int treeIndex, double bias, double treeWeight, QuantileRegressionTree tree, int nodeId)
        {
            if (tree.GetLeafSamplesAt(nodeId).Count() > 0)
                Assert.Equal(row.LeafSamples, tree.GetLeafSamplesAt(nodeId).ToArray());
            else
                Assert.Null(row.LeafSamples);
            if (tree.GetLeafSampleWeightsAt(nodeId).Count() > 0)
                Assert.Equal(row.LeafSampleWeights, tree.GetLeafSampleWeightsAt(nodeId).ToArray());
            else
                Assert.Null(row.LeafSampleWeights);
        }

        private void CheckSummary(ICanGetSummaryAsIDataView modelParameters, double bias, IReadOnlyList<double> treeWeights, IReadOnlyList<RegressionTreeBase> trees)
        {
            var quantileTrees = trees as IReadOnlyList<QuantileRegressionTree>;
            var summaryDataView = modelParameters.GetSummaryDataView(null);
            IEnumerable<SummaryDataRow> summaryDataEnumerable;

            if (quantileTrees == null)
                summaryDataEnumerable = ML.Data.CreateEnumerable<SummaryDataRow>(summaryDataView, false);
            else
                summaryDataEnumerable = ML.Data.CreateEnumerable<QuantileTestSummaryDataRow>(summaryDataView, false);

            var summaryDataEnumerator = summaryDataEnumerable.GetEnumerator();

            for (int i = 0; i < trees.Count(); i++)
            {
                for (int j = 0; j < trees[i].NumberOfNodes; j++)
                {
                    Assert.True(summaryDataEnumerator.MoveNext());
                    var row = summaryDataEnumerator.Current;
                    CheckSummaryRowTreeNode(row, i, bias, treeWeights[i], trees[i], j);
                }

                for (int j = 0; j < trees[i].NumberOfLeaves; j++)
                {
                    Assert.True(summaryDataEnumerator.MoveNext());
                    var row = summaryDataEnumerator.Current;
                    CheckSummaryRowLeafNode(row, i, bias, treeWeights[i], trees[i], j);
                    if (quantileTrees != null)
                    {
                        var quantileRow = row as QuantileTestSummaryDataRow;
                        Assert.NotNull(quantileRow);
                        CheckSummaryRowLeafNodeQuantileTree(quantileRow, i, bias, treeWeights[i], quantileTrees[i], j);
                    }
                }
            }
        }

        [Fact]
        public void FastTreeRegressorTestSummary()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastTree(
                new FastTreeRegressionTrainer.Options { NumberOfTrees = 10, NumberOfThreads = 1, NumberOfLeaves = 5 });

            var transformer = trainer.Fit(dataView);

            var trainedTreeEnsemble = transformer.Model.TrainedTreeEnsemble;

            var modelParameters = transformer.Model as ICanGetSummaryAsIDataView;
            Assert.NotNull(modelParameters);

            CheckSummary(modelParameters, trainedTreeEnsemble.Bias, trainedTreeEnsemble.TreeWeights, trainedTreeEnsemble.Trees);
            Done();
        }

        [Fact]
        public void FastForestRegressorTestSummary()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastForest(
                new FastForestRegressionTrainer.Options { NumberOfTrees = 10, NumberOfThreads = 1, NumberOfLeaves = 5 });

            var transformer = trainer.Fit(dataView);

            var trainedTreeEnsemble = transformer.Model.TrainedTreeEnsemble;

            var modelParameters = transformer.Model as ICanGetSummaryAsIDataView;
            Assert.NotNull(modelParameters);

            CheckSummary(modelParameters, trainedTreeEnsemble.Bias, trainedTreeEnsemble.TreeWeights, trainedTreeEnsemble.Trees);
            Done();
        }

        [Fact]
        public void FastTreeTweedieRegressorTestSummary()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastTreeTweedie(
                new FastTreeTweedieTrainer.Options { NumberOfTrees = 10, NumberOfThreads = 1, NumberOfLeaves = 5 });

            var transformer = trainer.Fit(dataView);

            var trainedTreeEnsemble = transformer.Model.TrainedTreeEnsemble;

            var modelParameters = transformer.Model as ICanGetSummaryAsIDataView;
            Assert.NotNull(modelParameters);

            CheckSummary(modelParameters, trainedTreeEnsemble.Bias, trainedTreeEnsemble.TreeWeights, trainedTreeEnsemble.Trees);
            Done();
        }

        [LightGBMFact]
        public void LightGbmRegressorTestSummary()
        {
            var dataView = GetRegressionPipeline();

            // Attention: Do not set NumberOfThreads here, left this to use default value to avoid test crash.
            // Details can be found here: https://github.com/dotnet/machinelearning/pull/4918
            var trainer = ML.Regression.Trainers.LightGbm(
                new LightGbmRegressionTrainer.Options
                {
                    NumberOfIterations = 10,
                    NumberOfLeaves = 5
                });

            var transformer = trainer.Fit(dataView);

            var trainedTreeEnsemble = transformer.Model.TrainedTreeEnsemble;

            var modelParameters = transformer.Model as ICanGetSummaryAsIDataView;
            Assert.NotNull(modelParameters);

            CheckSummary(modelParameters, trainedTreeEnsemble.Bias, trainedTreeEnsemble.TreeWeights, trainedTreeEnsemble.Trees);
            Done();
        }

        [Fact]
        public void FastTreeBinaryClassificationTestSummary()
        {
            var (pipeline, dataView) = GetBinaryClassificationPipeline();
            var estimator = pipeline.Append(ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options { NumberOfTrees = 2, NumberOfThreads = 1, NumberOfLeaves = 5 }));

            var transformer = estimator.Fit(dataView);

            var trainedTreeEnsemble = transformer.LastTransformer.Model.SubModel.TrainedTreeEnsemble;

            var modelParameters = transformer.LastTransformer.Model.SubModel as ICanGetSummaryAsIDataView;
            Assert.NotNull(modelParameters);

            CheckSummary(modelParameters, trainedTreeEnsemble.Bias, trainedTreeEnsemble.TreeWeights, trainedTreeEnsemble.Trees);
            Done();
        }

        [Fact]
        public void FastForestBinaryClassificationTestSummary()
        {
            var (pipeline, dataView) = GetOneHotBinaryClassificationPipeline();
            var estimator = pipeline.Append(ML.BinaryClassification.Trainers.FastForest(
                new FastForestBinaryTrainer.Options { NumberOfTrees = 2, NumberOfThreads = 1, NumberOfLeaves = 4, CategoricalSplit = true }));

            var transformer = estimator.Fit(dataView);

            var trainedTreeEnsemble = transformer.LastTransformer.Model.TrainedTreeEnsemble;

            var modelParameters = transformer.LastTransformer.Model as ICanGetSummaryAsIDataView;
            Assert.NotNull(modelParameters);

            CheckSummary(modelParameters, trainedTreeEnsemble.Bias, trainedTreeEnsemble.TreeWeights, trainedTreeEnsemble.Trees);
            Done();
        }

        [LightGBMFact]
        public void LightGbmBinaryClassificationTestSummary()
        {
            var (pipeline, dataView) = GetOneHotBinaryClassificationPipeline();

            // Attention: Do not set NumberOfThreads here, left this to use default value to avoid test crash.
            // Details can be found here: https://github.com/dotnet/machinelearning/pull/4918
            var trainer = pipeline.Append(ML.BinaryClassification.Trainers.LightGbm(
                new LightGbmBinaryTrainer.Options
                {
                    NumberOfIterations = 10,
                    NumberOfLeaves = 5,
                    UseCategoricalSplit = true
                }));

            var transformer = trainer.Fit(dataView);

            var trainedTreeEnsemble = transformer.LastTransformer.Model.SubModel.TrainedTreeEnsemble;

            var modelParameters = transformer.LastTransformer.Model.SubModel as ICanGetSummaryAsIDataView;
            Assert.NotNull(modelParameters);

            CheckSummary(modelParameters, trainedTreeEnsemble.Bias, trainedTreeEnsemble.TreeWeights, trainedTreeEnsemble.Trees);
            Done();
        }
    }
}
