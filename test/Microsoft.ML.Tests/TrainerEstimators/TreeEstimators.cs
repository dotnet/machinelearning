// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.Collections.Generic;
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
            [KeyType(Contiguous = true, Count =_classNumber, Min = 0)]
            public uint Label;
            [VectorType(_classNumber)]
            public float[] Score;
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmMultiClassEstimatorCompareOva()
        {
            //////////////////////////////////////////////////////////////////////////////////////////////
            // Prepare data and train LightGBM model via ML.NET
            //////////////////////////////////////////////////////////////////////////////////////////////

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

            var mlContext = new MLContext(seed: 0, conc: 1);
            var dataView = ComponentCreation.CreateDataView(mlContext, dataList);
            int numberOfTrainingIterations = 3;
            var gbmTrainer = new LightGbmMulticlassTrainer(mlContext, labelColumn: "Label", featureColumn: "Features", numBoostRound: numberOfTrainingIterations,
                advancedSettings: s => { s.MinDataPerGroup = 1; s.MinDataPerLeaf = 1; });
            var gbm = gbmTrainer.Fit(dataView);
            var predicted = gbm.Transform(dataView);
            var predictions = new List<GbmExample>(predicted.AsEnumerable<GbmExample>(mlContext, false));

            //////////////////////////////////////////////////////////////////////////////////////////////
            // Convert training to LightGBM's native format and train LightGBM model via its APIs
            //////////////////////////////////////////////////////////////////////////////////////////////

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
            var gbmDataSet = new Dataset(sampleValueGroupedByColumn, sampleIndicesGroupedByColumn, _columnNumber, sampleNonZeroCntPerColumn, _rowNumber, _rowNumber, "", floatLabels);

            // Push training examples into LightGBM data container.
            gbmDataSet.PushRows(dataMatrix, _rowNumber, _columnNumber, 0);

            // Probability output.
            var nativeResult0 = new double[_rowNumber * _classNumber];
            // Raw score.
            var nativeResult1 = new double[_rowNumber * _classNumber];

            // Get parameters used in ML.NET's LightGBM
            var gbmParams = gbmTrainer.GetGbmParameters();

            string modelString = null;
            using (var ch = (mlContext as IChannelProvider).Start("Training LightGBM..."))
            using (var pch = (mlContext as IProgressChannelProvider).StartProgressChannel("Training LightGBM..."))
            {
                var host = (mlContext as IHostEnvironment).Register("Training LightGBM...");
                var gbmNative = WrappedLightGbmTraining.Train(ch, pch, gbmParams, gbmDataSet, numIteration : numberOfTrainingIterations);

                int nativeLength = 0;
                unsafe
                {
                    fixed (float* data = dataMatrix)
                    fixed (double* result0 = nativeResult0)
                    fixed (double* result1 = nativeResult1)
                    {
                        WrappedLightGbmInterface.BoosterPredictForMat(gbmNative.Handle, (IntPtr)data, WrappedLightGbmInterface.CApiDType.Float32,
                            _rowNumber, _columnNumber, 1, 0, numberOfTrainingIterations, "", ref nativeLength, result0);
                        WrappedLightGbmInterface.BoosterPredictForMat(gbmNative.Handle, (IntPtr)data, WrappedLightGbmInterface.CApiDType.Float32,
                            _rowNumber, _columnNumber, 1, 1, numberOfTrainingIterations, "", ref nativeLength, result1);
                    }
                    modelString = gbmNative.GetModelString();
                }
            }

            //////////////////////////////////////////////////////////////////////////////////////////////
            // Compare native LightGBM's and ML.NET's LightGBM results
            //////////////////////////////////////////////////////////////////////////////////////////////

            Assert.True(modelString != null);
            float sigmoidScale = (float)(double)gbmParams["sigmoid"]; // Parameter passed to LightGBM
            for (int i = 0; i < _rowNumber; ++i)
            {
                double sum = 0;
                for (int j = 0; j < _classNumber; ++j)
                {
                    Assert.Equal(nativeResult0[j + i * _classNumber], predictions[i].Score[j], 6);
                    sum += MathUtils.SigmoidSlow(sigmoidScale* (float)nativeResult1[j + i * _classNumber]);
                }
                for (int j = 0; j < _classNumber; ++j)
                {
                    double prob = MathUtils.SigmoidSlow(sigmoidScale * (float)nativeResult1[j + i * _classNumber]);
                    Assert.Equal(prob / sum, predictions[i].Score[j], 6);
                }
            }

        }
    }
}
