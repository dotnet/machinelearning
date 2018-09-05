// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;

namespace Microsoft.ML.Runtime.Learners
{
    /// <summary>
    /// Extension methods and utilities.
    /// </summary>
    public static class SdcaStatic
    {
        public static Scalar<float> PredictSdcaRegression(this Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            float? l2Const = null,
            float? l1Threshold = null,
            float convergenceTolerance = 0.01f,
            int? maxIterations = null,
            bool shuffle = true,
            float biasLearningRate = 1,
            ISupportSdcaRegressionLossFactory loss = null,
            Action<LinearRegressionPredictor> onFit = null)
        {
            var args = new SdcaRegressionTrainer.Arguments()
            {
                L2Const = l2Const,
                L1Threshold = l1Threshold,
                ConvergenceTolerance = convergenceTolerance,
                MaxIterations = maxIterations,
                Shuffle = shuffle,
                BiasLearningRate = biasLearningRate,
                LossFunction = loss ?? new SquaredLossFactory()
            };

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaRegressionTrainer(env, args, featuresName, labelName, weightsName);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }

        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel)
            PredictSdcaBinaryClassification(this Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
                float? l2Const = null,
                float? l1Threshold = null,
                float convergenceTolerance = 0.1f,
                int? maxIterations = null,
                bool shuffle = true,
                float biasLearningRate = 0,
                Action<LinearBinaryPredictor, ParameterMixingCalibratedPredictor> onFit = null)
        {
            var args = new LinearClassificationTrainer.Arguments()
            {
                L2Const = l2Const,
                L1Threshold = l1Threshold,
                ConvergenceTolerance = convergenceTolerance,
                MaxIterations = maxIterations,
                Shuffle = shuffle,
                BiasLearningRate = biasLearningRate
            };

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LinearClassificationTrainer(env, args, featuresName, labelName, weightsName);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            // Under the default log-loss we assume a calibrated predictor.
                            var model = trans.Model;
                            var cali = (ParameterMixingCalibratedPredictor)model;
                            var pred = (LinearBinaryPredictor)cali.SubPredictor;
                            onFit(pred, cali);
                        });
                    }
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        private sealed class TrivialFactory : ISupportSdcaClassificationLossFactory
        {
            private readonly ISupportSdcaClassificationLoss _loss;

            public TrivialFactory(ISupportSdcaClassificationLoss loss)
            {
                _loss = loss;
            }

            public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env)
            {
                // REVIEW: We are ignoring env?
                return _loss;
            }
        }

        public static (Scalar<float> score, Scalar<bool> predictedLabel)
            PredictSdcaBinaryClassificationCustomLoss(this Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
                float? l2Const = null,
                float? l1Threshold = null,
                float convergenceTolerance = 0.1f,
                int? maxIterations = null,
                bool shuffle = true,
                float biasLearningRate = 0,
                ISupportSdcaClassificationLoss loss = null,
                Action<LinearBinaryPredictor> onFit = null
            )
        {
            ISupportSdcaClassificationLossFactory lossFactory = new LogLossFactory();
            if (loss != null)
                lossFactory = new TrivialFactory(loss);
            bool hasProbs = lossFactory is LogLossFactory || loss is LogLoss;

            var args = new LinearClassificationTrainer.Arguments()
            {
                L2Const = l2Const,
                L1Threshold = l1Threshold,
                ConvergenceTolerance = convergenceTolerance,
                MaxIterations = maxIterations,
                Shuffle = shuffle,
                LossFunction = lossFactory,
                BiasLearningRate = biasLearningRate
            };

            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LinearClassificationTrainer(env, args, featuresName, labelName, weightsName);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            var model = trans.Model;
                            if (model is ParameterMixingCalibratedPredictor cali)
                                onFit((LinearBinaryPredictor)cali.SubPredictor);
                            else
                                onFit((LinearBinaryPredictor)model);
                        });
                    }
                    return trainer;
                }, label, features, weights, hasProbs);

            return rec.Output;
        }
    }
}
