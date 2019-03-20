// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(LogisticRegressionBinaryTrainer.Summary, typeof(LogisticRegressionBinaryTrainer), typeof(LogisticRegressionBinaryTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LogisticRegressionBinaryTrainer.UserNameValue,
    LogisticRegressionBinaryTrainer.LoadNameValue,
    LogisticRegressionBinaryTrainer.ShortName,
    "logisticregressionwrapper")]

[assembly: LoadableClass(typeof(void), typeof(LogisticRegressionBinaryTrainer), null, typeof(SignatureEntryPointModule), LogisticRegressionBinaryTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers
{

    /// <include file='doc.xml' path='doc/members/member[@name="LBFGS"]/*' />
    /// <include file='doc.xml' path='docs/members/example[@name="LogisticRegressionBinaryClassifier"]/*' />
    public sealed partial class LogisticRegressionBinaryTrainer : LbfgsTrainerBase<LogisticRegressionBinaryTrainer.Options,
        BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>,
        CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        internal const string LoadNameValue = "LogisticRegression";
        internal const string UserNameValue = "Logistic Regression";
        internal const string ShortName = "lr";
        internal const string Summary = "Logistic Regression is a method in statistics used to predict the probability of occurrence of an event and can "
            + "be used as a classification algorithm. The algorithm predicts the probability of occurrence of an event by fitting data to a logistical function.";

        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// If set to <value>true</value> training statistics will be generated at the end of training.
            /// If you have a large number of learned training parameters(more than 500),
            /// generating the training statistics might take a few seconds.
            /// More than 1000 weights might take a few minutes. For those cases consider using the instance of <see cref="ComputeLogisticRegressionStandardDeviation"/>
            /// present in the Microsoft.ML.Mkl.Components package. That computes the statistics using hardware acceleration.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show statistics of training examples.", ShortName = "stat, ShowTrainingStats", SortOrder = 50)]
            public bool ShowTrainingStatistics = false;

            /// <summary>
            /// The instance of <see cref="ComputeLogisticRegressionStandardDeviation"/> that computes the std of the training statistics, at the end of training.
            /// The calculations are not part of Microsoft.ML package, due to the size of MKL.
            /// If you need these calculations, add the Microsoft.ML.Mkl.Components package, and initialize <see cref="LogisticRegressionBinaryTrainer.Options.ComputeStandardDeviation"/>.
            /// to the <see cref="ComputeLogisticRegressionStandardDeviation"/> implementation in the Microsoft.ML.Mkl.Components package.
            /// </summary>
            public ComputeLogisticRegressionStandardDeviation ComputeStandardDeviation;
        }

        private double _posWeight;
        private ModelStatisticsBase _stats;

        /// <summary>
        /// Initializes a new instance of <see cref="LogisticRegressionBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularizer term.</param>
        /// <param name="l2Weight">Weight of L2 regularizer term.</param>
        /// <param name="memorySize">Memory size for <see cref="LogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        internal LogisticRegressionBinaryTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = Options.Defaults.L1Regularization,
            float l2Weight = Options.Defaults.L2Regularization,
            float optimizationTolerance = Options.Defaults.OptimizationTolerance,
            int memorySize = Options.Defaults.HistorySize,
            bool enforceNoNegativity = Options.Defaults.EnforceNonNegativity)
            : base(env, featureColumn, TrainerUtils.MakeBoolScalarLabel(labelColumn), weights,
                  l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));

            _posWeight = 0;
            ShowTrainingStats = LbfgsTrainerOptions.ShowTrainingStatistics;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LogisticRegressionBinaryTrainer"/>
        /// </summary>
        internal LogisticRegressionBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            _posWeight = 0;
            ShowTrainingStats = LbfgsTrainerOptions.ShowTrainingStatistics;
        }

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
            MakeTransformer(CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator> model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Continues the training of a <see cref="LogisticRegressionBinaryTrainer"/> using an already trained <paramref name="modelParameters"/> and returns
        /// a <see cref="BinaryPredictionTransformer{CalibratedModelParametersBase}"/>.
        /// </summary>
        public BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>> Fit(IDataView trainData, LinearModelParameters modelParameters)
            => TrainTransformer(trainData, initPredictor: modelParameters);

        private protected override float AccumulateOneGradient(in VBuffer<float> feat, float label, float weight,
            in VBuffer<float> x, ref VBuffer<float> grad, ref float[] scratch)
        {
            float bias = 0;
            x.GetItemOrDefault(0, ref bias);
            float score = bias + VectorUtils.DotProductWithOffset(in x, 1, in feat);

            float s = score / 2;

            float logZ = MathUtils.SoftMax(s, -s);
            float label01 = Math.Min(1, Math.Max(label, 0));
            float label11 = 2 * label01 - 1; //(-1..1) label
            float modelProb1 = MathUtils.ExpSlow(s - logZ);
            float ls = label11 * s;
            float datumLoss = logZ - ls;
            //float loss2 = MathUtil.SoftMax(s-l_s, -s-l_s);

            Contracts.Check(!float.IsNaN(datumLoss), "Unexpected NaN");

            float mult = weight * (modelProb1 - label01);
            VectorUtils.AddMultWithOffset(in feat, mult, ref grad, 1); // Note that 0th L-BFGS weight is for bias.
            // Add bias using this strange trick that has advantage of working well for dense and sparse arrays.
            // Due to the call to EnsureBiases, we know this region is dense.
            var editor = VBufferEditor.CreateFromBuffer(ref grad);
            Contracts.Assert(editor.Values.Length >= BiasCount && (grad.IsDense || editor.Indices[BiasCount - 1] == BiasCount - 1));
            editor.Values[0] += mult;

            return weight * datumLoss;
        }

        private protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory cursorFactory, float loss, int numParams)
        {
            Contracts.AssertValue(ch);
            Contracts.AssertValue(cursorFactory);
            Contracts.Assert(NumGoodRows > 0);
            Contracts.Assert(WeightSum > 0);
            Contracts.Assert(BiasCount == 1);
            Contracts.Assert(loss >= 0);
            Contracts.Assert(numParams >= BiasCount);
            Contracts.Assert(CurrentWeights.IsDense);

            ch.Info("Model trained with {0} training examples.", NumGoodRows);

            // Compute deviance: start with loss function.
            float deviance = (float)(2 * loss * WeightSum);
            var currentWeightsValues = CurrentWeights.GetValues();

            if (L2Weight > 0)
            {
                // Need to subtract L2 regularization loss.
                // The bias term is not regularized.
                var regLoss = VectorUtils.NormSquared(currentWeightsValues.Slice(1)) * L2Weight;
                deviance -= regLoss;
            }

            if (L1Weight > 0)
            {
                // Need to subtract L1 regularization loss.
                // The bias term is not regularized.
                Double regLoss = 0;
                VBufferUtils.ForEachDefined(in CurrentWeights, (ind, value) => { if (ind >= BiasCount) regLoss += Math.Abs(value); });
                deviance -= (float)regLoss * L1Weight * 2;
            }

            ch.Info("Residual Deviance: \t{0} (on {1} degrees of freedom)", deviance, Math.Max(NumGoodRows - numParams, 0));

            // Compute null deviance, i.e., the deviance of null hypothesis.
            // Cap the prior positive rate at 1e-15.
            Double priorPosRate = _posWeight / WeightSum;
            Contracts.Assert(0 <= priorPosRate && priorPosRate <= 1);
            float nullDeviance = (priorPosRate <= 1e-15 || 1 - priorPosRate <= 1e-15) ?
                0f : (float)(2 * WeightSum * MathUtils.Entropy(priorPosRate, true));
            ch.Info("Null Deviance:     \t{0} (on {1} degrees of freedom)", nullDeviance, NumGoodRows - 1);

            // Compute AIC.
            ch.Info("AIC:               \t{0}", 2 * numParams + deviance);

            // Show the coefficients statistics table.
            var featureCol = cursorFactory.Data.Schema.Feature.Value;
            var schema = cursorFactory.Data.Data.Schema;
            var featureLength = CurrentWeights.Length - BiasCount;
            var namesSpans = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(featureLength);
            if (featureCol.HasSlotNames(featureLength))
                featureCol.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref namesSpans);
            Host.Assert(namesSpans.Length == featureLength);

            // Inverse mapping of non-zero weight slots.
            Dictionary<int, int> weightIndicesInvMap = null;

            // Indices of bias and non-zero weight slots.
            int[] weightIndices = null;

            // Whether all weights are non-zero.
            bool denseWeight = numParams == CurrentWeights.Length;

            // Extract non-zero indices of weight.
            if (!denseWeight)
            {
                weightIndices = new int[numParams];
                weightIndicesInvMap = new Dictionary<int, int>(numParams);
                weightIndices[0] = 0;
                weightIndicesInvMap[0] = 0;
                int j = 1;
                for (int i = 1; i < currentWeightsValues.Length; i++)
                {
                    if (currentWeightsValues[i] != 0)
                    {
                        weightIndices[j] = i;
                        weightIndicesInvMap[i] = j++;
                    }
                }

                Contracts.Assert(j == numParams);
            }

            // Compute the standard error of coefficients.
            long hessianDimension = (long)numParams * (numParams + 1) / 2;
            if (hessianDimension > int.MaxValue || LbfgsTrainerOptions.ComputeStandardDeviation == null)
            {
                ch.Warning("The number of parameters is too large. Cannot hold the variance-covariance matrix in memory. " +
                    "Skipping computation of standard errors and z-statistics of coefficients. Consider choosing a larger L1 regularizer" +
                    "to reduce the number of parameters.");
                _stats = new ModelStatisticsBase(Host, NumGoodRows, numParams, deviance, nullDeviance);
                return;
            }

            // Building the variance-covariance matrix for parameters.
            // The layout of this algorithm is a packed row-major lower triangular matrix.
            // For example, layout of indices for 4-by-4:
            // 0
            // 1 2
            // 3 4 5
            // 6 7 8 9
            var hessian = new Double[hessianDimension];

            // Initialize diagonal elements with L2 regularizers except for the first entry (index 0)
            // Since bias is not regularized.
            if (L2Weight > 0)
            {
                // i is the array index of the diagonal entry at iRow-th row and iRow-th column.
                // iRow is one-based.
                int i = 0;
                for (int iRow = 2; iRow <= numParams; iRow++)
                {
                    i += iRow;
                    hessian[i] = L2Weight;
                }

                Contracts.Assert(i == hessian.Length - 1);
            }

            // Initialize the remaining entries.
            var bias = currentWeightsValues[0];
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    var label = cursor.Label;
                    var weight = cursor.Weight;
                    var score = bias + VectorUtils.DotProductWithOffset(in CurrentWeights, 1, in cursor.Features);
                    // Compute Bernoulli variance n_i * p_i * (1 - p_i) for the i-th training example.
                    var variance = weight / (2 + 2 * Math.Cosh(score));

                    // Increment the first entry of hessian.
                    hessian[0] += variance;

                    var values = cursor.Features.GetValues();
                    if (cursor.Features.IsDense)
                    {
                        int ioff = 1;

                        // Increment remaining entries of hessian.
                        for (int i = 1; i < numParams; i++)
                        {
                            ch.Assert(ioff == i * (i + 1) / 2);
                            int wi = weightIndices == null ? i - 1 : weightIndices[i] - 1;
                            Contracts.Assert(0 <= wi && wi < cursor.Features.Length);
                            var val = values[wi] * variance;
                            // Add the implicit first bias term to X'X
                            hessian[ioff++] += val;
                            // Add the remainder of X'X
                            for (int j = 0; j < i; j++)
                            {
                                int wj = weightIndices == null ? j : weightIndices[j + 1] - 1;
                                Contracts.Assert(0 <= wj && wj < cursor.Features.Length);
                                hessian[ioff++] += val * values[wj];
                            }
                        }
                        ch.Assert(ioff == hessian.Length);
                    }
                    else
                    {
                        var indices = cursor.Features.GetIndices();
                        for (int ii = 0; ii < values.Length; ++ii)
                        {
                            int i = indices[ii];
                            int wi = i + 1;
                            if (weightIndicesInvMap != null && !weightIndicesInvMap.TryGetValue(i + 1, out wi))
                                continue;

                            Contracts.Assert(0 < wi && wi <= cursor.Features.Length);
                            int ioff = wi * (wi + 1) / 2;
                            var val = values[ii] * variance;
                            // Add the implicit first bias term to X'X
                            hessian[ioff] += val;
                            // Add the remainder of X'X
                            for (int jj = 0; jj <= ii; jj++)
                            {
                                int j = indices[jj];
                                int wj = j + 1;
                                if (weightIndicesInvMap != null && !weightIndicesInvMap.TryGetValue(j + 1, out wj))
                                    continue;

                                Contracts.Assert(0 < wj && wj <= cursor.Features.Length);
                                hessian[ioff + wj] += val * values[jj];
                            }
                        }
                    }
                }
            }

            VBuffer<float> weightsOnly = default(VBuffer<float>);
            CurrentWeights.CopyTo(ref weightsOnly, 1, CurrentWeights.Length - 1);
            var std = LbfgsTrainerOptions.ComputeStandardDeviation.ComputeStandardDeviation(hessian, weightIndices, numParams, CurrentWeights.Length, ch, L2Weight);
            _stats = new LinearModelParameterStatistics(Host, NumGoodRows, numParams, deviance, nullDeviance, std, weightsOnly, bias);
        }

        private protected override void ProcessPriorDistribution(float label, float weight)
        {
            if (label > 0)
                _posWeight += weight;
        }

        //Override default termination criterion MeanRelativeImprovementCriterion with
        private protected override Optimizer InitializeOptimizer(IChannel ch, FloatLabelCursor.Factory cursorFactory,
            out VBuffer<float> init, out ITerminationCriterion terminationCriterion)
        {
            var opt = base.InitializeOptimizer(ch, cursorFactory, out init, out terminationCriterion);

            // MeanImprovementCriterion:
            //   Terminates when the geometrically-weighted average improvement falls below the tolerance
            //terminationCriterion = new GradientCheckingMonitor(new MeanImprovementCriterion(CmdArgs.optTol, 0.25, MaxIterations),2);
            terminationCriterion = new MeanImprovementCriterion(OptTol, (float)0.25, MaxIterations);

            return opt;
        }

        private protected override VBuffer<float> InitializeWeightsFromPredictor(IPredictor srcPredictor)
        {
            Contracts.AssertValue(srcPredictor);

            var pred = srcPredictor as LinearModelParameters;
            Contracts.AssertValue(pred);
            return InitializeWeights(pred.Weights, new[] { pred.Bias });
        }

        private protected override CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator> CreatePredictor()
        {
            // Logistic regression is naturally calibrated to
            // output probabilities when transformed using
            // the logistic function, so there is no need to
            // train a separate calibrator.
            VBuffer<float> weights = default(VBuffer<float>);
            float bias = 0;
            CurrentWeights.GetItemOrDefault(0, ref bias);
            CurrentWeights.CopyTo(ref weights, 1, CurrentWeights.Length - 1);
            return new ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>(Host,
                new LinearBinaryModelParameters(Host, in weights, bias, _stats),
                new PlattCalibrator(Host, -1, 0));
        }

        [TlcModule.EntryPoint(Name = "Trainers.LogisticRegressionBinaryClassifier",
            Desc = Summary,
            UserName = UserNameValue,
            ShortName = ShortName)]

        internal static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLRBinary");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LogisticRegressionBinaryTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }

    /// <summary>
    /// Computes the standard deviation matrix of each of the non-zero training weights, needed to calculate further the standard deviation,
    /// p-value and z-Score.
    /// If you need fast calculations, use the <see cref="ComputeLogisticRegressionStandardDeviation"/> implementation in the Microsoft.ML.Mkl.Components package,
    /// which makes use of hardware acceleration.
    /// Due to the existence of regularization, an approximation is used to compute the variances of the trained linear coefficients.
    /// </summary>
    public abstract class ComputeLogisticRegressionStandardDeviation
    {
        /// <summary>
        /// Computes the standard deviation matrix of each of the non-zero training weights, needed to calculate further the standard deviation,
        /// p-value and z-Score.
        /// The calculations are not part of Microsoft.ML package, due to the size of MKL.
        /// If you need these calculations, add the Microsoft.ML.Mkl.Components package, and initialize <see cref="LogisticRegressionBinaryTrainer.Options.ComputeStandardDeviation"/>
        /// to the <see cref="ComputeLogisticRegressionStandardDeviation"/> implementation in the Microsoft.ML.Mkl.Components package.
        /// Due to the existence of regularization, an approximation is used to compute the variances of the trained linear coefficients.
        /// </summary>
        public abstract VBuffer<float> ComputeStandardDeviation(double[] hessian, int[] weightIndices, int parametersCount, int currentWeightsCount, IChannel ch, float l2Weight);
    }
}
