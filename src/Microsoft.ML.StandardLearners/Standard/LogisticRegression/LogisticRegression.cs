// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(LogisticRegression.Summary, typeof(LogisticRegression), typeof(LogisticRegression.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LogisticRegression.UserNameValue,
    LogisticRegression.LoadNameValue,
    LogisticRegression.ShortName,
    "logisticregressionwrapper")]

[assembly: LoadableClass(typeof(void), typeof(LogisticRegression), null, typeof(SignatureEntryPointModule), LogisticRegression.LoadNameValue)]

namespace Microsoft.ML.Runtime.Learners
{

    /// <include file='doc.xml' path='doc/members/member[@name="LBFGS"]/*' />
    /// <include file='doc.xml' path='docs/members/example[@name="LogisticRegressionBinaryClassifier"]/*' />
    public sealed partial class LogisticRegression : LbfgsTrainerBase<LogisticRegression.Arguments, BinaryPredictionTransformer<ParameterMixingCalibratedPredictor>, ParameterMixingCalibratedPredictor>
    {
        public const string LoadNameValue = "LogisticRegression";
        internal const string UserNameValue = "Logistic Regression";
        internal const string ShortName = "lr";
        internal const string Summary = "Logistic Regression is a method in statistics used to predict the probability of occurrence of an event and can "
            + "be used as a classification algorithm. The algorithm predicts the probability of occurrence of an event by fitting data to a logistical function.";

        public sealed class Arguments : ArgumentsBase
        {
            /// <summary>
            /// If set to <value>true</value>training statistics will be generated at the end of training.
            /// If you have a large number of learned training parameters(more than 500),
            /// generating the training statistics might take a few seconds.
            /// More than 1000 weights might take a few minutes. For those cases consider using the instance of <see cref="IComputeLRTrainingStd"/>
            /// present in the Microsoft.ML.HalLearners package. That computes the statistics using hardware acceleration.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show statistics of training examples.", ShortName = "stat", SortOrder = 50)]
            public bool ShowTrainingStats = false;

            /// <summary>
            /// The instance of <see cref="IComputeLRTrainingStd"/> that computes the training statistics at the end of training.
            /// If you have a large number of learned training parameters(more than 500),
            /// generating the training statistics might take a few seconds.
            /// More than 1000 weights might take a few minutes. For those cases consider using the instance of <see cref="IComputeLRTrainingStd"/>
            /// present in the Microsoft.ML.HalLearners package. That computes the statistics using hardware acceleration.
            /// </summary>
            public IComputeLRTrainingStd StdComputer;
        }

        private double _posWeight;
        private LinearModelStatistics _stats;

        /// <summary>
        /// Initializes a new instance of <see cref="LogisticRegression"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name for the example weight column.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularizer term.</param>
        /// <param name="l2Weight">Weight of L2 regularizer term.</param>
        /// <param name="memorySize">Memory size for <see cref="LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public LogisticRegression(IHostEnvironment env,
            string featureColumn,
            string labelColumn,
            string weightColumn = null,
            float l1Weight = Arguments.Defaults.L1Weight,
            float l2Weight = Arguments.Defaults.L2Weight,
            float optimizationTolerance = Arguments.Defaults.OptTol,
            int memorySize = Arguments.Defaults.MemorySize,
            bool enforceNoNegativity = Arguments.Defaults.EnforceNonNegativity,
            Action<Arguments> advancedSettings = null)
            : base(env, featureColumn, TrainerUtils.MakeBoolScalarLabel(labelColumn), weightColumn, advancedSettings,
                  l1Weight, l2Weight,  optimizationTolerance, memorySize, enforceNoNegativity)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));

            _posWeight = 0;
            ShowTrainingStats = Args.ShowTrainingStats;

            if (ShowTrainingStats && Args.StdComputer == null)
                Args.StdComputer = new ComputeLRTrainingStd();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LogisticRegression"/>
        /// </summary>
        internal LogisticRegression(IHostEnvironment env, Arguments args)
            : base(env, args, TrainerUtils.MakeBoolScalarLabel(args.LabelColumn))
        {
            _posWeight = 0;
            ShowTrainingStats = Args.ShowTrainingStats;

            if (ShowTrainingStats && Args.StdComputer == null)
                Args.StdComputer = new ComputeLRTrainingStd();
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override BinaryPredictionTransformer<ParameterMixingCalibratedPredictor> MakeTransformer(ParameterMixingCalibratedPredictor model, Schema trainSchema)
            => new BinaryPredictionTransformer<ParameterMixingCalibratedPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        protected override float AccumulateOneGradient(in VBuffer<float> feat, float label, float weight,
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
            Contracts.Assert(grad.Count >= BiasCount && (grad.IsDense || grad.Indices[BiasCount - 1] == BiasCount - 1));
            grad.Values[0] += mult;

            return weight * datumLoss;
        }

        protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory cursorFactory, float loss, int numParams)
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

            if (L2Weight > 0)
            {
                // Need to subtract L2 regularization loss.
                // The bias term is not regularized.
                var regLoss = VectorUtils.NormSquared(CurrentWeights.Values, 1, CurrentWeights.Length - 1) * L2Weight;
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
            var featureColIdx = cursorFactory.Data.Schema.Feature.Index;
            var schema = cursorFactory.Data.Data.Schema;
            var featureLength = CurrentWeights.Length - BiasCount;
            var namesSpans = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(featureLength);
            if (schema.HasSlotNames(featureColIdx, featureLength))
                schema.GetMetadata(MetadataUtils.Kinds.SlotNames, featureColIdx, ref namesSpans);
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
                for (int i = 1; i < CurrentWeights.Length; i++)
                {
                    if (CurrentWeights.Values[i] != 0)
                    {
                        weightIndices[j] = i;
                        weightIndicesInvMap[i] = j++;
                    }
                }

                Contracts.Assert(j == numParams);
            }

            // Compute the standard error of coefficients.
            long hessianDimension = (long)numParams * (numParams + 1) / 2;
            if (hessianDimension > int.MaxValue)
            {
                ch.Warning("The number of parameter is too large. Cannot hold the variance-covariance matrix in memory. " +
                    "Skipping computation of standard errors and z-statistics of coefficients. Consider choosing a larger L1 regularizer" +
                    "to reduce the number of parameters.");
                _stats = new LinearModelStatistics(Host, NumGoodRows, numParams, deviance, nullDeviance);
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
            var bias = CurrentWeights.Values[0];
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

                    var values = cursor.Features.Values;
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
                        var indices = cursor.Features.Indices;
                        for (int ii = 0; ii < cursor.Features.Count; ++ii)
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

            if (Args.StdComputer == null)
                _stats = new LinearModelStatistics(Host, NumGoodRows, numParams, deviance, nullDeviance);
            else
            {
                var std = Args.StdComputer.ComputeStd(hessian, weightIndices, numParams, CurrentWeights.Count, ch, L2Weight);
                _stats = new LinearModelStatistics(Host, NumGoodRows, numParams, deviance, nullDeviance, std);
            }
        }

        protected override void ProcessPriorDistribution(float label, float weight)
        {
            if (label > 0)
                _posWeight += weight;
        }

        //Override default termination criterion MeanRelativeImprovementCriterion with
        protected override Optimizer InitializeOptimizer(IChannel ch, FloatLabelCursor.Factory cursorFactory,
            out VBuffer<float> init, out ITerminationCriterion terminationCriterion)
        {
            var opt = base.InitializeOptimizer(ch, cursorFactory, out init, out terminationCriterion);

            // MeanImprovementCriterion:
            //   Terminates when the geometrically-weighted average improvement falls below the tolerance
            //terminationCriterion = new GradientCheckingMonitor(new MeanImprovementCriterion(CmdArgs.optTol, 0.25, MaxIterations),2);
            terminationCriterion = new MeanImprovementCriterion(OptTol, (float)0.25, MaxIterations);

            return opt;
        }

        protected override VBuffer<float> InitializeWeightsFromPredictor(ParameterMixingCalibratedPredictor srcPredictor)
        {
            Contracts.AssertValue(srcPredictor);

            var pred = srcPredictor.SubPredictor as LinearBinaryPredictor;
            Contracts.AssertValue(pred);
            return InitializeWeights(pred.Weights2, new[] { pred.Bias });
        }

        protected override ParameterMixingCalibratedPredictor CreatePredictor()
        {
            // Logistic regression is naturally calibrated to
            // output probabilities when transformed using
            // the logistic function, so there is no need to
            // train a separate calibrator.
            VBuffer<float> weights = default(VBuffer<float>);
            float bias = 0;
            CurrentWeights.GetItemOrDefault(0, ref bias);
            CurrentWeights.CopyTo(ref weights, 1, CurrentWeights.Length - 1);
            return new ParameterMixingCalibratedPredictor(Host,
                new LinearBinaryPredictor(Host, in weights, bias, _stats),
                new PlattCalibrator(Host, -1, 0));
        }

        [TlcModule.EntryPoint(Name = "Trainers.LogisticRegressionBinaryClassifier",
            Desc = Summary,
            UserName = UserNameValue,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/LogisticRegression/doc.xml' path='doc/members/member[@name=""LBFGS""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/LogisticRegression/doc.xml' path='doc/members/example[@name=""LogisticRegressionBinaryClassifier""]/*' />"})]

        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLRBinary");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LogisticRegression(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }

    /// <summary>
    /// Computes the standart deviation matrix of each of the non-zero training weights, needed to calculate further the standart deviation,
    /// p-value and z-Score.
    /// If you need fast calculations, use the <see cref="IComputeLRTrainingStd"/> implementation in the Microsoft.ML.HALLearners package,
    /// which makes use of hardware acceleration.
    /// Due to the existence of regularization, an approximation is used to compute the variances of the trained linear coefficients.
    /// </summary>
    public interface IComputeLRTrainingStd
    {
        /// <summary>
        /// Computes the standart deviation matrix of each of the non-zero training weights, needed to calculate further the standart deviation,
        /// p-value and z-Score.
        /// If you need fast calculations, use the ComputeStd method from the Microsoft.ML.HALLearners package, which makes use of hardware acceleration.
        /// Due to the existence of regularization, an approximation is used to compute the variances of the trained linear coefficients.
        /// </summary>
        VBuffer<float> ComputeStd(double[] hessian, int[] weightIndices, int parametersCount, int currentWeightsCount, IChannel ch, float l2Weight);
    }

    public sealed class ComputeLRTrainingStd: IComputeLRTrainingStd
    {
        /// <summary>
        /// Computes the standart deviation matrix of each of the non-zero training weights, needed to calculate further the standart deviation,
        /// p-value and z-Score.
        /// If you need faster calculations, use the ComputeStd method from the Microsoft.ML.HALLearners package, which makes use of hardware acceleration.
        /// Due to the existence of regularization, an approximation is used to compute the variances of the trained linear coefficients.
        /// </summary>
        /// <param name="hessian"></param>
        /// <param name="weightIndices"></param>
        /// <param name="numSelectedParams"></param>
        /// <param name="currentWeightsCount"></param>
        /// <param name="ch">The <see cref="IChannel"/> used for messaging.</param>
        /// <param name="l2Weight">The L2Weight used for training. (Supply the same one that got used during training.)</param>
        public VBuffer<float> ComputeStd(double[] hessian, int[] weightIndices, int numSelectedParams, int currentWeightsCount, IChannel ch, float l2Weight)
        {
            Contracts.AssertValue(ch);
            Contracts.AssertValue(hessian, nameof(hessian));
            Contracts.AssertNonEmpty(weightIndices);
            Contracts.Assert(numSelectedParams > 0);
            Contracts.Assert(currentWeightsCount > 0);
            Contracts.Assert(l2Weight > 0);

            double[,] matrixHessian = new double[numSelectedParams, numSelectedParams];

            int hessianLength = 0;
            int dimention = numSelectedParams - 1;

            for (int row = dimention; row >= 0; row--)
            {
                for (int col = 0; col <= dimention; col++)
                {
                    if ((row + col) <= dimention)
                    {
                        if ((row + col) == dimention)
                        {
                            matrixHessian[row, col] = hessian[hessianLength];
                        }
                        else
                        {
                            matrixHessian[row, col] = hessian[hessianLength];
                            matrixHessian[dimention - col, dimention - row] = hessian[hessianLength];
                        }
                        hessianLength++;
                    }
                    else
                        continue;
                }
            }

            var h = Matrix<double>.Build.DenseOfArray(matrixHessian);
            var invers = h.Inverse();

            float[] stdErrorValues2 = new float[numSelectedParams];
            stdErrorValues2[0] = (float)Math.Sqrt(invers[0, numSelectedParams - 1]);

            for (int i = 1; i < numSelectedParams; i++)
            {
                // Initialize with inverse Hessian.
                // The diagonal of the inverse Hessian.
                stdErrorValues2[i] = (float)invers[i, numSelectedParams - i - 1];
            }

            if (l2Weight > 0)
            {
                // Iterate through all entries of inverse Hessian to make adjustment to variance.
                // A discussion on ridge regularized LR coefficient covariance matrix can be found here:
                // http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3228544/
                // http://www.inf.unibz.it/dis/teaching/DWDM/project2010/LogisticRegression.pdf
                int ioffset = 1;
                for (int iRow = 1; iRow < numSelectedParams; iRow++)
                {
                    for (int iCol = 0; iCol <= iRow; iCol++)
                    {
                        float entry = (float)invers[iRow, numSelectedParams - iCol - 1];
                        var adjustment = l2Weight * entry * entry;
                        stdErrorValues2[iRow] -= adjustment;

                        if (0 < iCol && iCol < iRow)
                            stdErrorValues2[iCol] -= adjustment;
                        ioffset++;
                    }
                }
            }

            for (int i = 1; i < numSelectedParams; i++)
                stdErrorValues2[i] = (float)Math.Sqrt(stdErrorValues2[i]);

            return new VBuffer<float>(currentWeightsCount, numSelectedParams, stdErrorValues2, weightIndices);
        }
    }
}
