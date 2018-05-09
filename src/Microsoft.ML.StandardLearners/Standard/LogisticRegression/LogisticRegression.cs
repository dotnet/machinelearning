// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
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

[assembly: LoadableClass(typeof(void), typeof(LogisticRegression), null, typeof(SignatureEntryPointModule), "LogisticRegression")]

namespace Microsoft.ML.Runtime.Learners
{
    using Mkl = Microsoft.ML.Runtime.Learners.OlsLinearRegressionTrainer.Mkl;

    public sealed partial class LogisticRegression : LbfgsTrainerBase<Float, ParameterMixingCalibratedPredictor>
    {
        public const string LoadNameValue = "LogisticRegression";
        internal const string UserNameValue = "Logistic Regression";
        internal const string ShortName = "lr";
        internal const string Summary = "Logistic Regression is a method in statistics used to predict the probability of occurrence of an event and can "
            + "be used as a classification algorithm. The algorithm predicts the probability of occurrence of an event by fitting data to a logistical function.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show statistics of training examples.", ShortName = "stat", SortOrder = 50)]
            public bool ShowTrainingStats = false;
        }

        private Double _posWeight;
        private LinearModelStatistics _stats;

        public LogisticRegression(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue, Contracts.CheckRef(args, nameof(args)).ShowTrainingStats)
        {
            _posWeight = 0;
        }

        public override bool NeedCalibration { get { return false; } }

        public override PredictionKind PredictionKind { get { return PredictionKind.BinaryClassification; } }

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        protected override Float AccumulateOneGradient(ref VBuffer<Float> feat, Float label, Float weight,
            ref VBuffer<Float> x, ref VBuffer<Float> grad, ref Float[] scratch)
        {
            Float bias = 0;
            x.GetItemOrDefault(0, ref bias);
            Float score = bias + VectorUtils.DotProductWithOffset(ref x, 1, ref feat);

            Float s = score / 2;

            Float logZ = MathUtils.SoftMax(s, -s);
            Float label01 = Math.Min(1, Math.Max(label, 0));
            Float label11 = 2 * label01 - 1; //(-1..1) label
            Float modelProb1 = MathUtils.ExpSlow(s - logZ);
            Float ls = label11 * s;
            Float datumLoss = logZ - ls;
            //Float loss2 = MathUtil.SoftMax(s-l_s, -s-l_s);

            Contracts.Check(!Float.IsNaN(datumLoss), "Unexpected NaN");

            Float mult = weight * (modelProb1 - label01);
            VectorUtils.AddMultWithOffset(ref feat, mult, ref grad, 1); // Note that 0th L-BFGS weight is for bias.
            // Add bias using this strange trick that has advantage of working well for dense and sparse arrays.
            // Due to the call to EnsureBiases, we know this region is dense.
            Contracts.Assert(grad.Count >= BiasCount && (grad.IsDense || grad.Indices[BiasCount - 1] == BiasCount - 1));
            grad.Values[0] += mult;

            return weight * datumLoss;
        }

        protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory cursorFactory, Float loss, int numParams)
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
            Float deviance = (Float)(2 * loss * WeightSum);

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
                VBufferUtils.ForEachDefined(ref CurrentWeights, (ind, value) => { if (ind >= BiasCount) regLoss += Math.Abs(value); });
                deviance -= (Float)regLoss * L1Weight * 2;
            }

            ch.Info("Residual Deviance: \t{0} (on {1} degrees of freedom)", deviance, Math.Max(NumGoodRows - numParams, 0));

            // Compute null deviance, i.e., the deviance of null hypothesis. 
            // Cap the prior positive rate at 1e-15.
            Double priorPosRate = _posWeight / WeightSum;
            Contracts.Assert(0 <= priorPosRate && priorPosRate <= 1);
            Float nullDeviance = (priorPosRate <= 1e-15 || 1 - priorPosRate <= 1e-15) ?
                0f : (Float)(2 * WeightSum * MathUtils.Entropy(priorPosRate, true));
            ch.Info("Null Deviance:     \t{0} (on {1} degrees of freedom)", nullDeviance, NumGoodRows - 1);

            // Compute AIC.
            ch.Info("AIC:               \t{0}", 2 * numParams + deviance);

            // Show the coefficients statistics table.
            var featureColIdx = cursorFactory.Data.Schema.Feature.Index;
            var schema = cursorFactory.Data.Data.Schema;
            var featureLength = CurrentWeights.Length - BiasCount;
            var namesSpans = VBufferUtils.CreateEmpty<DvText>(featureLength);
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
            // E.g., layout of indices for 4-by-4:
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
                    var score = bias + VectorUtils.DotProductWithOffset(ref CurrentWeights, 1, ref cursor.Features);
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

            // Apply Cholesky Decomposition to find the inverse of the Hessian.
            Double[] invHessian = null;
            try
            {
                // First, find the Cholesky decomposition LL' of the Hessian.
                Mkl.Pptrf(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, numParams, hessian);
                // Note that hessian is already modified at this point. It is no longer the original Hessian,
                // but instead represents the Cholesky decomposition L.
                // Also note that the following routine is supposed to consume the Cholesky decomposition L instead
                // of the original information matrix.
                Mkl.Pptri(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, numParams, hessian);
                // At this point, hessian should contain the inverse of the original Hessian matrix.
                // Swap hessian with invHessian to avoid confusion in the following context.
                Utils.Swap(ref hessian, ref invHessian);
                Contracts.Assert(hessian == null);
            }
            catch (DllNotFoundException)
            {
                throw ch.ExceptNotSupp("The MKL library (Microsoft.MachineLearning.MklImports.dll) or one of its dependencies is missing.");
            }

            Float[] stdErrorValues = new Float[numParams];
            stdErrorValues[0] = (Float)Math.Sqrt(invHessian[0]);

            for (int i = 1; i < numParams; i++)
            {
                // Initialize with inverse Hessian.
                stdErrorValues[i] = (Single)invHessian[i * (i + 1) / 2 + i];
            }

            if (L2Weight > 0)
            {
                // Iterate through all entries of inverse Hessian to make adjustment to variance.
                // A discussion on ridge regularized LR coefficient covariance matrix can be found here:
                // http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3228544/
                // http://www.inf.unibz.it/dis/teaching/DWDM/project2010/LogisticRegression.pdf
                int ioffset = 1;
                for (int iRow = 1; iRow < numParams; iRow++)
                {
                    for (int iCol = 0; iCol <= iRow; iCol++)
                    {
                        var entry = (Single)invHessian[ioffset];
                        var adjustment = -L2Weight * entry * entry;
                        stdErrorValues[iRow] -= adjustment;
                        if (0 < iCol && iCol < iRow)
                            stdErrorValues[iCol] -= adjustment;
                        ioffset++;
                    }
                }

                Contracts.Assert(ioffset == invHessian.Length);
            }

            for (int i = 1; i < numParams; i++)
                stdErrorValues[i] = (Float)Math.Sqrt(stdErrorValues[i]);

            VBuffer<Float> stdErrors = new VBuffer<Float>(CurrentWeights.Length, numParams, stdErrorValues, weightIndices);
            _stats = new LinearModelStatistics(Host, NumGoodRows, numParams, deviance, nullDeviance, ref stdErrors);
        }

        protected override void ProcessPriorDistribution(Float label, Float weight)
        {
            if (label > 0)
                _posWeight += weight;
        }

        //Override default termination criterion MeanRelativeImprovementCriterion with
        protected override Optimizer InitializeOptimizer(IChannel ch, FloatLabelCursor.Factory cursorFactory,
            out VBuffer<Float> init, out ITerminationCriterion terminationCriterion)
        {
            var opt = base.InitializeOptimizer(ch, cursorFactory, out init, out terminationCriterion);

            // MeanImprovementCriterion:
            //   Terminates when the geometrically-weighted average improvement falls below the tolerance
            //terminationCriterion = new GradientCheckingMonitor(new MeanImprovementCriterion(CmdArgs.optTol, 0.25, MaxIterations),2);
            terminationCriterion = new MeanImprovementCriterion(OptTol, (Float)0.25, MaxIterations);

            return opt;
        }

        protected override VBuffer<Float> InitializeWeightsFromPredictor(ParameterMixingCalibratedPredictor srcPredictor)
        {
            Contracts.AssertValue(srcPredictor);

            var pred = srcPredictor.SubPredictor as LinearBinaryPredictor;
            Contracts.AssertValue(pred);
            return InitializeWeights(pred.Weights2, new[] { pred.Bias });
        }

        public override ParameterMixingCalibratedPredictor CreatePredictor()
        {
            // Logistic regression is naturally calibrated to
            // output probabilities when transformed using
            // the logistic function, so there is no need to
            // train a separate calibrator.
            VBuffer<Float> weights = default(VBuffer<Float>);
            Float bias = 0;
            CurrentWeights.GetItemOrDefault(0, ref bias);
            CurrentWeights.CopyTo(ref weights, 1, CurrentWeights.Length - 1);
            return new ParameterMixingCalibratedPredictor(Host,
                new LinearBinaryPredictor(Host, ref weights, bias, _stats),
                new PlattCalibrator(Host, -1, 0));
        }

        [TlcModule.EntryPoint(Name = "Trainers.BinaryLogisticRegressor", Desc = "Train a logistic regression binary model", UserName = UserNameValue, ShortName = ShortName)]
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
}
