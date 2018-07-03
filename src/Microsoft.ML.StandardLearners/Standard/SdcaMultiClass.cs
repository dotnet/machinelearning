// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(SdcaMultiClassTrainer.Summary, typeof(SdcaMultiClassTrainer), typeof(SdcaMultiClassTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SdcaMultiClassTrainer.UserNameValue,
    SdcaMultiClassTrainer.LoadNameValue,
    SdcaMultiClassTrainer.ShortName)]

namespace Microsoft.ML.Runtime.Learners
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Float>>;

    /// <summary>
    /// SDCA linear multiclass trainer.
    /// </summary>
    public class SdcaMultiClassTrainer : SdcaTrainerBase<TVectorPredictor>, ITrainerEx
    {
        public const string LoadNameValue = "SDCAMC";
        public const string UserNameValue = "Fast Linear Multi-class Classification (SA-SDCA)";
        public const string ShortName = "sasdcamc";
        internal const string Summary = "The SDCA linear multi-class classification trainer.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportSdcaClassificationLossFactory LossFunction = new LogLossFactory();
        }

        private readonly ISupportSdcaClassificationLoss _loss;
        private readonly Arguments _args;
        private int _numClasses;

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.MultiClassClassification; }
        }

        protected override int WeightArraySize
        {
            get
            {
                Contracts.Assert(_numClasses > 0, "_numClasses should already have been initialized when this property is called.");
                return _numClasses;
            }
        }

        public SdcaMultiClassTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
            _loss = args.LossFunction.CreateComponent(env);
            base.Loss = _loss;
            NeedShuffle = args.Shuffle;
            _args = args;
        }

        public override bool NeedCalibration { get { return false; } }

        /// <inheritdoc/>
        protected override void TrainWithoutLock(IProgressChannelProvider progress, FloatLabelCursor.Factory cursorFactory, IRandom rand,
            IdToIdxLookup idToIdx, int numThreads, DualsTableBase duals, Float[] biasReg, Float[] invariants, Float lambdaNInv,
            VBuffer<Float>[] weights, Float[] biasUnreg, VBuffer<Float>[] l1IntermediateWeights, Float[] l1IntermediateBias, Float[] featureNormSquared)
        {
            Contracts.AssertValueOrNull(progress);
            Contracts.Assert(_args.L1Threshold.HasValue);
            Contracts.AssertValueOrNull(idToIdx);
            Contracts.AssertValueOrNull(invariants);
            Contracts.AssertValueOrNull(featureNormSquared);
            int weightArraySize = WeightArraySize;
            Contracts.Assert(weightArraySize == _numClasses);
            Contracts.Assert(Utils.Size(weights) == weightArraySize);
            Contracts.Assert(Utils.Size(biasReg) == weightArraySize);
            Contracts.Assert(Utils.Size(biasUnreg) == weightArraySize);

            int maxUpdateTrials = 2 * numThreads;
            var l1Threshold = _args.L1Threshold.Value;
            bool l1ThresholdZero = l1Threshold == 0;
            var lr = _args.BiasLearningRate * _args.L2Const.Value;

            var pch = progress != null ? progress.StartProgressChannel("Dual update") : null;
            using (pch)
            using (var cursor = _args.Shuffle ? cursorFactory.Create(rand) : cursorFactory.Create())
            {
                long rowCount = 0;
                if (pch != null)
                    pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, rowCount));

                Func<UInt128, long> getIndexFromId = GetIndexFromIdGetter(idToIdx);
                while (cursor.MoveNext())
                {
                    long idx = getIndexFromId(cursor.Id);
                    long dualIndexInitPos = idx * weightArraySize;
                    var features = cursor.Features;
                    var label = (int)cursor.Label;
                    Float invariant;
                    Float normSquared;
                    if (invariants != null)
                    {
                        invariant = invariants[idx];
                        Contracts.AssertValue(featureNormSquared);
                        normSquared = featureNormSquared[idx];
                    }
                    else
                    {
                        normSquared = VectorUtils.NormSquared(features);
                        if (_args.BiasLearningRate == 0)
                            normSquared += 1;

                        invariant = _loss.ComputeDualUpdateInvariant(2 * normSquared * lambdaNInv * GetInstanceWeight(cursor));
                    }

                    // The output for the label class using current weights and bias.
                    var labelOutput = WDot(ref features, ref weights[label], biasReg[label] + biasUnreg[label]);
                    var instanceWeight = GetInstanceWeight(cursor);

                    // This will be the new dual variable corresponding to the label class.
                    Float labelDual = 0;

                    // This will be used to update the weights and regularized bias corresponding to the label class.
                    Float labelPrimalUpdate = 0;

                    // This will be used to update the unregularized bias corresponding to the label class.
                    Float labelAdjustment = 0;

                    // Iterates through all classes.
                    for (int iClass = 0; iClass < _numClasses; iClass++)
                    {
                        // Skip the dual/weights/bias update for label class. Will be taken care of at the end.
                        if (iClass == label)
                            continue;

                        // Loop trials for compare-and-swap updates of duals.
                        // In general, concurrent update conflict to the same dual variable is rare 
                        // if data is shuffled.
                        for (int numTrials = 0; numTrials < maxUpdateTrials; numTrials++)
                        {
                            long dualIndex = iClass + dualIndexInitPos;
                            var dual = duals[dualIndex];
                            var output = labelOutput + labelPrimalUpdate * normSquared - WDot(ref features, ref weights[iClass], biasReg[iClass] + biasUnreg[iClass]);
                            var dualUpdate = _loss.DualUpdate(output, 1, dual, invariant, numThreads);

                            // The successive over-relaxation apporach to adjust the sum of dual variables (biasReg) to zero.
                            // Reference to details: http://stat.rutgers.edu/home/tzhang/papers/ml02_dual.pdf, pp. 16-17. 
                            var adjustment = l1ThresholdZero ? lr * biasReg[iClass] : lr * l1IntermediateBias[iClass];
                            dualUpdate -= adjustment;
                            bool success = false;
                            duals.ApplyAt(dualIndex, (long index, ref Float value) =>
                            {
                                success = Interlocked.CompareExchange(ref value, dual + dualUpdate, dual) == dual;
                            });

                            if (success)
                            {
                                // Note: dualConstraint[iClass] = lambdaNInv * (sum of duals[iClass])
                                var primalUpdate = dualUpdate * lambdaNInv * instanceWeight;
                                labelDual -= dual + dualUpdate;
                                labelPrimalUpdate += primalUpdate;
                                biasUnreg[iClass] += adjustment * lambdaNInv * instanceWeight;
                                labelAdjustment -= adjustment;

                                if (l1ThresholdZero)
                                {
                                    VectorUtils.AddMult(ref features, weights[iClass].Values, -primalUpdate);
                                    biasReg[iClass] -= primalUpdate;
                                }
                                else
                                {
                                    //Iterative shrinkage-thresholding (aka. soft-thresholding)
                                    //Update v=denseWeights as if there's no L1
                                    //Thresholding: if |v[j]| < threshold, turn off weights[j]
                                    //If not, shrink: w[j] = v[i] - sign(v[j]) * threshold
                                    l1IntermediateBias[iClass] -= primalUpdate;
                                    if (_args.BiasLearningRate == 0)
                                    {
                                        biasReg[iClass] = Math.Abs(l1IntermediateBias[iClass]) - l1Threshold > 0.0
                                        ? l1IntermediateBias[iClass] - Math.Sign(l1IntermediateBias[iClass]) * l1Threshold
                                        : 0;
                                    }

                                    if (features.IsDense)
                                        SseUtils.SdcaL1UpdateDense(-primalUpdate, features.Length, features.Values, l1Threshold, l1IntermediateWeights[iClass].Values, weights[iClass].Values);
                                    else if (features.Count > 0)
                                        SseUtils.SdcaL1UpdateSparse(-primalUpdate, features.Length, features.Values, features.Indices, features.Count, l1Threshold, l1IntermediateWeights[iClass].Values, weights[iClass].Values);
                                }

                                break;
                            }
                        }
                    }

                    // Updating with label class weights and dual variable.
                    duals[label + dualIndexInitPos] = labelDual;
                    biasUnreg[label] += labelAdjustment * lambdaNInv * instanceWeight;
                    if (l1ThresholdZero)
                    {
                        VectorUtils.AddMult(ref features, weights[label].Values, labelPrimalUpdate);
                        biasReg[label] += labelPrimalUpdate;
                    }
                    else
                    {
                        l1IntermediateBias[label] += labelPrimalUpdate;
                        var intermediateBias = l1IntermediateBias[label];
                        biasReg[label] = Math.Abs(intermediateBias) - l1Threshold > 0.0
                            ? intermediateBias - Math.Sign(intermediateBias) * l1Threshold
                            : 0;

                        if (features.IsDense)
                            SseUtils.SdcaL1UpdateDense(labelPrimalUpdate, features.Length, features.Values, l1Threshold, l1IntermediateWeights[label].Values, weights[label].Values);
                        else if (features.Count > 0)
                            SseUtils.SdcaL1UpdateSparse(labelPrimalUpdate, features.Length, features.Values, features.Indices, features.Count, l1Threshold, l1IntermediateWeights[label].Values, weights[label].Values);
                    }

                    rowCount++;
                }
            }
        }

        /// <inheritdoc/>
        protected override bool CheckConvergence(
            IProgressChannel pch,
            int iter,
            FloatLabelCursor.Factory cursorFactory,
            DualsTableBase duals,
            IdToIdxLookup idToIdx,
            VBuffer<Float>[] weights,
            VBuffer<Float>[] bestWeights,
            Float[] biasUnreg,
            Float[] bestBiasUnreg,
            Float[] biasReg,
            Float[] bestBiasReg,
            long count,
            Double[] metrics,
            ref Double bestPrimalLoss,
            ref int bestIter)
        {
            Contracts.AssertValue(weights);
            Contracts.AssertValue(duals);
            Contracts.Assert(weights.Length == _numClasses);
            Contracts.Assert(duals.Length >= _numClasses * count);
            Contracts.AssertValueOrNull(idToIdx);
            int weightArraySize = WeightArraySize;
            Contracts.Assert(weightArraySize == _numClasses);
            Contracts.Assert(Utils.Size(weights) == weightArraySize);
            Contracts.Assert(Utils.Size(biasReg) == weightArraySize);
            Contracts.Assert(Utils.Size(biasUnreg) == weightArraySize);
            Contracts.Assert(Utils.Size(metrics) == 6);
            var reportedValues = new Double?[metrics.Length + 1];
            reportedValues[metrics.Length] = iter;
            var lossSum = new CompensatedSum();
            var dualLossSum = new CompensatedSum();

            using (var cursor = cursorFactory.Create())
            {
                long row = 0;
                Func<UInt128, long, long> getIndexFromIdAndRow = GetIndexFromIdAndRowGetter(idToIdx);
                // Iterates through data to compute loss function.
                while (cursor.MoveNext())
                {
                    var instanceWeight = GetInstanceWeight(cursor);
                    var features = cursor.Features;
                    var label = (int)cursor.Label;
                    var labelOutput = WDot(ref features, ref weights[label], biasReg[label] + biasUnreg[label]);
                    Double subLoss = 0;
                    Double subDualLoss = 0;
                    long idx = getIndexFromIdAndRow(cursor.Id, row);
                    long dualIndex = idx * _numClasses;
                    for (int iClass = 0; iClass < _numClasses; iClass++)
                    {
                        if (iClass == label)
                        {
                            dualIndex++;
                            continue;
                        }

                        var currentClassOutput = WDot(ref features, ref weights[iClass], biasReg[iClass] + biasUnreg[iClass]);
                        subLoss += _loss.Loss(labelOutput - currentClassOutput, 1);
                        Contracts.Assert(dualIndex == iClass + idx * _numClasses);
                        var dual = duals[dualIndex++];
                        subDualLoss += _loss.DualLoss(1, dual);
                    }

                    lossSum.Add(subLoss * instanceWeight);
                    dualLossSum.Add(subDualLoss * instanceWeight);

                    row++;
                }
                Host.Assert(idToIdx == null || row * WeightArraySize == duals.Length);
            }

            Contracts.Assert(_args.L2Const.HasValue);
            Contracts.Assert(_args.L1Threshold.HasValue);
            Double l2Const = _args.L2Const.Value;
            Double l1Threshold = _args.L1Threshold.Value;

            Double weightsL1Norm = 0;
            Double weightsL2NormSquared = 0;
            Double biasRegularizationAdjustment = 0;
            for (int iClass = 0; iClass < _numClasses; iClass++)
            {
                weightsL1Norm += VectorUtils.L1Norm(ref weights[iClass]) + Math.Abs(biasReg[iClass]);
                weightsL2NormSquared += VectorUtils.NormSquared(weights[iClass]) + biasReg[iClass] * biasReg[iClass];
                biasRegularizationAdjustment += biasReg[iClass] * biasUnreg[iClass];
            }

            Double l1Regularizer = _args.L1Threshold.Value * l2Const * weightsL1Norm;
            var l2Regularizer = l2Const * weightsL2NormSquared * 0.5;

            var newLoss = lossSum.Sum / count + l2Regularizer + l1Regularizer;
            var newDualLoss = dualLossSum.Sum / count - l2Regularizer - l2Const * biasRegularizationAdjustment;
            var dualityGap = newLoss - newDualLoss;

            metrics[(int)MetricKind.Loss] = newLoss;
            metrics[(int)MetricKind.DualLoss] = newDualLoss;
            metrics[(int)MetricKind.DualityGap] = dualityGap;
            metrics[(int)MetricKind.BiasUnreg] = biasUnreg[0];
            metrics[(int)MetricKind.BiasReg] = biasReg[0];
            metrics[(int)MetricKind.L1Sparsity] = _args.L1Threshold == 0 ? 1 : (Double)weights.Sum(weight => weight.Values.Count(w => w != 0)) / (_numClasses * NumFeatures);

            bool converged = dualityGap / newLoss < _args.ConvergenceTolerance;

            if (metrics[(int)MetricKind.Loss] < bestPrimalLoss)
            {
                for (int iClass = 0; iClass < _numClasses; iClass++)
                {
                    // Maintain a copy of weights and bias with best primal loss thus far. 
                    // This is some extra work and uses extra memory, but it seems worth doing it.
                    // REVIEW: Sparsify bestWeights?
                    weights[iClass].CopyTo(ref bestWeights[iClass]);
                    bestBiasReg[iClass] = biasReg[iClass];
                    bestBiasUnreg[iClass] = biasUnreg[iClass];
                }

                bestPrimalLoss = metrics[(int)MetricKind.Loss];
                bestIter = iter;
            }

            for (int i = 0; i < metrics.Length; i++)
                reportedValues[i] = metrics[i];
            if (pch != null)
                pch.Checkpoint(reportedValues);

            return converged;
        }

        public override TVectorPredictor CreatePredictor()
        {
            return new MulticlassLogisticRegressionPredictor(Host, Weights, Bias, _numClasses, NumFeatures, null, stats: null);
        }

        protected override void CheckLabel(RoleMappedData examples)
        {
            examples.CheckMultiClassLabel(out _numClasses);
        }

        protected override Float[] InitializeFeatureNormSquared(int length)
        {
            Contracts.Assert(0 < length & length <= Utils.ArrayMaxSize);
            return new Float[length];
        }

        protected override Float GetInstanceWeight(FloatLabelCursor cursor)
        {
            return cursor.Weight;
        }
    }

    /// <summary>
    /// A component to train an SDCA model.
    /// </summary>
    public static partial class Sdca
    {
        [TlcModule.EntryPoint(Name = "Trainers.StochasticDualCoordinateAscentClassifier", 
            Desc = SdcaMultiClassTrainer.Summary, 
            Remarks = SdcaMultiClassTrainer.Remarks,
            UserName = SdcaMultiClassTrainer.UserNameValue, 
            ShortName = SdcaMultiClassTrainer.ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClass(IHostEnvironment env, SdcaMultiClassTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSDCA");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<SdcaMultiClassTrainer.Arguments, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new SdcaMultiClassTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }
}