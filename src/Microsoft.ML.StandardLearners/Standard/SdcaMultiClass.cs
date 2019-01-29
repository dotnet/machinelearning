// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Threading;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Learners;
using Microsoft.ML.Numeric;
using Microsoft.ML.Trainers;
using Microsoft.ML.Training;
using Float = System.Single;

[assembly: LoadableClass(SdcaMultiClassTrainer.Summary, typeof(SdcaMultiClassTrainer), typeof(SdcaMultiClassTrainer.Options),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SdcaMultiClassTrainer.UserNameValue,
    SdcaMultiClassTrainer.LoadNameValue,
    SdcaMultiClassTrainer.ShortName)]

namespace Microsoft.ML.Trainers
{
    // SDCA linear multiclass trainer.
    /// <include file='doc.xml' path='doc/members/member[@name="SDCA"]/*' />
    public class SdcaMultiClassTrainer : SdcaTrainerBase<SdcaMultiClassTrainer.Options, MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters>, MulticlassLogisticRegressionModelParameters>
    {
        public const string LoadNameValue = "SDCAMC";
        public const string UserNameValue = "Fast Linear Multi-class Classification (SA-SDCA)";
        public const string ShortName = "sasdcamc";
        internal const string Summary = "The SDCA linear multi-class classification trainer.";

        public sealed class Options : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportSdcaClassificationLossFactory LossFunction = new LogLossFactory();
        }

        private readonly ISupportSdcaClassificationLoss _loss;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        /// <summary>
        /// Initializes a new instance of <see cref="SdcaMultiClassTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The label, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        internal SdcaMultiClassTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            ISupportSdcaClassificationLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
             : base(env, featureColumn, TrainerUtils.MakeU4ScalarColumn(labelColumn), TrainerUtils.MakeR4ScalarWeightColumn(weights),
                   l2Const, l1Threshold, maxIterations)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            _loss = loss ?? Args.LossFunction.CreateComponent(env);
            Loss = _loss;
        }

        internal SdcaMultiClassTrainer(IHostEnvironment env, Options options,
            string featureColumn, string labelColumn, string weightColumn = null)
            : base(env, options, TrainerUtils.MakeU4ScalarColumn(labelColumn), TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
            Host.CheckValue(labelColumn, nameof(labelColumn));
            Host.CheckValue(featureColumn, nameof(featureColumn));

            _loss = options.LossFunction.CreateComponent(env);
            Loss = _loss;
        }

        internal SdcaMultiClassTrainer(IHostEnvironment env, Options options)
            : this(env, options, options.FeatureColumn, options.LabelColumn)
        {
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var metadata = new SchemaShape(labelCol.Metadata.Where(x => x.Name == MetadataUtils.Kinds.KeyValues)
                .Concat(MetadataUtils.GetTrainerOutputMetadata()));
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(MetadataUtils.MetadataForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, metadata)
            };
        }

        protected override void CheckLabelCompatible(SchemaShape.Column labelCol)
        {
            Contracts.Assert(labelCol.IsValid);

            Action error =
                () => throw Host.ExceptSchemaMismatch(nameof(labelCol), RoleMappedSchema.ColumnRole.Label.Value, labelCol.Name, "R8, R4 or a Key", labelCol.GetTypeString());

            if (labelCol.Kind != SchemaShape.Column.VectorKind.Scalar)
                error();
            if (!labelCol.IsKey && labelCol.ItemType != NumberType.R4 && labelCol.ItemType != NumberType.R8)
                error();
        }

        /// <inheritdoc/>
        private protected override void TrainWithoutLock(IProgressChannelProvider progress, FloatLabelCursor.Factory cursorFactory, Random rand,
            IdToIdxLookup idToIdx, int numThreads, DualsTableBase duals, Float[] biasReg, Float[] invariants, Float lambdaNInv,
            VBuffer<Float>[] weights, Float[] biasUnreg, VBuffer<Float>[] l1IntermediateWeights, Float[] l1IntermediateBias, Float[] featureNormSquared)
        {
            Contracts.AssertValueOrNull(progress);
            Contracts.Assert(Args.L1Threshold.HasValue);
            Contracts.AssertValueOrNull(idToIdx);
            Contracts.AssertValueOrNull(invariants);
            Contracts.AssertValueOrNull(featureNormSquared);
            int numClasses = Utils.Size(weights);
            Contracts.Assert(Utils.Size(biasReg) == numClasses);
            Contracts.Assert(Utils.Size(biasUnreg) == numClasses);

            int maxUpdateTrials = 2 * numThreads;
            var l1Threshold = Args.L1Threshold.Value;
            bool l1ThresholdZero = l1Threshold == 0;
            var lr = Args.BiasLearningRate * Args.L2Const.Value;

            var pch = progress != null ? progress.StartProgressChannel("Dual update") : null;
            using (pch)
            using (var cursor = Args.Shuffle ? cursorFactory.Create(rand) : cursorFactory.Create())
            {
                long rowCount = 0;
                if (pch != null)
                    pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, rowCount));

                Func<RowId, long> getIndexFromId = GetIndexFromIdGetter(idToIdx, biasReg.Length);
                while (cursor.MoveNext())
                {
                    long idx = getIndexFromId(cursor.Id);
                    long dualIndexInitPos = idx * numClasses;
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
                        normSquared = VectorUtils.NormSquared(in features);
                        if (Args.BiasLearningRate == 0)
                            normSquared += 1;

                        invariant = _loss.ComputeDualUpdateInvariant(2 * normSquared * lambdaNInv * GetInstanceWeight(cursor));
                    }

                    // The output for the label class using current weights and bias.
                    var labelOutput = WDot(in features, in weights[label], biasReg[label] + biasUnreg[label]);
                    var instanceWeight = GetInstanceWeight(cursor);

                    // This will be the new dual variable corresponding to the label class.
                    Float labelDual = 0;

                    // This will be used to update the weights and regularized bias corresponding to the label class.
                    Float labelPrimalUpdate = 0;

                    // This will be used to update the unregularized bias corresponding to the label class.
                    Float labelAdjustment = 0;

                    // Iterates through all classes.
                    for (int iClass = 0; iClass < numClasses; iClass++)
                    {
                        // Skip the dual/weights/bias update for label class. Will be taken care of at the end.
                        if (iClass == label)
                            continue;

                        var weightsEditor = VBufferEditor.CreateFromBuffer(ref weights[iClass]);
                        var l1IntermediateWeightsEditor =
                            !l1ThresholdZero ? VBufferEditor.CreateFromBuffer(ref l1IntermediateWeights[iClass]) :
                            default;

                        // Loop trials for compare-and-swap updates of duals.
                        // In general, concurrent update conflict to the same dual variable is rare
                        // if data is shuffled.
                        for (int numTrials = 0; numTrials < maxUpdateTrials; numTrials++)
                        {
                            long dualIndex = iClass + dualIndexInitPos;
                            var dual = duals[dualIndex];
                            var output = labelOutput + labelPrimalUpdate * normSquared - WDot(in features, in weights[iClass], biasReg[iClass] + biasUnreg[iClass]);
                            var dualUpdate = _loss.DualUpdate(output, 1, dual, invariant, numThreads);

                            // The successive over-relaxation apporach to adjust the sum of dual variables (biasReg) to zero.
                            // Reference to details: http://stat.rutgers.edu/home/tzhang/papers/ml02_dual.pdf, pp. 16-17.
                            var adjustment = l1ThresholdZero ? lr * biasReg[iClass] : lr * l1IntermediateBias[iClass];
                            dualUpdate -= adjustment;
                            bool success = false;
                            duals.ApplyAt(dualIndex, (long index, ref Float value) =>
                                success = Interlocked.CompareExchange(ref value, dual + dualUpdate, dual) == dual);

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
                                    VectorUtils.AddMult(in features, weightsEditor.Values, -primalUpdate);
                                    biasReg[iClass] -= primalUpdate;
                                }
                                else
                                {
                                    //Iterative shrinkage-thresholding (aka. soft-thresholding)
                                    //Update v=denseWeights as if there's no L1
                                    //Thresholding: if |v[j]| < threshold, turn off weights[j]
                                    //If not, shrink: w[j] = v[i] - sign(v[j]) * threshold
                                    l1IntermediateBias[iClass] -= primalUpdate;
                                    if (Args.BiasLearningRate == 0)
                                    {
                                        biasReg[iClass] = Math.Abs(l1IntermediateBias[iClass]) - l1Threshold > 0.0
                                        ? l1IntermediateBias[iClass] - Math.Sign(l1IntermediateBias[iClass]) * l1Threshold
                                        : 0;
                                    }

                                    var featureValues = features.GetValues();
                                    if (features.IsDense)
                                        CpuMathUtils.SdcaL1UpdateDense(-primalUpdate, featureValues.Length, featureValues, l1Threshold, l1IntermediateWeightsEditor.Values, weightsEditor.Values);
                                    else if (featureValues.Length > 0)
                                        CpuMathUtils.SdcaL1UpdateSparse(-primalUpdate, featureValues.Length, featureValues, features.GetIndices(), l1Threshold, l1IntermediateWeightsEditor.Values, weightsEditor.Values);
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
                        var weightsEditor = VBufferEditor.CreateFromBuffer(ref weights[label]);
                        VectorUtils.AddMult(in features, weightsEditor.Values, labelPrimalUpdate);
                        biasReg[label] += labelPrimalUpdate;
                    }
                    else
                    {
                        l1IntermediateBias[label] += labelPrimalUpdate;
                        var intermediateBias = l1IntermediateBias[label];
                        biasReg[label] = Math.Abs(intermediateBias) - l1Threshold > 0.0
                            ? intermediateBias - Math.Sign(intermediateBias) * l1Threshold
                            : 0;

                        var weightsEditor = VBufferEditor.CreateFromBuffer(ref weights[label]);
                        var l1IntermediateWeightsEditor = VBufferEditor.CreateFromBuffer(ref l1IntermediateWeights[label]);
                        var featureValues = features.GetValues();
                        if (features.IsDense)
                            CpuMathUtils.SdcaL1UpdateDense(labelPrimalUpdate, featureValues.Length, featureValues, l1Threshold, l1IntermediateWeightsEditor.Values, weightsEditor.Values);
                        else if (featureValues.Length > 0)
                            CpuMathUtils.SdcaL1UpdateSparse(labelPrimalUpdate, featureValues.Length, featureValues, features.GetIndices(), l1Threshold, l1IntermediateWeightsEditor.Values, weightsEditor.Values);
                    }

                    rowCount++;
                }
            }
        }

        /// <inheritdoc/>
        private protected override bool CheckConvergence(
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
            int numClasses = weights.Length;
            Contracts.Assert(duals.Length >= numClasses * count);
            Contracts.AssertValueOrNull(idToIdx);
            Contracts.Assert(Utils.Size(weights) == numClasses);
            Contracts.Assert(Utils.Size(biasReg) == numClasses);
            Contracts.Assert(Utils.Size(biasUnreg) == numClasses);
            Contracts.Assert(Utils.Size(metrics) == 6);
            var reportedValues = new Double?[metrics.Length + 1];
            reportedValues[metrics.Length] = iter;
            var lossSum = new CompensatedSum();
            var dualLossSum = new CompensatedSum();
            int numFeatures = weights[0].Length;

            using (var cursor = cursorFactory.Create())
            {
                long row = 0;
                Func<RowId, long, long> getIndexFromIdAndRow = GetIndexFromIdAndRowGetter(idToIdx, biasReg.Length);
                // Iterates through data to compute loss function.
                while (cursor.MoveNext())
                {
                    var instanceWeight = GetInstanceWeight(cursor);
                    var features = cursor.Features;
                    var label = (int)cursor.Label;
                    var labelOutput = WDot(in features, in weights[label], biasReg[label] + biasUnreg[label]);
                    Double subLoss = 0;
                    Double subDualLoss = 0;
                    long idx = getIndexFromIdAndRow(cursor.Id, row);
                    long dualIndex = idx * numClasses;
                    for (int iClass = 0; iClass < numClasses; iClass++)
                    {
                        if (iClass == label)
                        {
                            dualIndex++;
                            continue;
                        }

                        var currentClassOutput = WDot(in features, in weights[iClass], biasReg[iClass] + biasUnreg[iClass]);
                        subLoss += _loss.Loss(labelOutput - currentClassOutput, 1);
                        Contracts.Assert(dualIndex == iClass + idx * numClasses);
                        var dual = duals[dualIndex++];
                        subDualLoss += _loss.DualLoss(1, dual);
                    }

                    lossSum.Add(subLoss * instanceWeight);
                    dualLossSum.Add(subDualLoss * instanceWeight);

                    row++;
                }
                Host.Assert(idToIdx == null || row * numClasses == duals.Length);
            }

            Contracts.Assert(Args.L2Const.HasValue);
            Contracts.Assert(Args.L1Threshold.HasValue);
            Double l2Const = Args.L2Const.Value;
            Double l1Threshold = Args.L1Threshold.Value;

            Double weightsL1Norm = 0;
            Double weightsL2NormSquared = 0;
            Double biasRegularizationAdjustment = 0;
            for (int iClass = 0; iClass < numClasses; iClass++)
            {
                weightsL1Norm += VectorUtils.L1Norm(in weights[iClass]) + Math.Abs(biasReg[iClass]);
                weightsL2NormSquared += VectorUtils.NormSquared(weights[iClass]) + biasReg[iClass] * biasReg[iClass];
                biasRegularizationAdjustment += biasReg[iClass] * biasUnreg[iClass];
            }

            Double l1Regularizer = Args.L1Threshold.Value * l2Const * weightsL1Norm;
            var l2Regularizer = l2Const * weightsL2NormSquared * 0.5;

            var newLoss = lossSum.Sum / count + l2Regularizer + l1Regularizer;
            var newDualLoss = dualLossSum.Sum / count - l2Regularizer - l2Const * biasRegularizationAdjustment;
            var dualityGap = newLoss - newDualLoss;

            metrics[(int)MetricKind.Loss] = newLoss;
            metrics[(int)MetricKind.DualLoss] = newDualLoss;
            metrics[(int)MetricKind.DualityGap] = dualityGap;
            metrics[(int)MetricKind.BiasUnreg] = biasUnreg[0];
            metrics[(int)MetricKind.BiasReg] = biasReg[0];
            metrics[(int)MetricKind.L1Sparsity] = Args.L1Threshold == 0 ? 1 : weights.Sum(
                weight => weight.GetValues().Count(w => w != 0)) / (numClasses * numFeatures);

            bool converged = dualityGap / newLoss < Args.ConvergenceTolerance;

            if (metrics[(int)MetricKind.Loss] < bestPrimalLoss)
            {
                for (int iClass = 0; iClass < numClasses; iClass++)
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

        protected override MulticlassLogisticRegressionModelParameters CreatePredictor(VBuffer<Float>[] weights, Float[] bias)
        {
            Host.CheckValue(weights, nameof(weights));
            Host.CheckValue(bias, nameof(bias));
            Host.CheckParam(weights.Length > 0, nameof(weights));
            Host.CheckParam(weights.Length == bias.Length, nameof(weights));

            return new MulticlassLogisticRegressionModelParameters(Host, weights, bias, bias.Length, weights[0].Length, null, stats: null);
        }

        private protected override void CheckLabel(RoleMappedData examples, out int weightSetCount)
        {
            examples.CheckMultiClassLabel(out weightSetCount);
        }

        protected override Float[] InitializeFeatureNormSquared(int length)
        {
            Contracts.Assert(0 < length & length <= Utils.ArrayMaxSize);
            return new Float[length];
        }

        private protected override Float GetInstanceWeight(FloatLabelCursor cursor)
        {
            return cursor.Weight;
        }

        protected override MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters> MakeTransformer(MulticlassLogisticRegressionModelParameters model, Schema trainSchema)
            => new MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);
    }

    /// <summary>
    /// The Entry Point for SDCA multiclass.
    /// </summary>
    internal static partial class Sdca
    {
        [TlcModule.EntryPoint(Name = "Trainers.StochasticDualCoordinateAscentClassifier",
            Desc = SdcaMultiClassTrainer.Summary,
            UserName = SdcaMultiClassTrainer.UserNameValue,
            ShortName = SdcaMultiClassTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/doc.xml' path='doc/members/member[@name=""SDCA""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/doc.xml' path='doc/members/example[@name=""StochasticDualCoordinateAscentClassifier""]/*' />" })]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClass(IHostEnvironment env, SdcaMultiClassTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSDCA");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<SdcaMultiClassTrainer.Options, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new SdcaMultiClassTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }
}