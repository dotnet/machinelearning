// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(SdcaMaximumEntropyMulticlassTrainer.Summary, typeof(SdcaMaximumEntropyMulticlassTrainer), typeof(SdcaMaximumEntropyMulticlassTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SdcaMaximumEntropyMulticlassTrainer.UserNameValue,
    SdcaMaximumEntropyMulticlassTrainer.LoadNameValue,
    SdcaMaximumEntropyMulticlassTrainer.ShortName)]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> to predict a target using a linear multiclass classifier model trained with a coordinate descent method.
    /// Depending on the used loss function, the trained model can be, for example, maximum entropy classifier or multi-class support vector machine.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer for maximum entropy classifier, use [SdcaMaximumEntropy](xref:Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,System.Nullable{System.Single},System.Nullable{System.Single},System.Nullable{System.Int32})) or
    /// [SdcaMaximumEntropy(Options)](xref:Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options)).
    /// To create this trainer for a [loss function](xref:Microsoft.ML.Trainers.ISupportSdcaClassificationLoss) (such as support vector machine's [hinge loss](xref:Microsoft.ML.Trainers.HingeLoss)) of your choice,
    /// use [SdcaNonCalibrated](xref:Microsoft.ML.StandardTrainersCatalog.SdcaNonCalibrated(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,Microsoft.ML.Trainers.ISupportSdcaClassificationLoss,System.Nullable{System.Single},System.Nullable{System.Single},System.Nullable{System.Int32})) or
    /// [SdcaNonCalibrated(Options)](xref:Microsoft.ML.StandardTrainersCatalog.SdcaNonCalibrated(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-multiclass-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | Yes |
    ///
    /// ### Scoring Function
    /// This trains linear model to solve multiclass classification problems.
    /// Assume that the number of classes is $m$ and number of features is $n$.
    /// It assigns the $c$-th class a coefficient vector $\textbf{w}_c \in {\mathbb R}^n$ and a bias $b_c \in {\mathbb R}$, for $c=1,\dots,m$.
    /// Given a feature vector $\textbf{x} \in {\mathbb R}^n$, the $c$-th class's score would be $\hat{y}^c = \textbf{w}_c^T \textbf{x} + b_c$.
    /// If $\textbf{x}$ belongs to class $c$, then $\hat{y}^c$ should be much larger than 0.
    /// In contrast, a $\hat{y}^c$ much smaller than 0 means the desired label should not be $c$.
    ///
    /// If and only if the trained model is a maximum entropy classifier, you can interpret the output score vector as the predicted class probabilities because [softmax function](https://en.wikipedia.org/wiki/Softmax_function) may be applied to post-process all classes' scores.
    /// More specifically, the probability of $\textbf{x}$ belonging to class $c$ is computed by $\tilde{P}( c | \textbf{x} ) = \frac{ e^{\hat{y}^c} }{ \sum_{c' = 1}^m e^{\hat{y}^{c'}} }$ and store at the $c$-th element in the score vector.
    /// In other cases, the output score vector is just $[\hat{y}^1, \dots, \hat{y}^m]$.
    ///
    /// ### Training Algorithm Details
    /// The optimization algorithm is an extension of [a coordinate descent method](http://jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf)
    /// following a similar path proposed in an earlier [paper](https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf).
    /// It is usually much faster than [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) and
    /// [truncated Newton methods](https://en.wikipedia.org/wiki/Truncated_Newton_method) for large-scale and sparse data sets.
    ///
    /// [!include[regularization](~/../docs/samples/docs/api-reference/regularization-l1-l2.md)]
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(MulticlassClassificationCatalog.MulticlassClassificationTrainers, SdcaMaximumEntropyMulticlassTrainer.Options)"/>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(MulticlassClassificationCatalog.MulticlassClassificationTrainers, string, string, string, float?, float?, int?)"/>
    /// <seealso cref="Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options"/>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaNonCalibrated(MulticlassClassificationCatalog.MulticlassClassificationTrainers, SdcaNonCalibratedMulticlassTrainer.Options)"/>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaNonCalibrated(MulticlassClassificationCatalog.MulticlassClassificationTrainers, string, string, string, ISupportSdcaClassificationLoss, float?, float?, int?)"/>
    /// <seealso cref="Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options"/>
    public abstract class SdcaMulticlassTrainerBase<TModel> : SdcaTrainerBase<SdcaMulticlassTrainerBase<TModel>.MulticlassOptions, MulticlassPredictionTransformer<TModel>, TModel>
        where TModel : class
    {
        internal const string LoadNameValue = "SDCAMC";
        internal const string UserNameValue = "Fast Linear Multi-class Classification (SA-SDCA)";
        internal const string ShortName = "sasdcamc";
        internal const string Summary = "The SDCA linear multi-class classification trainer.";

        /// <summary>
        /// Options for the <see cref="SdcaMulticlassTrainerBase{TModel}"/>.
        /// </summary>
        public class MulticlassOptions : OptionsBase
        {
            /// <summary>
            /// The custom <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a>.
            /// </summary>
            /// <value>
            /// If unspecified, <see cref="LogLoss"/> will be used.
            /// </value>
            [Argument(ArgumentType.Multiple, Name = "LossFunction", HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            internal ISupportSdcaClassificationLossFactory LossFunctionFactory = new LogLossFactory();

            /// <summary>
            /// Internal state of <see cref="SdcaNonCalibratedMulticlassTrainer.Options.Loss"/> or storage of
            /// a customized loss passed in. <see cref="SdcaMaximumEntropyMulticlassTrainer.Options"/> cannot set this field because its
            /// loss function is always <see cref="LogLoss"/>. In addition, <see cref="InternalLoss"/> and <see cref="LogLossFactory"/> are
            /// the two fields used to determined the actual loss function inside the training framework of <see cref="SdcaMulticlassTrainerBase{TModel}"/>.
            /// </summary>
            internal ISupportSdcaClassificationLoss InternalLoss;
        }

        private readonly ISupportSdcaClassificationLoss _loss;

        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        /// <summary>
        /// Initializes a new instance of <see cref="SdcaMulticlassTrainerBase{TModel}"/>.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The label, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        internal SdcaMulticlassTrainerBase(IHostEnvironment env,
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
            _loss = loss ?? SdcaTrainerOptions.InternalLoss ?? SdcaTrainerOptions.LossFunctionFactory.CreateComponent(env);
            Loss = _loss;
        }

        internal SdcaMulticlassTrainerBase(IHostEnvironment env, MulticlassOptions options,
            string featureColumn, string labelColumn, string weightColumn = null)
            : base(env, options, TrainerUtils.MakeU4ScalarColumn(labelColumn), TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
            Host.CheckValue(labelColumn, nameof(labelColumn));
            Host.CheckValue(featureColumn, nameof(featureColumn));

            _loss = options.InternalLoss ?? options.LossFunctionFactory.CreateComponent(env);
            Loss = _loss;
        }

        internal SdcaMulticlassTrainerBase(IHostEnvironment env, MulticlassOptions options)
            : this(env, options, options.FeatureColumnName, options.LabelColumnName, options.ExampleWeightColumnName)
        {
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var metadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, metadata)
            };
        }

        /// <inheritdoc/>
        private protected override void TrainWithoutLock(IProgressChannelProvider progress, FloatLabelCursor.Factory cursorFactory, Random rand,
            IdToIdxLookup idToIdx, int numThreads, DualsTableBase duals, float[] biasReg, float[] invariants, float lambdaNInv,
            VBuffer<float>[] weights, float[] biasUnreg, VBuffer<float>[] l1IntermediateWeights, float[] l1IntermediateBias, float[] featureNormSquared)
        {
            Contracts.AssertValueOrNull(progress);
            Contracts.Assert(SdcaTrainerOptions.L1Regularization.HasValue);
            Contracts.AssertValueOrNull(idToIdx);
            Contracts.AssertValueOrNull(invariants);
            Contracts.AssertValueOrNull(featureNormSquared);
            int numClasses = Utils.Size(weights);
            Contracts.Assert(Utils.Size(biasReg) == numClasses);
            Contracts.Assert(Utils.Size(biasUnreg) == numClasses);

            int maxUpdateTrials = 2 * numThreads;
            var l1Threshold = SdcaTrainerOptions.L1Regularization.Value;
            bool l1ThresholdZero = l1Threshold == 0;
            var lr = SdcaTrainerOptions.BiasLearningRate * SdcaTrainerOptions.L2Regularization.Value;

            var pch = progress != null ? progress.StartProgressChannel("Dual update") : null;
            using (pch)
            using (var cursor = SdcaTrainerOptions.Shuffle ? cursorFactory.Create(rand) : cursorFactory.Create())
            {
                long rowCount = 0;
                if (pch != null)
                    pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, rowCount));

                Func<DataViewRowId, long> getIndexFromId = GetIndexFromIdGetter(idToIdx, biasReg.Length);
                while (cursor.MoveNext())
                {
                    Host.CheckAlive();
                    long idx = getIndexFromId(cursor.Id);
                    long dualIndexInitPos = idx * numClasses;
                    var features = cursor.Features;
                    var label = (int)cursor.Label;
                    float invariant;
                    float normSquared;
                    if (invariants != null)
                    {
                        invariant = invariants[idx];
                        Contracts.AssertValue(featureNormSquared);
                        normSquared = featureNormSquared[idx];
                    }
                    else
                    {
                        normSquared = VectorUtils.NormSquared(in features);
                        if (SdcaTrainerOptions.BiasLearningRate == 0)
                            normSquared += 1;

                        invariant = _loss.ComputeDualUpdateInvariant(2 * normSquared * lambdaNInv * GetInstanceWeight(cursor));
                    }

                    // The output for the label class using current weights and bias.
                    var labelOutput = WDot(in features, in weights[label], biasReg[label] + biasUnreg[label]);
                    var instanceWeight = GetInstanceWeight(cursor);

                    // This will be the new dual variable corresponding to the label class.
                    float labelDual = 0;

                    // This will be used to update the weights and regularized bias corresponding to the label class.
                    float labelPrimalUpdate = 0;

                    // This will be used to update the unregularized bias corresponding to the label class.
                    float labelAdjustment = 0;

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

                            // The successive over-relaxation approach to adjust the sum of dual variables (biasReg) to zero.
                            // Reference to details: http://stat.rutgers.edu/home/tzhang/papers/ml02_dual.pdf, pp. 16-17.
                            var adjustment = l1ThresholdZero ? lr * biasReg[iClass] : lr * l1IntermediateBias[iClass];
                            dualUpdate -= adjustment;
                            bool success = false;
                            duals.ApplyAt(dualIndex, (long index, ref float value) =>
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
                                    if (SdcaTrainerOptions.BiasLearningRate == 0)
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
            VBuffer<float>[] weights,
            VBuffer<float>[] bestWeights,
            float[] biasUnreg,
            float[] bestBiasUnreg,
            float[] biasReg,
            float[] bestBiasReg,
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
                Func<DataViewRowId, long, long> getIndexFromIdAndRow = GetIndexFromIdAndRowGetter(idToIdx, biasReg.Length);
                // Iterates through data to compute loss function.
                while (cursor.MoveNext())
                {
                    Host.CheckAlive();
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

            Contracts.Assert(SdcaTrainerOptions.L2Regularization.HasValue);
            Contracts.Assert(SdcaTrainerOptions.L1Regularization.HasValue);
            Double l2Const = SdcaTrainerOptions.L2Regularization.Value;
            Double l1Threshold = SdcaTrainerOptions.L1Regularization.Value;

            Double weightsL1Norm = 0;
            Double weightsL2NormSquared = 0;
            Double biasRegularizationAdjustment = 0;
            for (int iClass = 0; iClass < numClasses; iClass++)
            {
                weightsL1Norm += VectorUtils.L1Norm(in weights[iClass]) + Math.Abs(biasReg[iClass]);
                weightsL2NormSquared += VectorUtils.NormSquared(weights[iClass]) + biasReg[iClass] * biasReg[iClass];
                biasRegularizationAdjustment += biasReg[iClass] * biasUnreg[iClass];
            }

            Double l1Regularizer = SdcaTrainerOptions.L1Regularization.Value * l2Const * weightsL1Norm;
            var l2Regularizer = l2Const * weightsL2NormSquared * 0.5;

            var newLoss = lossSum.Sum / count + l2Regularizer + l1Regularizer;
            var newDualLoss = dualLossSum.Sum / count - l2Regularizer - l2Const * biasRegularizationAdjustment;
            var dualityGap = newLoss - newDualLoss;

            metrics[(int)MetricKind.Loss] = newLoss;
            metrics[(int)MetricKind.DualLoss] = newDualLoss;
            metrics[(int)MetricKind.DualityGap] = dualityGap;
            metrics[(int)MetricKind.BiasUnreg] = biasUnreg[0];
            metrics[(int)MetricKind.BiasReg] = biasReg[0];
            metrics[(int)MetricKind.L1Sparsity] = SdcaTrainerOptions.L1Regularization == 0 ? 1 : weights.Sum(
                weight => weight.GetValues().Count(w => w != 0)) / (numClasses * numFeatures);

            bool converged = dualityGap / newLoss < SdcaTrainerOptions.ConvergenceTolerance;

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

        private protected override void CheckLabel(RoleMappedData examples, out int weightSetCount)
        {
            examples.CheckMulticlassLabel(out weightSetCount);
        }

        private protected override float[] InitializeFeatureNormSquared(int length)
        {
            Contracts.Assert(0 < length && length <= Utils.ArrayMaxSize);
            return new float[length];
        }

        private protected override float GetInstanceWeight(FloatLabelCursor cursor)
        {
            return cursor.Weight;
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> to predict a target using a maximum entropy multiclass classifier.
    /// The trained model <see cref="MaximumEntropyModelParameters"/> produces probabilities of classes.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [SdcaMaximumEntropy](xref:Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,System.Nullable{System.Single},System.Nullable{System.Single},System.Nullable{System.Int32})) or
    /// [SdcaMaximumEntropy(Options)](xref:Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-multiclass-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | Yes |
    ///
    /// ### Scoring Function
    /// This trains a linear model to solve multiclass classification problems.
    /// Assume that the number of classes is $m$ and number of features is $n$.
    /// It assigns the $c$-th class a coefficient vector $\textbf{w}\_c \in {\mathbb R}^n$ and a bias $b_c \in {\mathbb R}$, for $c=1,\dots,m$.
    /// Given a feature vector $\textbf{x} \in {\mathbb R}^n$, the $c$-th class's score would be $\tilde{P}(c | \textbf{x}) = \frac{ e^{\hat{y}^c} }{ \sum\_{c' = 1}^m e^{\hat{y}^{c'}} }$, where $\hat{y}^c = \textbf{w}\_c^T \textbf{x} + b_c$.
    /// Note that $\tilde{P}(c | \textbf{x})$ is the probability of observing class $c$ when the feature vector is $\textbf{x}$.
    ///
    /// ### Training Algorithm Details
    /// See the documentation of [SdcaMulticlassTrainerBase](xref:Microsoft.ML.Trainers.SdcaMulticlassTrainerBase`1).
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(MulticlassClassificationCatalog.MulticlassClassificationTrainers, SdcaMaximumEntropyMulticlassTrainer.Options)"/>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(MulticlassClassificationCatalog.MulticlassClassificationTrainers, string, string, string, float?, float?, int?)"/>
    /// <seealso cref="Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options"/>
    public sealed class SdcaMaximumEntropyMulticlassTrainer : SdcaMulticlassTrainerBase<MaximumEntropyModelParameters>
    {
        /// <summary>
        /// <see cref="Options"/> for <see cref="SdcaMaximumEntropyMulticlassTrainer"/> as used in
        /// <see cref="Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(MulticlassClassificationCatalog.MulticlassClassificationTrainers, string, string, string, float?, float?, int?)"/>
        /// </summary>
        public sealed class Options : MulticlassOptions
        {
        }

        internal SdcaMaximumEntropyMulticlassTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
             : base(env, labelColumn: labelColumn, featureColumn: featureColumn, weights: weights, loss: new LogLoss(),
                   l2Const: l2Const, l1Threshold: l1Threshold, maxIterations: maxIterations)
        {
        }

        internal SdcaMaximumEntropyMulticlassTrainer(IHostEnvironment env, Options options,
            string featureColumn, string labelColumn, string weightColumn = null)
            : base(env, options: options, featureColumn: featureColumn, labelColumn: labelColumn, weightColumn: weightColumn)
        {
        }

        internal SdcaMaximumEntropyMulticlassTrainer(IHostEnvironment env, Options options)
            : base(env, options)
        {
        }

        private protected override MaximumEntropyModelParameters CreatePredictor(VBuffer<float>[] weights, float[] bias)
        {
            Host.CheckValue(weights, nameof(weights));
            Host.CheckValue(bias, nameof(bias));
            Host.CheckParam(weights.Length > 0, nameof(weights));
            Host.CheckParam(weights.Length == bias.Length, nameof(weights));

            return new MaximumEntropyModelParameters(Host, weights, bias, bias.Length, weights[0].Length, null, stats: null);
        }

        private protected override MulticlassPredictionTransformer<MaximumEntropyModelParameters> MakeTransformer(
            MaximumEntropyModelParameters model, DataViewSchema trainSchema) =>
            new MulticlassPredictionTransformer<MaximumEntropyModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);
    }

    /// <summary>
    /// The<see cref="IEstimator{TTransformer}"/> to predict a target using a linear multiclass classifier.
    /// The trained model <see cref="LinearMulticlassModelParameters"/> produces probabilities of classes.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [SdcaMaximumEntropy](xref:Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,System.Nullable{System.Single},System.Nullable{System.Single},System.Nullable{System.Int32})) or
    /// [SdcaMaximumEntropy(Options)](xref:Microsoft.ML.StandardTrainersCatalog.SdcaMaximumEntropy(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-multiclass-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | Yes |
    ///
    /// ### Scoring Function
    /// This trains a linear model to solve multiclass classification problems.
    /// Assume that the number of classes is $m$ and number of features is $n$.
    /// It assigns the $c$-th class a coefficient vector $\textbf{w}_c \in {\mathbb R}^n$ and a bias $b_c \in {\mathbb R}$, for $c=1,\dots,m$.
    /// Given a feature vector $\textbf{x} \in {\mathbb R}^n$, the $c$-th class's score would be $\hat{y}^c = \textbf{w}_c^T \textbf{x} + b_c$.
    /// Note that the $c$-th value in the output score column is just $\hat{y}^c$.
    ///
    /// ### Training Algorithm Details
    /// See the documentation of [SdcaMulticlassTrainerBase](xref:Microsoft.ML.Trainers.SdcaMulticlassTrainerBase).
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaNonCalibrated(MulticlassClassificationCatalog.MulticlassClassificationTrainers, SdcaNonCalibratedMulticlassTrainer.Options)"/>
    /// <seealso cref="Microsoft.ML.StandardTrainersCatalog.SdcaNonCalibrated(MulticlassClassificationCatalog.MulticlassClassificationTrainers, string, string, string, ISupportSdcaClassificationLoss, float?, float?, int?)"/>
    /// <seealso cref="Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options"/>
    public sealed class SdcaNonCalibratedMulticlassTrainer : SdcaMulticlassTrainerBase<LinearMulticlassModelParameters>
    {
        /// <summary>
        /// <see cref="Options"/> for <see cref="SdcaNonCalibratedMulticlassTrainer"/> as used in
        /// <see cref="Microsoft.ML.StandardTrainersCatalog.SdcaNonCalibrated(MulticlassClassificationCatalog.MulticlassClassificationTrainers, string, string, string, ISupportSdcaClassificationLoss, float?, float?, int?)"/>.
        /// </summary>
        public sealed class Options : MulticlassOptions
        {
            /// <summary>
            /// Loss function minimized by this trainer.
            /// </summary>
            /// <value>
            /// If unspecified, <see cref="LogLoss"/> will be used.
            /// </value>
            public ISupportSdcaClassificationLoss Loss
            {
                get { return InternalLoss; }
                set { InternalLoss = value; }
            }
        }

        internal SdcaNonCalibratedMulticlassTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            ISupportSdcaClassificationLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
             : base(env, labelColumn: labelColumn, featureColumn: featureColumn, weights: weights, loss: loss,
                   l2Const: l2Const, l1Threshold: l1Threshold, maxIterations: maxIterations)
        {
        }

        internal SdcaNonCalibratedMulticlassTrainer(IHostEnvironment env, Options options,
            string featureColumn, string labelColumn, string weightColumn = null)
            : base(env, options: options, featureColumn: featureColumn, labelColumn: labelColumn, weightColumn: weightColumn)
        {
        }

        internal SdcaNonCalibratedMulticlassTrainer(IHostEnvironment env, Options options)
            : base(env, options)
        {
        }

        private protected override LinearMulticlassModelParameters CreatePredictor(VBuffer<float>[] weights, float[] bias)
        {
            Host.CheckValue(weights, nameof(weights));
            Host.CheckValue(bias, nameof(bias));
            Host.CheckParam(weights.Length > 0, nameof(weights));
            Host.CheckParam(weights.Length == bias.Length, nameof(weights));

            return new LinearMulticlassModelParameters(Host, weights, bias, bias.Length, weights[0].Length, null, stats: null);
        }

        private protected override MulticlassPredictionTransformer<LinearMulticlassModelParameters> MakeTransformer(
            LinearMulticlassModelParameters model, DataViewSchema trainSchema) =>
            new MulticlassPredictionTransformer<LinearMulticlassModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);
    }

    /// <summary>
    /// The Entry Point for SDCA multiclass.
    /// </summary>
    internal static partial class Sdca
    {
        [TlcModule.EntryPoint(Name = "Trainers.StochasticDualCoordinateAscentClassifier",
            Desc = SdcaMaximumEntropyMulticlassTrainer.Summary,
            UserName = SdcaMaximumEntropyMulticlassTrainer.UserNameValue,
            ShortName = SdcaMaximumEntropyMulticlassTrainer.ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMulticlass(IHostEnvironment env, SdcaMaximumEntropyMulticlassTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSDCA");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<SdcaMaximumEntropyMulticlassTrainer.Options, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new SdcaMaximumEntropyMulticlassTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName));
        }
    }
}
