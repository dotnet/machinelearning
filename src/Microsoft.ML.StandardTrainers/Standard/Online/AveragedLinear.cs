// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

// TODO: Check if it works properly if Averaged is set to false

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Arguments class for averaged linear trainers.
    /// </summary>
    public abstract class AveragedLinearOptions : OnlineLinearOptions
    {
        /// <summary>
        /// <a href="tmpurl_lr">Learning rate</a>.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate", ShortName = "lr", SortOrder = 50)]
        [TGUI(Label = "Learning rate", SuggestedSweeps = "0.01,0.1,0.5,1.0")]
        [TlcModule.SweepableDiscreteParam("LearningRate", new object[] { 0.01, 0.1, 0.5, 1.0 })]
        public float LearningRate = AveragedDefault.LearningRate;

        /// <summary>
        /// Determine whether to decrease the <see cref="LearningRate"/> or not.
        /// </summary>
        /// <value>
        /// <see langword="true" /> to decrease the <see cref="LearningRate"/> as iterations progress; otherwise, <see langword="false" />.
        /// Default is <see langword="false" />. The learning rate will be reduced with every weight update proportional to the square root of the number of updates.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Decrease learning rate", ShortName = "decreaselr", SortOrder = 50)]
        [TGUI(Label = "Decrease Learning Rate", Description = "Decrease learning rate as iterations progress")]
        [TlcModule.SweepableDiscreteParam("DecreaseLearningRate", new object[] { false, true })]
        public bool DecreaseLearningRate = AveragedDefault.DecreaseLearningRate;

        /// <summary>
        /// Number of examples after which weights will be reset to the current average.
        /// </summary>
        /// <value>
        /// Default is <see langword="null" />, which disables this feature.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of examples after which weights will be reset to the current average", ShortName = "numreset")]
        public long? ResetWeightsAfterXExamples = null;

        /// <summary>
        /// Determines when to update averaged weights.
        /// </summary>
        /// <value>
        /// <see langword="true" /> to update averaged weights only when loss is nonzero.
        /// <see langword="false" /> to update averaged weights on every example.
        /// Default is <see langword="true" />.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Instead of updating averaged weights on every example, only update when loss is nonzero", ShortName = "lazy,DoLazyUpdates")]
        public bool LazyUpdate = true;

        /// <summary>
        /// The L2 weight for <a href='tmpurl_regularization'>regularization</a>.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization Weight", ShortName = "reg,L2RegularizerWeight", SortOrder = 50)]
        [TGUI(Label = "L2 Regularization Weight")]
        [TlcModule.SweepableFloatParam("L2RegularizerWeight", 0.0f, 0.4f)]
        public float L2Regularization = AveragedDefault.L2Regularization;

        /// <summary>
        /// Extra weight given to more recent updates.
        /// </summary>
        /// <value>
        /// Default is 0, i.e. no extra gain.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Extra weight given to more recent updates", ShortName = "rg")]
        public float RecencyGain = 0;

        /// <summary>
        /// Determines whether <see cref="RecencyGain"/> is multiplicative or additive.
        /// </summary>
        /// <value>
        /// <see langword="true" /> means <see cref="RecencyGain"/> is multiplicative.
        /// <see langword="false" /> means <see cref="RecencyGain"/> is additive.
        /// Default is <see langword="false" />.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Whether Recency Gain is multiplicative (vs. additive)", ShortName = "rgm,RecencyGainMulti")]
        public bool RecencyGainMultiplicative = false;

        /// <summary>
        /// Determines whether to do averaging or not.
        /// </summary>
        /// <value>
        /// <see langword="true" /> to do averaging; otherwise, <see langword="false" />.
        /// Default is <see langword="true" />.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Do averaging?", ShortName = "avg")]
        public bool Averaged = true;

        /// <summary>
        /// The inexactness tolerance for averaging.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "The inexactness tolerance for averaging", ShortName = "avgtol")]
        internal float AveragedTolerance = (float)1e-2;

        [BestFriend]
        internal class AveragedDefault : OnlineLinearOptions.OnlineDefault
        {
            public const float LearningRate = 1;
            public const bool DecreaseLearningRate = false;
            public const float L2Regularization = 0;
        }

        internal abstract IComponentFactory<IScalarLoss> LossFunctionFactory { get; }
    }

    public abstract class AveragedLinearTrainer<TTransformer, TModel> : OnlineLinearTrainer<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        private protected readonly AveragedLinearOptions AveragedLinearTrainerOptions;
        private protected IScalarLoss LossFunction;

        private protected abstract class AveragedTrainStateBase : TrainStateBase
        {
            protected float Gain;

            protected int NumNoUpdates;

            // For computing averaged weights and bias (if needed)
            protected VBuffer<float> TotalWeights;
            protected float TotalBias;
            protected double NumWeightUpdates;

            // The accumulated gradient of loss against gradient for all updates so far in the
            // totalled model, versus those pending in the weight vector that have not yet been
            // added to the total model.
            protected double TotalMultipliers;
            protected double PendingMultipliers;

            protected readonly bool Averaged;
            private readonly long _resetWeightsAfterXExamples;
            private readonly AveragedLinearOptions _args;
            private readonly IScalarLoss _loss;

            private protected AveragedTrainStateBase(IChannel ch, int numFeatures, LinearModelParameters predictor, AveragedLinearTrainer<TTransformer, TModel> parent)
                : base(ch, numFeatures, predictor, parent)
            {
                // Do the other initializations by setting the setters as if user had set them
                // Initialize the averaged weights if needed (i.e., do what happens when Averaged is set)
                Averaged = parent.AveragedLinearTrainerOptions.Averaged;
                if (Averaged)
                {
                    if (parent.AveragedLinearTrainerOptions.AveragedTolerance > 0)
                        VBufferUtils.Densify(ref Weights);
                    Weights.CopyTo(ref TotalWeights);
                }
                else
                {
                    // It is definitely advantageous to keep weights dense if we aren't adding them
                    // to another vector with each update.
                    VBufferUtils.Densify(ref Weights);
                }
                _resetWeightsAfterXExamples = parent.AveragedLinearTrainerOptions.ResetWeightsAfterXExamples ?? 0;
                _args = parent.AveragedLinearTrainerOptions;
                _loss = parent.LossFunction;

                Gain = 1;
            }

            /// <summary>
            /// Return the raw margin from the decision hyperplane
            /// </summary>
            public float AveragedMargin(in VBuffer<float> feat)
            {
                Contracts.Assert(Averaged);
                return (TotalBias + VectorUtils.DotProduct(in feat, in TotalWeights)) / (float)NumWeightUpdates;
            }

            public override float Margin(in VBuffer<float> feat)
                => Averaged ? AveragedMargin(in feat) : CurrentMargin(in feat);

            public override void FinishIteration(IChannel ch)
            {
                // Finalize things
                if (Averaged)
                {
                    if (_args.LazyUpdate && NumNoUpdates > 0)
                    {
                        // Update the total weights to include the final loss=0 updates
                        VectorUtils.AddMult(in Weights, NumNoUpdates * WeightsScale, ref TotalWeights);
                        TotalBias += Bias * NumNoUpdates;
                        NumWeightUpdates += NumNoUpdates;
                        NumNoUpdates = 0;
                        TotalMultipliers += PendingMultipliers;
                        PendingMultipliers = 0;
                    }

                    // reset the weights to averages if needed
                    if (_args.ResetWeightsAfterXExamples == 0)
                    {
                        ch.Info("Resetting weights to average weights");
                        VectorUtils.ScaleInto(in TotalWeights, 1 / (float)NumWeightUpdates, ref Weights);
                        WeightsScale = 1;
                        Bias = TotalBias / (float)NumWeightUpdates;
                    }
                }

                base.FinishIteration(ch);
            }

            public override void ProcessDataInstance(IChannel ch, in VBuffer<float> feat, float label, float weight)
            {
                base.ProcessDataInstance(ch, in feat, label, weight);

                // compute the update and update if needed
                float output = CurrentMargin(in feat);
                Double loss = _loss.Loss(output, label);

                // REVIEW: Should this be biasUpdate != 0?
                // This loss does not incorporate L2 if present, but the chance of that addition to the loss
                // exactly cancelling out loss is remote.
                if (loss != 0 || _args.L2Regularization > 0)
                {
                    // If doing lazy weights, we need to update the totalWeights and totalBias before updating weights/bias
                    if (_args.LazyUpdate && _args.Averaged && NumNoUpdates > 0 && TotalMultipliers * _args.AveragedTolerance <= PendingMultipliers)
                    {
                        VectorUtils.AddMult(in Weights, NumNoUpdates * WeightsScale, ref TotalWeights);
                        TotalBias += Bias * NumNoUpdates * WeightsScale;
                        NumWeightUpdates += NumNoUpdates;
                        NumNoUpdates = 0;
                        TotalMultipliers += PendingMultipliers;
                        PendingMultipliers = 0;
                    }

                    // Make final adjustments to update parameters.
                    float rate = _args.LearningRate;
                    if (_args.DecreaseLearningRate)
                        rate /= MathUtils.Sqrt((float)NumWeightUpdates + NumNoUpdates + 1);
                    float biasUpdate = -rate * _loss.Derivative(output, label);

                    // Perform the update to weights and bias.
                    VectorUtils.AddMult(in feat, biasUpdate / WeightsScale, ref Weights);
                    WeightsScale *= 1 - 2 * _args.L2Regularization; // L2 regularization.
                    ScaleWeightsIfNeeded();
                    Bias += biasUpdate;
                    PendingMultipliers += Math.Abs(biasUpdate);
                }

                // Add to averaged weights and increment the count.
                if (Averaged)
                {
                    if (!_args.LazyUpdate)
                        IncrementAverageNonLazy();
                    else
                        NumNoUpdates++;

                    // Reset the weights to averages if needed.
                    if (_resetWeightsAfterXExamples > 0 && NumIterExamples % _resetWeightsAfterXExamples == 0)
                    {
                        ch.Info("Resetting weights to average weights");
                        VectorUtils.ScaleInto(in TotalWeights, 1 / (float)NumWeightUpdates, ref Weights);
                        WeightsScale = 1;
                        Bias = TotalBias / (float)NumWeightUpdates;
                    }
                }
            }

            /// <summary>
            /// Add current weights and bias to average weights/bias.
            /// </summary>
            private void IncrementAverageNonLazy()
            {
                if (_args.RecencyGain == 0)
                {
                    VectorUtils.AddMult(in Weights, WeightsScale, ref TotalWeights);
                    TotalBias += Bias;
                    NumWeightUpdates++;
                    return;
                }
                VectorUtils.AddMult(in Weights, Gain * WeightsScale, ref TotalWeights);
                TotalBias += Gain * Bias;
                NumWeightUpdates += Gain;
                Gain = (_args.RecencyGainMultiplicative ? Gain * _args.RecencyGain : Gain + _args.RecencyGain);

                // If gains got too big, rescale!
                if (Gain > 1000)
                {
                    const float scale = (float)1e-6;
                    Gain *= scale;
                    TotalBias *= scale;
                    VectorUtils.ScaleBy(ref TotalWeights, scale);
                    NumWeightUpdates *= scale;
                }
            }
        }

        private protected AveragedLinearTrainer(AveragedLinearOptions options, IHostEnvironment env, string name, SchemaShape.Column label)
            : base(options, env, name, label)
        {
            Contracts.CheckUserArg(options.LearningRate > 0, nameof(options.LearningRate), UserErrorPositive);
            Contracts.CheckUserArg(!options.ResetWeightsAfterXExamples.HasValue || options.ResetWeightsAfterXExamples > 0, nameof(options.ResetWeightsAfterXExamples), UserErrorPositive);

            // Weights are scaled down by 2 * L2 regularization on each update step, so 0.5 would scale all weights to 0, which is not sensible.
            Contracts.CheckUserArg(0 <= options.L2Regularization && options.L2Regularization < 0.5, nameof(options.L2Regularization), "must be in range [0, 0.5)");
            Contracts.CheckUserArg(options.RecencyGain >= 0, nameof(options.RecencyGain), UserErrorNonNegative);
            Contracts.CheckUserArg(options.AveragedTolerance >= 0, nameof(options.AveragedTolerance), UserErrorNonNegative);
            // Verify user didn't specify parameters that conflict
            Contracts.Check(!options.LazyUpdate || !options.RecencyGainMultiplicative && options.RecencyGain == 0, "Cannot have both recency gain and lazy updates.");

            AveragedLinearTrainerOptions = options;
        }
    }
}
