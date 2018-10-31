// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime;

// TODO: Check if it works properly if Averaged is set to false

namespace Microsoft.ML.Trainers.Online
{
    public abstract class AveragedLinearArguments : OnlineLinearArguments
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate", ShortName = "lr", SortOrder = 50)]
        [TGUI(Label = "Learning rate", SuggestedSweeps = "0.01,0.1,0.5,1.0")]
        [TlcModule.SweepableDiscreteParam("LearningRate", new object[] { 0.01, 0.1, 0.5, 1.0 })]
        public Float LearningRate = AveragedDefaultArgs.LearningRate;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Decrease learning rate", ShortName = "decreaselr", SortOrder = 50)]
        [TGUI(Label = "Decrease Learning Rate", Description = "Decrease learning rate as iterations progress")]
        [TlcModule.SweepableDiscreteParam("DecreaseLearningRate", new object[] { false, true })]
        public bool DecreaseLearningRate = AveragedDefaultArgs.DecreaseLearningRate;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of examples after which weights will be reset to the current average", ShortName = "numreset")]
        public long? ResetWeightsAfterXExamples = null;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Instead of updating averaged weights on every example, only update when loss is nonzero", ShortName = "lazy")]
        public bool DoLazyUpdates = true;

        [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization Weight", ShortName = "reg", SortOrder = 50)]
        [TGUI(Label = "L2 Regularization Weight")]
        [TlcModule.SweepableFloatParam("L2RegularizerWeight", 0.0f, 0.4f)]
        public Float L2RegularizerWeight = AveragedDefaultArgs.L2RegularizerWeight;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Extra weight given to more recent updates", ShortName = "rg")]
        public Float RecencyGain = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Whether Recency Gain is multiplicative (vs. additive)", ShortName = "rgm")]
        public bool RecencyGainMulti = false;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Do averaging?", ShortName = "avg")]
        public bool Averaged = true;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The inexactness tolerance for averaging", ShortName = "avgtol")]
        public Float AveragedTolerance = (Float)1e-2;

        internal class AveragedDefaultArgs : OnlineDefaultArgs
        {
            internal const Float LearningRate = 1;
            internal const bool DecreaseLearningRate = false;
            internal const Float L2RegularizerWeight = 0;
        }

        internal abstract IComponentFactory<IScalarOutputLoss> LossFunctionFactory { get; }
    }

    public abstract class AveragedLinearTrainer<TTransformer, TModel> : OnlineLinearTrainer<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : IPredictor
    {
        protected readonly new AveragedLinearArguments Args;
        protected IScalarOutputLoss LossFunction;

        private protected abstract class AveragedTrainStateBase : TrainStateBase
        {
            protected Float Gain;

            protected int NumNoUpdates;

            // For computing averaged weights and bias (if needed)
            protected VBuffer<Float> TotalWeights;
            protected Float TotalBias;
            protected Double NumWeightUpdates;

            // The accumulated gradient of loss against gradient for all updates so far in the
            // totalled model, versus those pending in the weight vector that have not yet been
            // added to the total model.
            protected Double TotalMultipliers;
            protected Double PendingMultipliers;

            protected readonly bool Averaged;
            private readonly long _resetWeightsAfterXExamples;
            private readonly AveragedLinearArguments _args;
            private readonly IScalarOutputLoss _loss;

            private protected AveragedTrainStateBase(IChannel ch, int numFeatures, LinearPredictor predictor, AveragedLinearTrainer<TTransformer, TModel> parent)
                : base(ch, numFeatures, predictor, parent)
            {
                // Do the other initializations by setting the setters as if user had set them
                // Initialize the averaged weights if needed (i.e., do what happens when Averaged is set)
                Averaged = parent.Args.Averaged;
                if (Averaged)
                {
                    if (parent.Args.AveragedTolerance > 0)
                        VBufferUtils.Densify(ref Weights);
                    Weights.CopyTo(ref TotalWeights);
                }
                else
                {
                    // It is definitely advantageous to keep weights dense if we aren't adding them
                    // to another vector with each update.
                    VBufferUtils.Densify(ref Weights);
                }
                _resetWeightsAfterXExamples = parent.Args.ResetWeightsAfterXExamples ?? 0;
                _args = parent.Args;
                _loss = parent.LossFunction;

                Gain = 1;
            }

            /// <summary>
            /// Return the raw margin from the decision hyperplane
            /// </summary>
            public Float AveragedMargin(ref VBuffer<Float> feat)
            {
                Contracts.Assert(Averaged);
                return (TotalBias + VectorUtils.DotProduct(ref feat, ref TotalWeights)) / (Float)NumWeightUpdates;
            }

            public override Float Margin(ref VBuffer<Float> feat)
                => Averaged ? AveragedMargin(ref feat) : CurrentMargin(ref feat);

            public override void FinishIteration(IChannel ch)
            {
                // Finalize things
                if (Averaged)
                {
                    if (_args.DoLazyUpdates && NumNoUpdates > 0)
                    {
                        // Update the total weights to include the final loss=0 updates
                        VectorUtils.AddMult(ref Weights, NumNoUpdates * WeightsScale, ref TotalWeights);
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
                        VectorUtils.ScaleInto(ref TotalWeights, 1 / (Float)NumWeightUpdates, ref Weights);
                        WeightsScale = 1;
                        Bias = TotalBias / (Float)NumWeightUpdates;
                    }
                }

                base.FinishIteration(ch);
            }

            public override void ProcessDataInstance(IChannel ch, ref VBuffer<Float> feat, Float label, Float weight)
            {
                base.ProcessDataInstance(ch, ref feat, label, weight);

                // compute the update and update if needed
                Float output = CurrentMargin(ref feat);
                Double loss = _loss.Loss(output, label);

                // REVIEW: Should this be biasUpdate != 0?
                // This loss does not incorporate L2 if present, but the chance of that addition to the loss
                // exactly cancelling out loss is remote.
                if (loss != 0 || _args.L2RegularizerWeight > 0)
                {
                    // If doing lazy weights, we need to update the totalWeights and totalBias before updating weights/bias
                    if (_args.DoLazyUpdates && _args.Averaged && NumNoUpdates > 0 && TotalMultipliers * _args.AveragedTolerance <= PendingMultipliers)
                    {
                        VectorUtils.AddMult(ref Weights, NumNoUpdates * WeightsScale, ref TotalWeights);
                        TotalBias += Bias * NumNoUpdates * WeightsScale;
                        NumWeightUpdates += NumNoUpdates;
                        NumNoUpdates = 0;
                        TotalMultipliers += PendingMultipliers;
                        PendingMultipliers = 0;
                    }

                    // Make final adjustments to update parameters.
                    Float rate = _args.LearningRate;
                    if (_args.DecreaseLearningRate)
                        rate /= MathUtils.Sqrt((Float)NumWeightUpdates + NumNoUpdates + 1);
                    Float biasUpdate = -rate * _loss.Derivative(output, label);

                    // Perform the update to weights and bias.
                    VectorUtils.AddMult(ref feat, biasUpdate / WeightsScale, ref Weights);
                    WeightsScale *= 1 - 2 * _args.L2RegularizerWeight; // L2 regularization.
                    ScaleWeightsIfNeeded();
                    Bias += biasUpdate;
                    PendingMultipliers += Math.Abs(biasUpdate);
                }

                // Add to averaged weights and increment the count.
                if (Averaged)
                {
                    if (!_args.DoLazyUpdates)
                        IncrementAverageNonLazy();
                    else
                        NumNoUpdates++;

                    // Reset the weights to averages if needed.
                    if (_resetWeightsAfterXExamples > 0 && NumIterExamples % _resetWeightsAfterXExamples == 0)
                    {
                        ch.Info("Resetting weights to average weights");
                        VectorUtils.ScaleInto(ref TotalWeights, 1 / (Float)NumWeightUpdates, ref Weights);
                        WeightsScale = 1;
                        Bias = TotalBias / (Float)NumWeightUpdates;
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
                    VectorUtils.AddMult(ref Weights, WeightsScale, ref TotalWeights);
                    TotalBias += Bias;
                    NumWeightUpdates++;
                    return;
                }
                VectorUtils.AddMult(ref Weights, Gain * WeightsScale, ref TotalWeights);
                TotalBias += Gain * Bias;
                NumWeightUpdates += Gain;
                Gain = (_args.RecencyGainMulti ? Gain * _args.RecencyGain : Gain + _args.RecencyGain);

                // If gains got too big, rescale!
                if (Gain > 1000)
                {
                    const Float scale = (Float)1e-6;
                    Gain *= scale;
                    TotalBias *= scale;
                    VectorUtils.ScaleBy(ref TotalWeights, scale);
                    NumWeightUpdates *= scale;
                }
            }
        }

        protected AveragedLinearTrainer(AveragedLinearArguments args, IHostEnvironment env, string name, SchemaShape.Column label)
            : base(args, env, name, label)
        {
            Contracts.CheckUserArg(args.LearningRate > 0, nameof(args.LearningRate), UserErrorPositive);
            Contracts.CheckUserArg(!args.ResetWeightsAfterXExamples.HasValue || args.ResetWeightsAfterXExamples > 0, nameof(args.ResetWeightsAfterXExamples), UserErrorPositive);

            // Weights are scaled down by 2 * L2 regularization on each update step, so 0.5 would scale all weights to 0, which is not sensible.
            Contracts.CheckUserArg(0 <= args.L2RegularizerWeight && args.L2RegularizerWeight < 0.5, nameof(args.L2RegularizerWeight), "must be in range [0, 0.5)");
            Contracts.CheckUserArg(args.RecencyGain >= 0, nameof(args.RecencyGain), UserErrorNonNegative);
            Contracts.CheckUserArg(args.AveragedTolerance >= 0, nameof(args.AveragedTolerance), UserErrorNonNegative);
            // Verify user didn't specify parameters that conflict
            Contracts.Check(!args.DoLazyUpdates || !args.RecencyGainMulti && args.RecencyGain == 0, "Cannot have both recency gain and lazy updates.");

            Args = args;
        }
    }
}