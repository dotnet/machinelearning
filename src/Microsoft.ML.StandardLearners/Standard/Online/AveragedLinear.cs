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

// TODO: Check if it works properly if Averaged is set to false

namespace Microsoft.ML.Runtime.Learners
{
    public abstract class AveragedLinearArguments : OnlineLinearArguments
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate", ShortName = "lr", SortOrder = 50)]
        [TGUI(Label = "Learning rate", SuggestedSweeps = "0.01,0.1,0.5,1.0")]
        [TlcModule.SweepableDiscreteParam("LearningRate", new object[] { 0.01, 0.1, 0.5, 1.0 })]
        public Float LearningRate = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Decrease learning rate", ShortName = "decreaselr", SortOrder = 50)]
        [TGUI(Label = "Decrease Learning Rate", Description = "Decrease learning rate as iterations progress")]
        [TlcModule.SweepableDiscreteParam("DecreaseLearningRate", new object[] { false, true })]
        public bool DecreaseLearningRate = false;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of examples after which weights will be reset to the current average", ShortName = "numreset")]
        public long? ResetWeightsAfterXExamples = null;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Instead of updating averaged weights on every example, only update when loss is nonzero", ShortName = "lazy")]
        public bool DoLazyUpdates = true;

        [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization Weight", ShortName = "reg", SortOrder = 50)]
        [TGUI(Label = "L2 Regularization Weight")]
        [TlcModule.SweepableFloatParam("L2RegularizerWeight", 0.0f, 0.4f)]
        public Float L2RegularizerWeight = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Extra weight given to more recent updates", ShortName = "rg")]
        public Float RecencyGain = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Whether Recency Gain is multiplicative (vs. additive)", ShortName = "rgm")]
        public bool RecencyGainMulti = false;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Do averaging?", ShortName = "avg")]
        public bool Averaged = true;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The inexactness tolerance for averaging", ShortName = "avgtol")]
        public Float AveragedTolerance = (Float)1e-2;
    }

    public abstract class AveragedLinearTrainer<TTransformer, TModel> : OnlineLinearTrainer<TTransformer, TModel>
        where TTransformer : IPredictionTransformer<TModel>
        where TModel : IPredictor
    {
        protected readonly new AveragedLinearArguments Args;
        protected IScalarOutputLoss LossFunction;
        protected Float Gain;

        // For computing averaged weights and bias (if needed)
        protected VBuffer<Float> TotalWeights;
        protected Float TotalBias;
        protected Double NumWeightUpdates;

        // The accumulated gradient of loss against gradient for all updates so far in the
        // totalled model, versus those pending in the weight vector that have not yet been
        // added to the total model.
        protected Double TotalMultipliers;
        protected Double PendingMultipliers;

        // We'll keep a few things global to prevent garbage collection
        protected int NumNoUpdates;

        protected AveragedLinearTrainer(AveragedLinearArguments args, IHostEnvironment env, string name, SchemaShape.Column label)
            : base(args, env, name, label)
        {
            Contracts.CheckUserArg(args.LearningRate > 0, nameof(args.LearningRate), UserErrorPositive);
            Contracts.CheckUserArg(!args.ResetWeightsAfterXExamples.HasValue || args.ResetWeightsAfterXExamples > 0, nameof(args.ResetWeightsAfterXExamples), UserErrorPositive);

            // Weights are scaled down by 2 * L2 regularization on each update step, so 0.5 would scale all weights to 0, which is not sensible.
            Contracts.CheckUserArg(0 <= args.L2RegularizerWeight && args.L2RegularizerWeight < 0.5, nameof(args.L2RegularizerWeight), "must be in range [0, 0.5)");
            Contracts.CheckUserArg(args.RecencyGain >= 0, nameof(args.RecencyGain), UserErrorNonNegative);
            Contracts.CheckUserArg(args.AveragedTolerance >= 0, nameof(args.AveragedTolerance), UserErrorNonNegative);

            Args = args;
        }

        protected override void InitCore(IChannel ch, int numFeatures, LinearPredictor predictor)
        {
            base.InitCore(ch, numFeatures, predictor);

            // Verify user didn't specify parameters that conflict
            Contracts.Check(!Args.DoLazyUpdates || !Args.RecencyGainMulti && Args.RecencyGain == 0,
                "Cannot have both recency gain and lazy updates.");

            // Do the other initializations by setting the setters as if user had set them
            // Initialize the averaged weights if needed (i.e., do what happens when Averaged is set)
            if (Args.Averaged)
            {
                if (Args.AveragedTolerance > 0)
                    VBufferUtils.Densify(ref Weights);
                Weights.CopyTo(ref TotalWeights);
            }
            else
            {
                // It is definitely advantageous to keep weights dense if we aren't adding them
                // to another vector with each update.
                VBufferUtils.Densify(ref Weights);
            }
            Gain = 1;
        }

        /// <summary>
        /// Return the raw margin from the decision hyperplane
        /// </summary>
        protected Float AveragedMargin(ref VBuffer<Float> feat)
        {
            Contracts.Assert(Args.Averaged);
            return (TotalBias + VectorUtils.DotProduct(ref feat, ref TotalWeights)) / (Float)NumWeightUpdates;
        }

        protected override Float Margin(ref VBuffer<Float> feat)
        {
            return Args.Averaged ? AveragedMargin(ref feat) : CurrentMargin(ref feat);
        }

        protected override void FinishIteration(IChannel ch)
        {
            // REVIEW: Very odd - the old AP and OGD did different things here and neither seemed correct.

            // Finalize things
            if (Args.Averaged)
            {
                if (Args.DoLazyUpdates && NumNoUpdates > 0)
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
                if (Args.ResetWeightsAfterXExamples == 0)
                {
                    // #if OLD_TRACING // REVIEW: How should this be ported?
                    Console.WriteLine("");
                    // #endif
                    ch.Info("Resetting weights to average weights");
                    VectorUtils.ScaleInto(ref TotalWeights, 1 / (Float)NumWeightUpdates, ref Weights);
                    WeightsScale = 1;
                    Bias = TotalBias / (Float)NumWeightUpdates;
                }
            }

            base.FinishIteration(ch);
        }

#if OLD_TRACING // REVIEW: How should this be ported?
        protected override void PrintWeightsHistogram()
        {
            if (_args.averaged)
                PrintWeightsHistogram(ref _totalWeights, _totalBias, (Float)_numWeightUpdates);
            else
                base.PrintWeightsHistogram();
        }
#endif

        protected override void ProcessDataInstance(IChannel ch, ref VBuffer<Float> feat, Float label, Float weight)
        {
            base.ProcessDataInstance(ch, ref feat, label, weight);

            // compute the update and update if needed
            Float output = CurrentMargin(ref feat);
            Double loss = LossFunction.Loss(output, label);

            // REVIEW: Should this be biasUpdate != 0?
            // This loss does not incorporate L2 if present, but the chance of that addition to the loss
            // exactly cancelling out loss is remote.
            if (loss != 0 || Args.L2RegularizerWeight > 0)
            {
                // If doing lazy weights, we need to update the totalWeights and totalBias before updating weights/bias
                if (Args.DoLazyUpdates && Args.Averaged && NumNoUpdates > 0 && TotalMultipliers * Args.AveragedTolerance <= PendingMultipliers)
                {
                    VectorUtils.AddMult(ref Weights, NumNoUpdates * WeightsScale, ref TotalWeights);
                    TotalBias += Bias * NumNoUpdates * WeightsScale;
                    NumWeightUpdates += NumNoUpdates;
                    NumNoUpdates = 0;
                    TotalMultipliers += PendingMultipliers;
                    PendingMultipliers = 0;
                }

#if OLD_TRACING // REVIEW: How should this be ported?
                // If doing debugging and have L2 regularization, adjust the loss to account for that component.
                if (DebugLevel > 2 && _args.l2RegularizerWeight != 0)
                    loss += _args.l2RegularizerWeight * VectorUtils.NormSquared(_weights) * _weightsScale * _weightsScale;
#endif

                // Make final adjustments to update parameters.
                Float rate = Args.LearningRate;
                if (Args.DecreaseLearningRate)
                    rate /= MathUtils.Sqrt((Float)NumWeightUpdates + NumNoUpdates + 1);
                Float biasUpdate = -rate * LossFunction.Derivative(output, label);

                // Perform the update to weights and bias.
                VectorUtils.AddMult(ref feat, biasUpdate / WeightsScale, ref Weights);
                WeightsScale *= 1 - 2 * Args.L2RegularizerWeight; // L2 regularization.
                ScaleWeightsIfNeeded();
                Bias += biasUpdate;
                PendingMultipliers += Math.Abs(biasUpdate);

#if OLD_TRACING // REVIEW: How should this be ported?
                if (DebugLevel > 2)
                { // sanity check:   did loss for the example decrease?
                    Double newLoss = _lossFunction.Loss(CurrentMargin(instance), instance.Label);
                    if (_args.l2RegularizerWeight != 0)
                        newLoss += _args.l2RegularizerWeight * VectorUtils.NormSquared(_weights) * _weightsScale * _weightsScale;

                    if (newLoss - loss > 0 && (newLoss - loss > 0.01 || _args.l2RegularizerWeight == 0))
                    {
                        Host.StdErr.WriteLine("Loss increased (unexpected):  Old value: {0}, new value: {1}", loss, newLoss);
                        Host.StdErr.WriteLine("Offending instance #{0}: {1}", _numIterExamples, instance);
                    }
                }
#endif
            }

            // Add to averaged weights and increment the count.
            if (Args.Averaged)
            {
                if (!Args.DoLazyUpdates)
                    IncrementAverageNonLazy();
                else
                    NumNoUpdates++;

                // Reset the weights to averages if needed.
                if (Args.ResetWeightsAfterXExamples > 0 &&
                    NumIterExamples % Args.ResetWeightsAfterXExamples.Value == 0)
                {
                    // #if OLD_TRACING // REVIEW: How should this be ported?
                    Console.WriteLine();
                    // #endif
                    ch.Info("Resetting weights to average weights");
                    VectorUtils.ScaleInto(ref TotalWeights, 1 / (Float)NumWeightUpdates, ref Weights);
                    WeightsScale = 1;
                    Bias = TotalBias / (Float)NumWeightUpdates;
                }
            }

#if OLD_TRACING // REVIEW: How should this be ported?
            if (DebugLevel > 3)
            {
                // Output the weights.
                Host.StdOut.Write("Weights after the instance are: ");
                foreach (var iv in _weights.Items(all: true))
                {
                    Host.StdOut.Write('\t');
                    Host.StdOut.Write(iv.Value * _weightsScale);
                }
                Host.StdOut.WriteLine();
                Host.StdOut.WriteLine();
            }
#endif
        }

        /// <summary>
        /// Add current weights and bias to average weights/bias.
        /// </summary>
        protected void IncrementAverageNonLazy()
        {
            if (Args.RecencyGain == 0)
            {
                VectorUtils.AddMult(ref Weights, WeightsScale, ref TotalWeights);
                TotalBias += Bias;
                NumWeightUpdates++;
                return;
            }
            VectorUtils.AddMult(ref Weights, Gain * WeightsScale, ref TotalWeights);
            TotalBias += Gain * Bias;
            NumWeightUpdates += Gain;
            Gain = (Args.RecencyGainMulti ? Gain * Args.RecencyGain : Gain + Args.RecencyGain);

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
}