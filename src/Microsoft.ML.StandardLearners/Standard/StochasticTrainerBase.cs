// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Training;
using System;

namespace Microsoft.ML.Runtime.Learners
{
    public abstract class StochasticTrainerBase<TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : IPredictor
    {
        public StochasticTrainerBase(IHost host, SchemaShape.Column feature, SchemaShape.Column label, SchemaShape.Column weight = null)
            : base(host, feature, label, weight)
        {
        }

        /// <summary>
        /// Whether data is to be shuffled every epoch.
        /// </summary>
        protected abstract bool ShuffleData { get; }

        private static readonly TrainerInfo _info = new TrainerInfo();
        public override TrainerInfo Info => _info;

        protected override TModel TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            TModel pred;
            using (var ch = Host.Start("Training"))
            {
                var preparedData = PrepareDataFromTrainingExamples(ch, context.TrainingSet, out int weightSetCount);
                var initPred = context.InitialPredictor;
                var linInitPred = (initPred as CalibratedPredictorBase)?.SubPredictor as LinearPredictor;
                linInitPred = linInitPred ?? initPred as LinearPredictor;
                Host.CheckParam(context.InitialPredictor == null || linInitPred != null, nameof(context),
                    "Initial predictor was not a linear predictor.");
                pred = TrainCore(ch, preparedData, linInitPred, weightSetCount);
                ch.Done();
            }
            return pred;
        }

        protected virtual int ComputeNumThreads(FloatLabelCursor.Factory cursorFactory)
        {
            int maxThreads = Math.Min(8, Math.Max(1, Environment.ProcessorCount / 2));
            if (0 < Host.ConcurrencyFactor && Host.ConcurrencyFactor < maxThreads)
                maxThreads = Host.ConcurrencyFactor;

            return maxThreads;
        }

        /// <summary>
        /// This method ensures that the data meets the requirements of this trainer and its
        /// subclasses, injects necessary transforms, and throws if it couldn't meet them.
        /// </summary>
        /// <param name="ch">The channel</param>
        /// <param name="examples">The training examples</param>
        /// <param name="weightSetCount">Gets the length of weights and bias array. For binary classification and regression,
        /// this is 1. For multi-class classification, this equals the number of classes on the label.</param>
        /// <returns>A potentially modified version of <paramref name="examples"/></returns>
        protected RoleMappedData PrepareDataFromTrainingExamples(IChannel ch, RoleMappedData examples, out int weightSetCount)
        {
            ch.AssertValue(examples);
            CheckLabel(examples, out weightSetCount);
            examples.CheckFeatureFloatVector();
            var idvToShuffle = examples.Data;
            IDataView idvToFeedTrain;
            if (idvToShuffle.CanShuffle)
                idvToFeedTrain = idvToShuffle;
            else
            {
                var shuffleArgs = new ShuffleTransform.Arguments
                {
                    PoolOnly = false,
                    ForceShuffle = ShuffleData
                };
                idvToFeedTrain = new ShuffleTransform(Host, shuffleArgs, idvToShuffle);
            }

            ch.Assert(idvToFeedTrain.CanShuffle);

            var roles = examples.Schema.GetColumnRoleNames();
            var examplesToFeedTrain = new RoleMappedData(idvToFeedTrain, roles);

            ch.AssertValue(examplesToFeedTrain.Schema.Label);
            ch.AssertValue(examplesToFeedTrain.Schema.Feature);
            if (examples.Schema.Weight != null)
                ch.AssertValue(examplesToFeedTrain.Schema.Weight);

            int numFeatures = examplesToFeedTrain.Schema.Feature.Type.VectorSize;
            ch.Check(numFeatures > 0, "Training set has no features, aborting training.");
            return examplesToFeedTrain;
        }

        protected abstract TModel TrainCore(IChannel ch, RoleMappedData data, LinearPredictor predictor, int weightSetCount);

        protected abstract void CheckLabel(RoleMappedData examples, out int weightSetCount);
    }
}
