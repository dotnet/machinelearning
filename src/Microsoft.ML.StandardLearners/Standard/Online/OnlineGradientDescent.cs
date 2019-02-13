// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Numeric;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Training;

[assembly: LoadableClass(OnlineGradientDescentTrainer.Summary, typeof(OnlineGradientDescentTrainer), typeof(OnlineGradientDescentTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    OnlineGradientDescentTrainer.UserNameValue,
    OnlineGradientDescentTrainer.LoadNameValue,
    OnlineGradientDescentTrainer.ShortName,
    "sgdr",
    "stochasticgradientdescentregression")]
[assembly: LoadableClass(typeof(void), typeof(OnlineGradientDescentTrainer), null, typeof(SignatureEntryPointModule), "OGD")]

namespace Microsoft.ML.Trainers.Online
{

    /// <include file='doc.xml' path='doc/members/member[@name="OGD"]/*' />
    public sealed class OnlineGradientDescentTrainer : AveragedLinearTrainer<RegressionPredictionTransformer<LinearRegressionModelParameters>, LinearRegressionModelParameters>
    {
        internal const string LoadNameValue = "OnlineGradientDescent";
        internal const string UserNameValue = "Stochastic Gradient Descent (Regression)";
        internal const string Summary = "Stochastic gradient descent is an optimization method used to train a wide range of models in machine learning. "
            + "In the TLC implementation of OGD, it is for linear regression.";
        internal const string ShortName = "ogd";

        public sealed class Options : AveragedLinearArguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            [TGUI(Label = "Loss Function")]
            public ISupportRegressionLossFactory LossFunction = new SquaredLossFactory();

            /// <summary>
            /// Set defaults that vary from the base type.
            /// </summary>
            public Options()
            {
                LearningRate = OgdDefaultArgs.LearningRate;
                DecreaseLearningRate = OgdDefaultArgs.DecreaseLearningRate;
            }

            internal override IComponentFactory<IScalarOutputLoss> LossFunctionFactory => LossFunction;

            [BestFriend]
            internal class OgdDefaultArgs : AveragedDefaultArgs
            {
                public new const float LearningRate = 0.1f;
                public new const bool DecreaseLearningRate = true;
            }
        }

        private sealed class TrainState : AveragedTrainStateBase
        {
            public TrainState(IChannel ch, int numFeatures, LinearModelParameters predictor, OnlineGradientDescentTrainer parent)
                : base(ch, numFeatures, predictor, parent)
            {
            }

            public override LinearRegressionModelParameters CreatePredictor()
            {
                Contracts.Assert(WeightsScale == 1);
                VBuffer<float> weights = default;
                float bias;

                if (!Averaged)
                {
                    Weights.CopyTo(ref weights);
                    bias = Bias;
                }
                else
                {
                    TotalWeights.CopyTo(ref weights);
                    VectorUtils.ScaleBy(ref weights, 1 / (float)NumWeightUpdates);
                    bias = TotalBias / (float)NumWeightUpdates;
                }
                return new LinearRegressionModelParameters(ParentHost, in weights, bias);
            }
        }

        /// <summary>
        /// Trains a new <see cref="RegressionPredictionTransformer{LinearRegressionPredictor}"/>.
        /// </summary>
        /// <param name="env">The pricate instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">Name of the label column.</param>
        /// <param name="featureColumn">Name of the feature column.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 Regularization Weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        /// <param name="weightsColumn">The name of the weights column.</param>
        /// <param name="lossFunction">The custom loss functions. Defaults to <see cref="SquaredLoss"/> if not provided.</param>
        internal OnlineGradientDescentTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            float learningRate = Options.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = Options.OgdDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = Options.OgdDefaultArgs.L2RegularizerWeight,
            int numIterations = Options.OgdDefaultArgs.NumIterations,
            string weightsColumn = null,
            IRegressionLoss lossFunction = null)
            : this(env, new Options
            {
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLearningRate,
                L2RegularizerWeight = l2RegularizerWeight,
                NumIterations = numIterations,
                LabelColumn = labelColumn,
                FeatureColumn = featureColumn,
                InitialWeights = weightsColumn,
                LossFunction = new TrivialFactory(lossFunction ?? new SquaredLoss())
            })
        {
        }

        private sealed class TrivialFactory : ISupportRegressionLossFactory
        {
            private IRegressionLoss _loss;

            public TrivialFactory(IRegressionLoss loss)
            {
                _loss = loss;
            }

            IRegressionLoss IComponentFactory<IRegressionLoss>.CreateComponent(IHostEnvironment env) => _loss;
        }

        internal OnlineGradientDescentTrainer(IHostEnvironment env, Options options)
        : base(options, env, UserNameValue, TrainerUtils.MakeR4ScalarColumn(options.LabelColumn))
        {
            LossFunction = options.LossFunction.CreateComponent(env);
        }

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        private protected override void CheckLabels(RoleMappedData data)
        {
            data.CheckRegressionLabel();
        }

        private protected override TrainStateBase MakeState(IChannel ch, int numFeatures, LinearModelParameters predictor)
        {
            return new TrainState(ch, numFeatures, predictor, this);
        }

        [TlcModule.EntryPoint(Name = "Trainers.OnlineGradientDescentRegressor",
            Desc = "Train a Online gradient descent perceptron.",
            UserName = UserNameValue,
            ShortName = ShortName)]
        internal static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainOGD");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Options, CommonOutputs.RegressionOutput>(host, input,
                () => new OnlineGradientDescentTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }

        protected override RegressionPredictionTransformer<LinearRegressionModelParameters> MakeTransformer(LinearRegressionModelParameters model, Schema trainSchema)
        => new RegressionPredictionTransformer<LinearRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        public RegressionPredictionTransformer<LinearRegressionModelParameters> Train(IDataView trainData, IPredictor initialPredictor = null)
            => TrainTransformer(trainData, initPredictor: initialPredictor);
    }
}
