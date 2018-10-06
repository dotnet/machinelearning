// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(OnlineGradientDescentTrainer.Summary, typeof(OnlineGradientDescentTrainer), typeof(OnlineGradientDescentTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    OnlineGradientDescentTrainer.UserNameValue,
    OnlineGradientDescentTrainer.LoadNameValue,
    OnlineGradientDescentTrainer.ShortName,
    "sgdr",
    "stochasticgradientdescentregression")]
[assembly: LoadableClass(typeof(void), typeof(OnlineGradientDescentTrainer), null, typeof(SignatureEntryPointModule), "OGD")]

namespace Microsoft.ML.Runtime.Learners
{

    /// <include file='doc.xml' path='doc/members/member[@name="OGD"]/*' />
    public sealed class OnlineGradientDescentTrainer : AveragedLinearTrainer<RegressionPredictionTransformer<LinearRegressionPredictor>, LinearRegressionPredictor>
    {
        internal const string LoadNameValue = "OnlineGradientDescent";
        internal const string UserNameValue = "Stochastic Gradient Descent (Regression)";
        internal const string Summary = "Stochastic gradient descent is an optimization method used to train a wide range of models in machine learning. "
            + "In the TLC implementation of OGD, it is for linear regression.";
        internal const string ShortName = "ogd";

        public sealed class Arguments : AveragedLinearArguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            [TGUI(Label = "Loss Function")]
            public ISupportRegressionLossFactory LossFunction = new SquaredLossFactory();

            /// <summary>
            /// Set defaults that vary from the base type.
            /// </summary>
            public Arguments()
            {
                LearningRate = OgdDefaultArgs.LearningRate;
                DecreaseLearningRate = OgdDefaultArgs.DecreaseLearningRate;
            }

            internal class OgdDefaultArgs : AveragedDefaultArgs
            {
                internal new const float LearningRate = 0.1f;
                internal new const bool DecreaseLearningRate = true;
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
        public OnlineGradientDescentTrainer(IHostEnvironment env,
            string labelColumn,
            string featureColumn,
            float learningRate = Arguments.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = Arguments.OgdDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = Arguments.OgdDefaultArgs.L2RegularizerWeight,
            int numIterations = Arguments.OgdDefaultArgs.NumIterations,
            string weightsColumn = null,
            IRegressionLoss lossFunction = null)
            : base(new Arguments
            {
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLearningRate,
                L2RegularizerWeight = l2RegularizerWeight,
                NumIterations = numIterations,
                LabelColumn = labelColumn,
                FeatureColumn = featureColumn,
                InitialWeights = weightsColumn

            }, env, UserNameValue, TrainerUtils.MakeR4ScalarLabel(labelColumn))
        {
            LossFunction = lossFunction ?? new SquaredLoss();
        }

        internal OnlineGradientDescentTrainer(IHostEnvironment env, Arguments args)
        : base(args, env, UserNameValue, TrainerUtils.MakeR4ScalarLabel(args.LabelColumn))
        {
            LossFunction = args.LossFunction.CreateComponent(env);
        }

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override void CheckLabel(RoleMappedData data)
        {
            data.CheckRegressionLabel();
        }

        protected override LinearRegressionPredictor CreatePredictor()
        {
            Contracts.Assert(WeightsScale == 1);
            VBuffer<float> weights = default(VBuffer<float>);
            float bias;

            if (!Args.Averaged)
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
            return new LinearRegressionPredictor(Host, ref weights, bias);
        }

        [TlcModule.EntryPoint(Name = "Trainers.OnlineGradientDescentRegressor",
            Desc = "Train a Online gradient descent perceptron.",
            UserName = UserNameValue,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/Online/doc.xml' path='doc/members/member[@name=""OGD""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/Online/doc.xml' path='doc/members/example[@name=""OGD""]/*' />"})]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainOGD");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.RegressionOutput>(host, input,
                () => new OnlineGradientDescentTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }

        protected override RegressionPredictionTransformer<LinearRegressionPredictor> MakeTransformer(LinearRegressionPredictor model, Schema trainSchema)
        => new RegressionPredictionTransformer<LinearRegressionPredictor>(Host, model, trainSchema, FeatureColumn.Name);
    }
}
