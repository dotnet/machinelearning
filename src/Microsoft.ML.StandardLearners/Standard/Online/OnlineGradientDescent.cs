// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

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
                LearningRate = (Float)0.1;
                DecreaseLearningRate = true;
            }
        }

        public OnlineGradientDescentTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, UserNameValue, MakeLabelColumn(args.LabelColumn))
        {
            LossFunction = args.LossFunction.CreateComponent(env);

            OutputColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false)
            };
        }

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        protected override SchemaShape.Column[] OutputColumns { get; }

        protected override void CheckLabel(RoleMappedData data)
        {
            data.CheckRegressionLabel();
        }

        protected override LinearRegressionPredictor CreatePredictor()
        {
            Contracts.Assert(WeightsScale == 1);
            VBuffer<Float> weights = default(VBuffer<Float>);
            Float bias;

            if (!Args.Averaged)
            {
                Weights.CopyTo(ref weights);
                bias = Bias;
            }
            else
            {
                TotalWeights.CopyTo(ref weights);
                VectorUtils.ScaleBy(ref weights, 1 / (Float)NumWeightUpdates);
                bias = TotalBias / (Float)NumWeightUpdates;
            }
            return new LinearRegressionPredictor(Host, ref weights, bias);
        }

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
        {
            return new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);
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

        protected override RegressionPredictionTransformer<LinearRegressionPredictor> MakeTransformer(LinearRegressionPredictor model, ISchema trainSchema)
        => new RegressionPredictionTransformer<LinearRegressionPredictor>(Host, model, trainSchema, FeatureColumn.Name);
    }
}
