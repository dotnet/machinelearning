// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(OnlineGradientDescentTrainer.Summary, typeof(OnlineGradientDescentTrainer), typeof(OnlineGradientDescentTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    OnlineGradientDescentTrainer.UserNameValue,
    OnlineGradientDescentTrainer.LoadNameValue,
    OnlineGradientDescentTrainer.ShortName,
    "sgdr",
    "stochasticgradientdescentregression")]
[assembly: LoadableClass(typeof(void), typeof(OnlineGradientDescentTrainer), null, typeof(SignatureEntryPointModule), "OGD")]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a linear regression model using
    /// <a href="https://en.wikipedia.org/wiki/Gradient_descent">Online Gradient Descent (OGD)</a> for estimating the parameters of the linear regression model.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [OnlineGradientDescent](xref:Microsoft.ML.StandardTrainersCatalog.OnlineGradientDescent(Microsoft.ML.RegressionCatalog.RegressionTrainers,System.String,System.String,Microsoft.ML.Trainers.IRegressionLoss,System.Single,System.Boolean,System.Single,System.Int32))
    /// or [OnlineGradientDescent(Options)](xref:Microsoft.ML.StandardTrainersCatalog.OnlineGradientDescent(Microsoft.ML.RegressionCatalog.RegressionTrainers,Microsoft.ML.Trainers.OnlineGradientDescentTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-regression.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Regression |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | Yes |
    ///
    /// ### Training Algorithm Details
    /// Stochastic gradient descent uses a simple yet efficient iterative technique to fit model coefficients using error gradients for convex loss functions.
    /// Online Gradient Descent (OGD) implements the standard (non-batch) stochastic gradient descent, with a choice of loss functions,
    /// and an option to update the weight vector using the average of the vectors seen over time (averaged argument is set to True by default).
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="StandardTrainersCatalog.OnlineGradientDescent(RegressionCatalog.RegressionTrainers, string, string, IRegressionLoss, float, bool, float, int)"/>
    /// <seealso cref="StandardTrainersCatalog.OnlineGradientDescent(RegressionCatalog.RegressionTrainers, OnlineGradientDescentTrainer.Options)"/>
    /// <seealso cref="Options"/>
    public sealed class OnlineGradientDescentTrainer : AveragedLinearTrainer<RegressionPredictionTransformer<LinearRegressionModelParameters>, LinearRegressionModelParameters>
    {
        internal const string LoadNameValue = "OnlineGradientDescent";
        internal const string UserNameValue = "Stochastic Gradient Descent (Regression)";
        internal const string Summary = "Stochastic gradient descent is an optimization method used to train a wide range of models in machine learning. "
            + "In the TLC implementation of OGD, it is for linear regression.";
        internal const string ShortName = "ogd";

        /// <summary>
        /// Options for the <see cref="OnlineGradientDescentTrainer"/> as used in
        /// [OnlineGradientDescent(Options)](xref:Microsoft.ML.StandardTrainersCatalog.OnlineGradientDescent(Microsoft.ML.RegressionCatalog.RegressionTrainers,Microsoft.ML.Trainers.OnlineGradientDescentTrainer.Options)).
        /// </summary>
        public sealed class Options : AveragedLinearOptions
        {
            [Argument(ArgumentType.Multiple, Name = "LossFunction", HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            [TGUI(Label = "Loss Function")]
            internal ISupportRegressionLossFactory RegressionLossFunctionFactory = new SquaredLossFactory();

            /// <summary>
            /// A custom <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a>.
            /// </summary>
            public IRegressionLoss LossFunction { get; set; }

            internal override IComponentFactory<IScalarLoss> LossFunctionFactory => RegressionLossFunctionFactory;

            /// <summary>
            /// Create a new <see cref="Options"/> object with default values.
            /// </summary>
            public Options()
            {
                LearningRate = OgdDefaultArgs.LearningRate;
                DecreaseLearningRate = OgdDefaultArgs.DecreaseLearningRate;
            }

            [BestFriend]
            internal class OgdDefaultArgs : AveragedDefault
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
        /// <param name="l2Regularization">Weight of L2 regularization term.</param>
        /// <param name="numberOfIterations">Number of training iterations through the data.</param>
        /// <param name="lossFunction">The custom loss functions. Defaults to <see cref="SquaredLoss"/> if not provided.</param>
        internal OnlineGradientDescentTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            float learningRate = Options.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = Options.OgdDefaultArgs.DecreaseLearningRate,
            float l2Regularization = Options.OgdDefaultArgs.L2Regularization,
            int numberOfIterations = Options.OgdDefaultArgs.NumberOfIterations,
            IRegressionLoss lossFunction = null)
            : this(env, new Options
            {
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLearningRate,
                L2Regularization = l2Regularization,
                NumberOfIterations = numberOfIterations,
                LabelColumnName = labelColumn,
                FeatureColumnName = featureColumn,
                LossFunction = lossFunction ?? new SquaredLoss()
            })
        {
        }

        internal OnlineGradientDescentTrainer(IHostEnvironment env, Options options)
        : base(options, env, UserNameValue, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
            LossFunction = options.LossFunction ?? options.LossFunctionFactory.CreateComponent(env);
        }

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
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

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.RegressionOutput>(host, input,
                () => new OnlineGradientDescentTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName));
        }

        private protected override RegressionPredictionTransformer<LinearRegressionModelParameters> MakeTransformer(LinearRegressionModelParameters model, DataViewSchema trainSchema)
        => new RegressionPredictionTransformer<LinearRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);
    }
}
