// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers.Online;
using System;

[assembly: LoadableClass(AveragedPerceptronTrainer.Summary, typeof(AveragedPerceptronTrainer), typeof(AveragedPerceptronTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    AveragedPerceptronTrainer.UserNameValue,
    AveragedPerceptronTrainer.LoadNameValue, "avgper", AveragedPerceptronTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(AveragedPerceptronTrainer), null, typeof(SignatureEntryPointModule), "AP")]

namespace Microsoft.ML.Trainers.Online
{
    // This is an averaged perceptron classifier.
    // Configurable subcomponents:
    //     - Loss function. By default, hinge loss (aka max-margin avgd perceptron)
    //     - Feature normalization. By default, rescaling between min and max values for every feature
    //     - Prediction calibration to produce probabilities. Off by default, if on, uses exponential (aka Platt) calibration.
    /// <include file='doc.xml' path='doc/members/member[@name="AP"]/*' />
    public sealed class AveragedPerceptronTrainer : AveragedLinearTrainer<BinaryPredictionTransformer<LinearBinaryPredictor>, LinearBinaryPredictor>
    {
        public const string LoadNameValue = "AveragedPerceptron";
        internal const string UserNameValue = "Averaged Perceptron";
        internal const string ShortName = "ap";
        internal const string Summary = "Averaged Perceptron Binary Classifier.";

        private readonly Arguments _args;

        public sealed class Arguments : AveragedLinearArguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportClassificationLossFactory LossFunction = new HingeLoss.Arguments();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public int MaxCalibrationExamples = 1000000;

            internal override IComponentFactory<IScalarOutputLoss> LossFunctionFactory => LossFunction;
        }

        private sealed class TrainState : AveragedTrainStateBase
        {
            public TrainState(IChannel ch, int numFeatures, LinearPredictor predictor, AveragedPerceptronTrainer parent)
                : base(ch, numFeatures, predictor, parent)
            {
            }

            public override LinearBinaryPredictor CreatePredictor()
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

                return new LinearBinaryPredictor(ParentHost, in weights, bias);
            }
        }

        internal AveragedPerceptronTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, UserNameValue, TrainerUtils.MakeBoolScalarLabel(args.LabelColumn))
        {
            _args = args;
            LossFunction = _args.LossFunction.CreateComponent(env);
        }

        /// <summary>
        /// Trains a linear binary classifier using the averaged perceptron.
        /// <a href='https://en.wikipedia.org/wiki/Perceptron'>Wikipedia entry for Perceptron</a>
        /// </summary>
        /// <param name="env">The local instance of the <see cref="IHostEnvironment"/></param>
        /// <param name="lossFunction">The classification loss function. </param>
        /// <param name="labelColumn">The name of the label column. </param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The optional name of the weights column.</param>
        /// <param name="learningRate">The learning rate. </param>
        /// <param name="decreaseLearningRate">Wheather to decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 Regularization Weight.</param>
        /// <param name="numIterations">The number of training iteraitons.</param>
        /// <param name="advancedSettings">A delegate to supply more advanced arguments to the algorithm.</param>
        public AveragedPerceptronTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            IClassificationLoss lossFunction = null,
            float learningRate = Arguments.AveragedDefaultArgs.LearningRate,
            bool decreaseLearningRate = Arguments.AveragedDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = Arguments.AveragedDefaultArgs.L2RegularizerWeight,
            int numIterations = Arguments.AveragedDefaultArgs.NumIterations,
            Action<Arguments> advancedSettings = null)
            : this(env, InvokeAdvanced(advancedSettings, new Arguments
            {
                LabelColumn = labelColumn,
                FeatureColumn = featureColumn,
                InitialWeights = weights,
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLearningRate,
                L2RegularizerWeight = l2RegularizerWeight,
                NumIterations = numIterations,
                LossFunction = new TrivialFactory(lossFunction ?? new HingeLoss())
            }))
        {
        }

        private sealed class TrivialFactory : ISupportClassificationLossFactory
        {
            private IClassificationLoss _loss;

            public TrivialFactory(IClassificationLoss loss)
            {
                _loss = loss;
            }

            IClassificationLoss IComponentFactory<IClassificationLoss>.CreateComponent(IHostEnvironment env) => _loss;
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        protected override bool NeedCalibration => true;

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                // REVIEW AP is currently not calibrating. Add the probability column after fixing the behavior.
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override void CheckLabels(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        protected override void CheckLabelCompatible(SchemaShape.Column labelCol)
        {
            Contracts.AssertValue(labelCol);

            Action error =
                () => throw Host.ExceptSchemaMismatch(nameof(labelCol), RoleMappedSchema.ColumnRole.Label.Value, labelCol.Name, "BL, R8, R4 or a Key", labelCol.GetTypeString());

            if (labelCol.Kind != SchemaShape.Column.VectorKind.Scalar)
                error();

            if (!labelCol.IsKey && labelCol.ItemType != NumberType.R4 && labelCol.ItemType != NumberType.R8 && !labelCol.ItemType.IsBool)
                error();
        }

        private protected override TrainStateBase MakeState(IChannel ch, int numFeatures, LinearPredictor predictor)
        {
            return new TrainState(ch, numFeatures, predictor, this);
        }

        protected override BinaryPredictionTransformer<LinearBinaryPredictor> MakeTransformer(LinearBinaryPredictor model, Schema trainSchema)
        => new BinaryPredictionTransformer<LinearBinaryPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        [TlcModule.EntryPoint(Name = "Trainers.AveragedPerceptronBinaryClassifier",
            Desc = Summary,
            UserName = UserNameValue,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/Online/doc.xml' path='doc/members/member[@name=""AP""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/Online/doc.xml' path='doc/members/example[@name=""AP""]/*' />"})]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainAP");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new AveragedPerceptronTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }
    }
}