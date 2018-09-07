// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Core.Data;

[assembly: LoadableClass(AveragedPerceptronTrainer.Summary, typeof(AveragedPerceptronTrainer), typeof(AveragedPerceptronTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    AveragedPerceptronTrainer.UserNameValue,
    AveragedPerceptronTrainer.LoadNameValue, "avgper", AveragedPerceptronTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(AveragedPerceptronTrainer), null, typeof(SignatureEntryPointModule), "AP")]

namespace Microsoft.ML.Runtime.Learners
{
    // This is an averaged perceptron classifier.
    // Configurable subcomponents:
    //     - Loss function. By default, hinge loss (aka max-margin avgd perceptron)
    //     - Feature normalization. By default, rescaling between min and max values for every feature
    //     - Prediction calibration to produce probabilities. Off by default, if on, uses exponential (aka Platt) calibration.
    /// <include file='doc.xml' path='doc/members/member[@name="AP"]/*' />
    public sealed class AveragedPerceptronTrainer : AveragedLinearTrainer<BinaryPredictionTransformer<LinearBinaryPredictor> , LinearBinaryPredictor>
    {
        public const string LoadNameValue = "AveragedPerceptron";
        internal const string UserNameValue = "Averaged Perceptron";
        internal const string ShortName = "ap";
        internal const string Summary = "Averaged Perceptron Binary Classifier.";

        private readonly Arguments _args;

        public class Arguments : AveragedLinearArguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportClassificationLossFactory LossFunction = new HingeLoss.Arguments();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public int MaxCalibrationExamples = 1000000;
        }

        public AveragedPerceptronTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, UserNameValue, MakeLabelColumn(args.LabelColumn))
        {
            _args = args;
            LossFunction = _args.LossFunction.CreateComponent(env);

            OutputColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false)
            };
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        protected override bool NeedCalibration => true;

        protected override SchemaShape.Column[] OutputColumns { get; }

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
        {
            return new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false);
        }

        protected override LinearBinaryPredictor CreatePredictor()
        {
            Contracts.Assert(WeightsScale == 1);

            VBuffer<Float> weights = default(VBuffer<Float>);
            Float bias;

            if (!_args.Averaged)
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

            return new LinearBinaryPredictor(Host, ref weights, bias);
        }

        protected override BinaryPredictionTransformer<LinearBinaryPredictor> MakeTransformer(LinearBinaryPredictor model, ISchema trainSchema)
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