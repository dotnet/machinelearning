// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Numeric;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Training;

[assembly: LoadableClass(AveragedPerceptronTrainer.Summary, typeof(AveragedPerceptronTrainer), typeof(AveragedPerceptronTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    AveragedPerceptronTrainer.UserNameValue,
    AveragedPerceptronTrainer.LoadNameValue, "avgper", AveragedPerceptronTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(AveragedPerceptronTrainer), null, typeof(SignatureEntryPointModule), "AP")]

namespace Microsoft.ML.Trainers.Online
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for the averaged perceptron trainer.
    /// </summary>
    /// <remarks>
    /// The perceptron is a classification algorithm that makes its predictions by finding a separating hyperplane.
    /// For instance, with feature values f0, f1,..., f_D-1, the prediction is given by determining what side of the hyperplane the point falls into.
    /// That is the same as the sign of sigma[0, D-1] (w_i * f_i), where w_0, w_1,..., w_D-1 are the weights computed by the algorithm.
    ///
    /// The perceptron is an online algorithm, which means it processes the instances in the training set one at a time.
    /// It starts with a set of initial weights (zero, random, or initialized from a previous learner). Then, for each example in the training set, the weighted sum of the features (sigma[0, D-1] (w_i * f_i)) is computed.
    /// If this value has the same sign as the label of the current example, the weights remain the same. If they have opposite signs,
    /// the weights vector is updated by either adding or subtracting (if the label is positive or negative, respectively) the feature vector of the current example,
    /// multiplied by a factor 0 &lt; a &lt;= 1, called the learning rate. In a generalization of this algorithm, the weights are updated by adding the feature vector multiplied by the learning rate,
    /// and by the gradient of some loss function (in the specific case described above, the loss is hinge-loss, whose gradient is 1 when it is non-zero).
    ///
    /// In Averaged Perceptron (aka voted-perceptron), for each iteration, i.e. pass through the training data, a weight vector is calculated as explained above.
    /// The final prediction is then calculate by averaging the weighted sum from each weight vector and looking at the sign of the result.
    ///
    /// For more information see <a href="https://en.wikipedia.org/wiki/Perceptron">Wikipedia entry for Perceptron</a>
    /// or <a href="https://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.8200">Large Margin Classification Using the Perceptron Algorithm</a>
    /// </remarks>
    public sealed class AveragedPerceptronTrainer : AveragedLinearTrainer<BinaryPredictionTransformer<LinearBinaryModelParameters>, LinearBinaryModelParameters>
    {
        internal const string LoadNameValue = "AveragedPerceptron";
        internal const string UserNameValue = "Averaged Perceptron";
        internal const string ShortName = "ap";
        internal const string Summary = "Averaged Perceptron Binary Classifier.";

        private readonly Options _args;

        /// <summary>
        /// Options for the averaged perceptron trainer.
        /// </summary>
        public sealed class Options : AveragedLinearOptions
        {
            /// <summary>
            /// A custom <a href="tmpurl_loss">loss</a>.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportClassificationLossFactory LossFunction = new HingeLoss.Options();

            /// <summary>
            /// The <a href="tmpurl_calib">calibrator</a> for producing probabilities. Default is exponential (aka Platt) calibration.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            /// <summary>
            /// The maximum number of examples to use when training the calibrator.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal int MaxCalibrationExamples = 1000000;

            internal override IComponentFactory<IScalarOutputLoss> LossFunctionFactory => LossFunction;
        }

        private sealed class TrainState : AveragedTrainStateBase
        {
            public TrainState(IChannel ch, int numFeatures, LinearModelParameters predictor, AveragedPerceptronTrainer parent)
                : base(ch, numFeatures, predictor, parent)
            {
            }

            public override LinearBinaryModelParameters CreatePredictor()
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

                return new LinearBinaryModelParameters(ParentHost, in weights, bias);
            }
        }

        internal AveragedPerceptronTrainer(IHostEnvironment env, Options options)
            : base(options, env, UserNameValue, TrainerUtils.MakeBoolScalarLabel(options.LabelColumn))
        {
            _args = options;
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
        /// <param name="learningRate">The learning rate. </param>
        /// <param name="decreaseLearningRate">Whether to decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 Regularization Weight.</param>
        /// <param name="numIterations">The number of training iterations.</param>
        internal AveragedPerceptronTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            IClassificationLoss lossFunction = null,
            float learningRate = Options.AveragedDefault.LearningRate,
            bool decreaseLearningRate = Options.AveragedDefault.DecreaseLearningRate,
            float l2RegularizerWeight = Options.AveragedDefault.L2RegularizerWeight,
            int numIterations = Options.AveragedDefault.NumIterations)
            : this(env, new Options
            {
                LabelColumn = labelColumn,
                FeatureColumn = featureColumn,
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLearningRate,
                L2RegularizerWeight = l2RegularizerWeight,
                NumberOfIterations = numIterations,
                LossFunction = new TrivialFactory(lossFunction ?? new HingeLoss())
            })
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

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override bool NeedCalibration => true;

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                // REVIEW AP is currently not calibrating. Add the probability column after fixing the behavior.
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        private protected override void CheckLabels(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        private protected override void CheckLabelCompatible(SchemaShape.Column labelCol)
        {
            Contracts.Assert(labelCol.IsValid);

            Action error =
                () => throw Host.ExceptSchemaMismatch(nameof(labelCol), "label", labelCol.Name, "float, double, bool or KeyType", labelCol.GetTypeString());

            if (labelCol.Kind != SchemaShape.Column.VectorKind.Scalar)
                error();

            if (!labelCol.IsKey && labelCol.ItemType != NumberDataViewType.Single && labelCol.ItemType != NumberDataViewType.Double && !(labelCol.ItemType is BooleanDataViewType))
                error();
        }

        private protected override TrainStateBase MakeState(IChannel ch, int numFeatures, LinearModelParameters predictor)
        {
            return new TrainState(ch, numFeatures, predictor, this);
        }

        private protected override BinaryPredictionTransformer<LinearBinaryModelParameters> MakeTransformer(LinearBinaryModelParameters model, DataViewSchema trainSchema)
        => new BinaryPredictionTransformer<LinearBinaryModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        [TlcModule.EntryPoint(Name = "Trainers.AveragedPerceptronBinaryClassifier",
             Desc = Summary,
             UserName = UserNameValue,
             ShortName = ShortName)]
        internal static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainAP");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new AveragedPerceptronTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }
    }
}
