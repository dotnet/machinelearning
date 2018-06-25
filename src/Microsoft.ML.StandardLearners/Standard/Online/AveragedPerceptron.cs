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

[assembly: LoadableClass(AveragedPerceptronTrainer.Summary, typeof(AveragedPerceptronTrainer), typeof(AveragedPerceptronTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    AveragedPerceptronTrainer.UserNameValue,
    AveragedPerceptronTrainer.LoadNameValue, "avgper", AveragedPerceptronTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(AveragedPerceptronTrainer), null, typeof(SignatureEntryPointModule), "AP")]

namespace Microsoft.ML.Runtime.Learners
{
    /// <summary>
    /// This is an averaged perceptron classifier.
    /// Configurable subcomponents:
    ///     - Loss function. By default, hinge loss (aka max-margin avgd perceptron)
    ///     - Feature normalization. By default, rescaling between min and max values for every feature
    ///     - Prediction calibration to produce probabilities. Off by default, if on, uses exponential (aka Platt) calibration.
    /// </summary>
    public sealed class AveragedPerceptronTrainer :
        AveragedLinearTrainer<AveragedPerceptronTrainer.Arguments, LinearBinaryPredictor>
    {
        public const string LoadNameValue = "AveragedPerceptron";
        internal const string UserNameValue = "Averaged Perceptron";
        internal const string ShortName = "ap";
        internal const string Summary = "Perceptron is a binary classification algorithm that makes its predictions based on a linear function.";
        internal const string DetailedSummary = @"Perceptron is a classification algorithm that makes its predictions based on a linear function.
I.e., for an instance with feature values f0, f1,..., f_D-1, , the prediction is given by the sign of sigma[0,D-1] ( w_i * f_i), where w_0, w_1,...,w_D-1 are the weights computed by the algorithm.

Perceptron is an online algorithm, i.e., it processes the instances in the training set one at a time.
The weights are initialized to be 0, or some random values. Then, for each example in the training set, the value of sigma[0, D-1] (w_i * f_i) is computed. 
If this value has the same sign as the label of the current example, the weights remain the same. If they have opposite signs,
the weights vector is updated by either subtracting or adding (if the label is negative or positive, respectively) the feature vector of the current example,
multiplied by a factor 0 < a <= 1, called the learning rate. In a generalization of this algorithm, the weights are updated by adding the feature vector multiplied by the learning rate, 
and by the gradient of some loss function (in the specific case described above, the loss is hinge-loss, whose gradient is 1 when it is non-zero).

In Averaged Perceptron (AKA voted-perceptron), the weight vectors are stored, 
together with a weight that counts the number of iterations it survived (this is equivalent to storing the weight vector after every iteration, regardless of whether it was updated or not).
The prediction is then calculated by taking the weighted average of all the sums sigma[0, D-1] (w_i * f_i) or the different weight vectors.";

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
            : base(args, env, UserNameValue)
        {
            LossFunction = Args.LossFunction.CreateComponent(env);
        }

        public override bool NeedCalibration
        {
            get { return true; }
        }

        public override PredictionKind PredictionKind { get { return PredictionKind.BinaryClassification; } }

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        public override LinearBinaryPredictor CreatePredictor()
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

            return new LinearBinaryPredictor(Host, ref weights, bias);
        }

        [TlcModule.EntryPoint(Name = "Trainers.AveragedPerceptronBinaryClassifier", Desc = DetailedSummary, UserName = UserNameValue, ShortName = ShortName)]
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