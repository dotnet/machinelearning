// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(LinearSvmTrainer.Summary, typeof(LinearSvmTrainer), typeof(LinearSvmTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LinearSvmTrainer.UserNameValue,
    LinearSvmTrainer.LoadNameValue,
    LinearSvmTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(LinearSvmTrainer), null, typeof(SignatureEntryPointModule), "LinearSvm")]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Linear SVM that implements PEGASOS for training. See: http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf
    /// </summary>
    public sealed class LinearSvmTrainer : OnlineLinearTrainer<BinaryPredictionTransformer<LinearBinaryModelParameters>, LinearBinaryModelParameters>
    {
        internal const string LoadNameValue = "LinearSVM";
        internal const string ShortName = "svm";
        internal const string UserNameValue = "SVM (Pegasos-Linear)";
        internal const string Summary = "The idea behind support vector machines, is to map the instances into a high dimensional space "
            + "in which instances of the two classes are linearly separable, i.e., there exists a hyperplane such that all the positive examples are on one side of it, "
            + "and all the negative examples are on the other. After this mapping, quadratic programming is used to find the separating hyperplane that maximizes the "
            + "margin, i.e., the minimal distance between it and the instances.";

        internal readonly Options Opts;

        public sealed class Options : OnlineLinearOptions
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularizer constant", ShortName = "lambda", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "0.00001-0.1;log;inc:10")]
            [TlcModule.SweepableFloatParamAttribute("Lambda", 0.00001f, 0.1f, 10, isLogScale: true)]
            public float Lambda = 0.001f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Batch size", ShortName = "batch", SortOrder = 190)]
            [TGUI(Label = "Batch Size")]
            public int BatchSize = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Perform projection to unit-ball? Typically used with batch size > 1.", ShortName = "project", SortOrder = 50)]
            [TlcModule.SweepableDiscreteParam("PerformProjection", null, isBool: true)]
            public bool PerformProjection = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "No bias")]
            [TlcModule.SweepableDiscreteParam("NoBias", null, isBool: true)]
            public bool NoBias = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal int MaxCalibrationExamples = 1000000;

            /// <summary>
            /// Column to use for example weight.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight,WeightColumn", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string ExampleWeightColumnName = null;
        }

        private sealed class TrainState : TrainStateBase
        {
            private int _batch;
            private long _numBatchExamples;
            // A vector holding the next update to the model, in the case where we have multiple batch sizes.
            // This vector will remain unused in the case where our batch size is 1, since in that case, the
            // example vector will just be used directly. The semantics of
            // weightsUpdate/weightsUpdateScale/biasUpdate are similar to weights/weightsScale/bias, in that
            // all elements of weightsUpdate are considered to be multiplied by weightsUpdateScale, and the
            // bias update term is not considered to be multiplied by the scale.
            private VBuffer<float> _weightsUpdate;
            private float _weightsUpdateScale;
            private float _biasUpdate;

            private readonly int _batchSize;
            private readonly bool _noBias;
            private readonly bool _performProjection;
            private readonly float _lambda;

            public TrainState(IChannel ch, int numFeatures, LinearModelParameters predictor, LinearSvmTrainer parent)
                : base(ch, numFeatures, predictor, parent)
            {
                _batchSize = parent.Opts.BatchSize;
                _noBias = parent.Opts.NoBias;
                _performProjection = parent.Opts.PerformProjection;
                _lambda = parent.Opts.Lambda;

                if (_noBias)
                    Bias = 0;

                if (predictor == null)
                    VBufferUtils.Densify(ref Weights);

                _weightsUpdate = VBufferUtils.CreateEmpty<float>(numFeatures);

            }

            public override void BeginIteration(IChannel ch)
            {
                base.BeginIteration(ch);
                BeginBatch();
            }

            private void BeginBatch()
            {
                _batch++;
                _numBatchExamples = 0;
                _biasUpdate = 0;
                VBufferUtils.Resize(ref _weightsUpdate, _weightsUpdate.Length, 0);
            }

            private void FinishBatch(in VBuffer<float> weightsUpdate, float weightsUpdateScale)
            {
                if (_numBatchExamples > 0)
                    UpdateWeights(in weightsUpdate, weightsUpdateScale);
                _numBatchExamples = 0;
            }

            /// <summary>
            /// Observe an example and update weights if necesary.
            /// </summary>
            public override void ProcessDataInstance(IChannel ch, in VBuffer<float> feat, float label, float weight)
            {
                base.ProcessDataInstance(ch, in feat, label, weight);

                // compute the update and update if needed
                float output = Margin(in feat);
                float trueOutput = (label > 0 ? 1 : -1);
                float loss = output * trueOutput - 1;

                // Accumulate the update if there is a loss and we have larger batches.
                if (_batchSize > 1 && loss < 0)
                {
                    float currentBiasUpdate = trueOutput * weight;
                    _biasUpdate += currentBiasUpdate;
                    // Only aggregate in the case where we're handling multiple instances.
                    if (_weightsUpdate.GetValues().Length == 0)
                    {
                        VectorUtils.ScaleInto(in feat, currentBiasUpdate, ref _weightsUpdate);
                        _weightsUpdateScale = 1;
                    }
                    else
                        VectorUtils.AddMult(in feat, currentBiasUpdate, ref _weightsUpdate);
                }

                if (++_numBatchExamples >= _batchSize)
                {
                    if (_batchSize == 1 && loss < 0)
                    {
                        Contracts.Assert(_weightsUpdate.GetValues().Length == 0);
                        // If we aren't aggregating multiple instances, just use the instance's
                        // vector directly.
                        float currentBiasUpdate = trueOutput * weight;
                        _biasUpdate += currentBiasUpdate;
                        FinishBatch(in feat, currentBiasUpdate);
                    }
                    else
                        FinishBatch(in _weightsUpdate, _weightsUpdateScale);
                    BeginBatch();
                }
            }

            /// <summary>
            /// Updates the weights at the end of the batch. Since weightsUpdate can be an instance
            /// feature vector, this function should not change the contents of weightsUpdate.
            /// </summary>
            private void UpdateWeights(in VBuffer<float> weightsUpdate, float weightsUpdateScale)
            {
                Contracts.Assert(_batch > 0);

                // REVIEW: This is really odd - normally lambda is small, so the learning rate is initially huge!?!?!
                // Changed from the paper's recommended rate = 1 / (lambda * t) to rate = 1 / (1 + lambda * t).
                float rate = 1 / (1 + _lambda * _batch);

                // w_{t+1/2} = (1 - eta*lambda) w_t + eta/k * totalUpdate
                WeightsScale *= 1 - rate * _lambda;
                ScaleWeightsIfNeeded();
                VectorUtils.AddMult(in weightsUpdate, rate * weightsUpdateScale / (_numBatchExamples * WeightsScale), ref Weights);

                Contracts.Assert(!_noBias || Bias == 0);
                if (!_noBias)
                    Bias += rate / _numBatchExamples * _biasUpdate;

                // w_{t+1} = min{1, 1/sqrt(lambda)/|w_{t+1/2}|} * w_{t+1/2}
                if (_performProjection)
                {
                    float normalizer = 1 / (MathUtils.Sqrt(_lambda) * VectorUtils.Norm(Weights) * Math.Abs(WeightsScale));
                    if (normalizer < 1)
                    {
                        // REVIEW: Why would we not scale _bias if we're scaling the weights?
                        WeightsScale *= normalizer;
                        ScaleWeightsIfNeeded();
                        //_bias *= normalizer;
                    }
                }
            }

            /// <summary>
            /// Return the raw margin from the decision hyperplane.
            /// </summary>
            public override float Margin(in VBuffer<float> feat)
                => Bias + VectorUtils.DotProduct(in feat, in Weights) * WeightsScale;

            public override LinearBinaryModelParameters CreatePredictor()
            {
                Contracts.Assert(WeightsScale == 1);
                // below should be `in Weights`, but can't because of https://github.com/dotnet/roslyn/issues/29371
                return new LinearBinaryModelParameters(ParentHost, Weights, Bias);
            }
        }

        private protected override bool NeedCalibration => true;

        /// <summary>
        /// Initializes a new instance of <see cref="LinearSvmTrainer"/>.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column. </param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfIterations">The number of training iteraitons.</param>
        [BestFriend]
        internal LinearSvmTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = Options.OnlineDefault.NumberOfIterations)
            : this(env, new Options
            {
                LabelColumnName = labelColumn,
                FeatureColumnName = featureColumn,
                ExampleWeightColumnName = exampleWeightColumnName,
                NumberOfIterations = numberOfIterations,
            })
        {
        }

        internal LinearSvmTrainer(IHostEnvironment env, Options options)
            : base(options, env, UserNameValue, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            Contracts.CheckUserArg(options.Lambda > 0, nameof(options.Lambda), UserErrorPositive);
            Contracts.CheckUserArg(options.BatchSize > 0, nameof(options.BatchSize), UserErrorPositive);

            Opts = options;
        }

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override void CheckLabels(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        private protected override TrainStateBase MakeState(IChannel ch, int numFeatures, LinearModelParameters predictor)
            => new TrainState(ch, numFeatures, predictor, this);

        [TlcModule.EntryPoint(Name = "Trainers.LinearSvmBinaryClassifier", Desc = "Train a linear SVM.", UserName = UserNameValue, ShortName = ShortName)]
        internal static CommonOutputs.BinaryClassificationOutput TrainLinearSvm(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLinearSVM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LinearSvmTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }

        private protected override BinaryPredictionTransformer<LinearBinaryModelParameters> MakeTransformer(LinearBinaryModelParameters model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<LinearBinaryModelParameters>(Host, model, trainSchema, FeatureColumn.Name);
    }
}
