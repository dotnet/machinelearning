// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers.Online;

[assembly: LoadableClass(LinearSvm.Summary, typeof(LinearSvm), typeof(LinearSvm.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LinearSvm.UserNameValue,
    LinearSvm.LoadNameValue,
    LinearSvm.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(LinearSvm), null, typeof(SignatureEntryPointModule), "LinearSvm")]

namespace Microsoft.ML.Trainers.Online
{
    using Microsoft.ML.Core.Data;
    using TPredictor = LinearBinaryPredictor;

    /// <summary>
    /// Linear SVM that implements PEGASOS for training. See: http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf
    /// </summary>
    public sealed class LinearSvm : OnlineLinearTrainer<BinaryPredictionTransformer<LinearBinaryPredictor>, LinearBinaryPredictor>
    {
        internal const string LoadNameValue = "LinearSVM";
        internal const string ShortName = "svm";
        internal const string UserNameValue = "SVM (Pegasos-Linear)";
        internal const string Summary = "The idea behind support vector machines, is to map the instances into a high dimensional space "
            + "in which instances of the two classes are linearly separable, i.e., there exists a hyperplane such that all the positive examples are on one side of it, "
            + "and all the negative examples are on the other. After this mapping, quadratic programming is used to find the separating hyperplane that maximizes the "
            + "margin, i.e., the minimal distance between it and the instances.";

        internal new readonly Arguments Args;

        public sealed class Arguments : OnlineLinearArguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularizer constant", ShortName = "lambda", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "0.00001-0.1;log;inc:10")]
            [TlcModule.SweepableFloatParamAttribute("Lambda", 0.00001f, 0.1f, 10, isLogScale: true)]
            public Float Lambda = (Float)0.001;

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
            public ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public int MaxCalibrationExamples = 1000000;
        }

        private int _batch;
        private long _numBatchExamples;

        // A vector holding the next update to the model, in the case where we have multiple batch sizes.
        // This vector will remain unused in the case where our batch size is 1, since in that case, the
        // example vector will just be used directly. The semantics of
        // weightsUpdate/weightsUpdateScale/biasUpdate are similar to weights/weightsScale/bias, in that
        // all elements of weightsUpdate are considered to be multiplied by weightsUpdateScale, and the
        // bias update term is not considered to be multiplied by the scale.
        private VBuffer<Float> _weightsUpdate;
        private Float _weightsUpdateScale;
        private Float _biasUpdate;

        protected override bool NeedCalibration => true;

        public LinearSvm(IHostEnvironment env, Arguments args)
            : base(args, env, UserNameValue, MakeLabelColumn(args.LabelColumn))
        {
            Contracts.CheckUserArg(args.Lambda > 0, nameof(args.Lambda), UserErrorPositive);
            Contracts.CheckUserArg(args.BatchSize > 0, nameof(args.BatchSize), UserErrorPositive);

            Args = args;
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false)
            };
        }

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        /// <summary>
        /// Return the raw margin from the decision hyperplane
        /// </summary>
        protected override Float Margin(in VBuffer<Float> feat)
        {
            return Bias + VectorUtils.DotProduct(in feat, in Weights) * WeightsScale;
        }

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
        {
            return new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false);
        }

        protected override void InitCore(IChannel ch, int numFeatures, LinearPredictor predictor)
        {
            base.InitCore(ch, numFeatures, predictor);

            if (Args.NoBias)
                Bias = 0;

            if (predictor == null)
                VBufferUtils.Densify(ref Weights);

            _weightsUpdate = VBufferUtils.CreateEmpty<Float>(numFeatures);
        }

        protected override void BeginIteration(IChannel ch)
        {
            base.BeginIteration(ch);
            BeginBatch();
        }

        private void BeginBatch()
        {
            _batch++;
            _numBatchExamples = 0;
            _biasUpdate = 0;
            _weightsUpdate = new VBuffer<Float>(_weightsUpdate.Length, 0, _weightsUpdate.Values, _weightsUpdate.Indices);
        }

        private void FinishBatch(in VBuffer<Float> weightsUpdate, Float weightsUpdateScale)
        {
            if (_numBatchExamples > 0)
                UpdateWeights(in weightsUpdate, weightsUpdateScale);
            _numBatchExamples = 0;
        }

        /// <summary>
        /// Observe an example and update weights if necessary
        /// </summary>
        protected override void ProcessDataInstance(IChannel ch, in VBuffer<Float> feat, Float label, Float weight)
        {
            base.ProcessDataInstance(ch, in feat, label, weight);

            // compute the update and update if needed
            Float output = Margin(in feat);
            Float trueOutput = (label > 0 ? 1 : -1);
            Float loss = output * trueOutput - 1;

            // Accumulate the update if there is a loss and we have larger batches.
            if (Args.BatchSize > 1 && loss < 0)
            {
                Float currentBiasUpdate = trueOutput * weight;
                _biasUpdate += currentBiasUpdate;
                // Only aggregate in the case where we're handling multiple instances.
                if (_weightsUpdate.Count == 0)
                {
                    VectorUtils.ScaleInto(in feat, currentBiasUpdate, ref _weightsUpdate);
                    _weightsUpdateScale = 1;
                }
                else
                    VectorUtils.AddMult(in feat, currentBiasUpdate, ref _weightsUpdate);
            }

            if (++_numBatchExamples >= Args.BatchSize)
            {
                if (Args.BatchSize == 1 && loss < 0)
                {
                    Contracts.Assert(_weightsUpdate.Count == 0);
                    // If we aren't aggregating multiple instances, just use the instance's
                    // vector directly.
                    Float currentBiasUpdate = trueOutput * weight;
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
        private void UpdateWeights(in VBuffer<Float> weightsUpdate, Float weightsUpdateScale)
        {
            Contracts.Assert(_batch > 0);

            // REVIEW: This is really odd - normally lambda is small, so the learning rate is initially huge!?!?!
            // Changed from the paper's recommended rate = 1 / (lambda * t) to rate = 1 / (1 + lambda * t).
            Float rate = 1 / (1 + Args.Lambda * _batch);

            // w_{t+1/2} = (1 - eta*lambda) w_t + eta/k * totalUpdate
            WeightsScale *= 1 - rate * Args.Lambda;
            ScaleWeightsIfNeeded();
            VectorUtils.AddMult(in weightsUpdate, rate * weightsUpdateScale / (_numBatchExamples * WeightsScale), ref Weights);

            Contracts.Assert(!Args.NoBias || Bias == 0);
            if (!Args.NoBias)
                Bias += rate / _numBatchExamples * _biasUpdate;

            // w_{t+1} = min{1, 1/sqrt(lambda)/|w_{t+1/2}|} * w_{t+1/2}
            if (Args.PerformProjection)
            {
                Float normalizer = 1 / (MathUtils.Sqrt(Args.Lambda) * VectorUtils.Norm(Weights) * Math.Abs(WeightsScale));
                if (normalizer < 1)
                {
                    // REVIEW: Why would we not scale _bias if we're scaling the weights?
                    WeightsScale *= normalizer;
                    ScaleWeightsIfNeeded();
                    //_bias *= normalizer;
                }
            }
        }

        protected override TPredictor CreatePredictor()
        {
            Contracts.Assert(WeightsScale == 1);
            return new LinearBinaryPredictor(Host, in Weights, Bias);
        }

        [TlcModule.EntryPoint(Name = "Trainers.LinearSvmBinaryClassifier", Desc = "Train a linear SVM.", UserName = UserNameValue, ShortName = ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainLinearSvm(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLinearSVM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LinearSvm(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }

        protected override BinaryPredictionTransformer<LinearBinaryPredictor> MakeTransformer(LinearBinaryPredictor model, Schema trainSchema)
        => new BinaryPredictionTransformer<LinearBinaryPredictor>(Host, model, trainSchema, FeatureColumn.Name);
    }
}
