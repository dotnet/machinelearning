// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(FieldAwareFactorizationMachineTrainer.Summary, typeof(FieldAwareFactorizationMachineTrainer), typeof(FieldAwareFactorizationMachineTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) }, FieldAwareFactorizationMachineTrainer.UserName, FieldAwareFactorizationMachineTrainer.LoadName,
    FieldAwareFactorizationMachineTrainer.ShortName, DocName = "trainer/FactorizationMachine.md")]

[assembly: LoadableClass(typeof(void), typeof(FieldAwareFactorizationMachineTrainer), null, typeof(SignatureEntryPointModule), FieldAwareFactorizationMachineTrainer.LoadName)]

namespace Microsoft.ML.Runtime.FactorizationMachine
{
    /*
     Train a field-aware factorization machine using ADAGRAD (an advanced stochastic gradient method). See references below
     for details. This trainer is essentially faster the one introduced in [2] because of some implemtation tricks[3].
     [1] http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
     [2] http://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
     [3] https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
    */
    /// <include file='doc.xml' path='doc/members/member[@name="FieldAwareFactorizationMachineBinaryClassifier"]/*' />
    public sealed class FieldAwareFactorizationMachineTrainer : TrainerBase<RoleMappedData, FieldAwareFactorizationMachinePredictor>,
        IIncrementalTrainer<RoleMappedData, FieldAwareFactorizationMachinePredictor>, IValidatingTrainer<RoleMappedData>,
        IIncrementalValidatingTrainer<RoleMappedData, FieldAwareFactorizationMachinePredictor>
    {
        public const string Summary = "Train a field-aware factorization machine for binary classification";
        public const string UserName = "Field-aware Factorization Machine";
        public const string LoadName = "FieldAwareFactorizationMachine";
        public const string ShortName = "ffm";

        public sealed class Arguments : LearnerInputBaseWithLabel
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial learning rate", ShortName = "lr", SortOrder = 1)]
            [TlcModule.SweepableFloatParam(0.001f, 1.0f, isLogScale: true)]
            public float LearningRate = (float)0.1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of training iterations", ShortName = "iter", SortOrder = 2)]
            [TlcModule.SweepableLongParam(1, 100)]
            public int Iters = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Latent space dimension", ShortName = "d", SortOrder = 3)]
            [TlcModule.SweepableLongParam(4, 100)]
            public int LatentDim = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularization coefficient of linear weights", ShortName = "lambdaLinear", SortOrder = 4)]
            [TlcModule.SweepableFloatParam(1e-8f, 1f, isLogScale: true)]
            public float LambdaLinear = 0.0001f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularization coefficient of latent weights", ShortName = "lambdaLatent", SortOrder = 5)]
            [TlcModule.SweepableFloatParam(1e-8f, 1f, isLogScale: true)]
            public float LambdaLatent = 0.0001f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to normalize the input vectors so that the concatenation of all fields' feature vectors is unit-length", ShortName = "norm", SortOrder = 6)]
            public bool Norm = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to shuffle for each training iteration", ShortName = "shuf", SortOrder = 90)]
            public bool Shuffle = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Report traning progress or not", ShortName = "verbose", SortOrder = 91)]
            public bool Verbose = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Radius of initial latent factors", ShortName = "rad", SortOrder = 110)]
            [TlcModule.SweepableFloatParam(0.1f, 1f)]
            public float Radius = 0.5f;
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        public override bool NeedNormalization => true;
        public override bool NeedCalibration => false;
        public override bool WantCaching => true;
        private readonly int _latentDim;
        private readonly int _latentDimAligned;
        private readonly float _lambdaLinear;
        private readonly float _lambdaLatent;
        private readonly float _learningRate;
        private readonly int _numIterations;
        private readonly bool _norm;
        private readonly bool _shuffle;
        private readonly bool _verbose;
        private readonly float _radius;
        private FieldAwareFactorizationMachinePredictor _pred;

        public FieldAwareFactorizationMachineTrainer(IHostEnvironment env, Arguments args) : base(env, LoadName)
        {
            Host.CheckUserArg(args.LatentDim > 0, nameof(args.LatentDim), "Must be positive");
            Host.CheckUserArg(args.LambdaLinear >= 0, nameof(args.LambdaLinear), "Must be non-negative");
            Host.CheckUserArg(args.LambdaLatent >= 0, nameof(args.LambdaLatent), "Must be non-negative");
            Host.CheckUserArg(args.LearningRate > 0, nameof(args.LearningRate), "Must be positive");
            Host.CheckUserArg(args.Iters >= 0, nameof(args.Iters), "Must be non-negative");
            _latentDim = args.LatentDim;
            _latentDimAligned = FieldAwareFactorizationMachineUtils.GetAlignedVectorLength(_latentDim);
            _lambdaLinear = args.LambdaLinear;
            _lambdaLatent = args.LambdaLatent;
            _learningRate = args.LearningRate;
            _numIterations = args.Iters;
            _norm = args.Norm;
            _shuffle = args.Shuffle;
            _verbose = args.Verbose;
            _radius = args.Radius;
        }

        private void InitializeTrainingState(int fieldCount, int featureCount, FieldAwareFactorizationMachinePredictor predictor, out float[] linearWeights,
            out AlignedArray latentWeightsAligned, out float[] linearAccumulatedSquaredGrads, out AlignedArray latentAccumulatedSquaredGradsAligned)
        {
            linearWeights = new float[featureCount];
            latentWeightsAligned = new AlignedArray(featureCount * fieldCount * _latentDimAligned, 16);
            linearAccumulatedSquaredGrads = new float[featureCount];
            latentAccumulatedSquaredGradsAligned = new AlignedArray(featureCount * fieldCount * _latentDimAligned, 16);

            if (predictor == null)
            {
                var rng = Host.Rand;
                for (int j = 0; j < featureCount; j++)
                {
                    linearWeights[j] = 0;
                    linearAccumulatedSquaredGrads[j] = 1;
                    for (int f = 0; f < fieldCount; f++)
                    {
                        int vBias = j * fieldCount * _latentDimAligned + f * _latentDimAligned;
                        for (int k = 0; k < _latentDimAligned; k++)
                        {
                            if (k < _latentDim)
                                latentWeightsAligned[vBias + k] = _radius * (float)rng.NextDouble();
                            else
                                latentWeightsAligned[vBias + k] = 0;
                            latentAccumulatedSquaredGradsAligned[vBias + k] = 1;
                        }
                    }
                }
            }
            else
            {
                predictor.CopyLinearWeightsTo(linearWeights);
                predictor.CopyLatentWeightsTo(latentWeightsAligned);
                for (int j = 0; j < featureCount; j++)
                {
                    linearAccumulatedSquaredGrads[j] = 1;
                    for (int f = 0; f < fieldCount; f++)
                    {
                        int vBias = j * fieldCount * _latentDimAligned + f * _latentDimAligned;
                        for (int k = 0; k < _latentDimAligned; k++)
                            latentAccumulatedSquaredGradsAligned[vBias + k] = 1;
                    }
                }
            }
        }

        private static float CalculateLoss(float label, float modelResponse)
        {
            float margin = label > 0 ? modelResponse : -modelResponse;
            if (margin > 0)
                return MathUtils.Log(1 + MathUtils.ExpSlow(-margin));
            else
                return -margin + MathUtils.Log(1 + MathUtils.ExpSlow(margin));
        }

        private static float CalculateLossSlope(float label, float modelResponse)
        {
            float sign = label > 0 ? 1 : -1;
            float margin = sign * modelResponse;
            return -sign * MathUtils.Sigmoid(-margin);
        }

        private static double CalculateAvgLoss(IChannel ch, RoleMappedData data, bool norm, float[] linearWeights, AlignedArray latentWeightsAligned,
            int latentDimAligned, AlignedArray latentSum, int[] featureFieldBuffer, int[] featureIndexBuffer, float[] featureValueBuffer, VBuffer<float> buffer, ref long badExampleCount)
        {
            var featureColumns = data.Schema.GetColumns(RoleMappedSchema.ColumnRole.Feature);
            Func<int, bool> pred = c => featureColumns.Select(ci => ci.Index).Contains(c) || c == data.Schema.Label.Index || (data.Schema.Weight != null && c == data.Schema.Weight.Index);
            var getters = new ValueGetter<VBuffer<float>>[featureColumns.Count];
            float label = 0;
            float weight = 1;
            double loss = 0;
            float modelResponse = 0;
            long exampleCount = 0;
            badExampleCount = 0;
            int count = 0;
            using (var cursor = data.Data.GetRowCursor(pred))
            {
                var labelGetter = cursor.GetGetter<float>(data.Schema.Label.Index);
                var weightGetter = data.Schema.Weight == null ? null : cursor.GetGetter<float>(data.Schema.Weight.Index);
                for (int f = 0; f < featureColumns.Count; f++)
                    getters[f] = cursor.GetGetter<VBuffer<float>>(featureColumns[f].Index);
                while (cursor.MoveNext())
                {
                    labelGetter(ref label);
                    weightGetter?.Invoke(ref weight);
                    float annihilation = label - label + weight - weight;
                    if (!FloatUtils.IsFinite(annihilation))
                    {
                        badExampleCount++;
                        continue;
                    }
                    if (!FieldAwareFactorizationMachineUtils.LoadOneExampleIntoBuffer(getters, buffer, norm, ref count,
                        featureFieldBuffer, featureIndexBuffer, featureValueBuffer))
                    {
                        badExampleCount++;
                        continue;
                    }
                    FieldAwareFactorizationMachineInterface.CalculateIntermediateVariables(featureColumns.Count, latentDimAligned, count,
                        featureFieldBuffer, featureIndexBuffer, featureValueBuffer, linearWeights, latentWeightsAligned, latentSum, ref modelResponse);
                    loss += weight * CalculateLoss(label, modelResponse);
                    exampleCount++;
                }
            }
            return loss / exampleCount;
        }

        private void TrainCore(IChannel ch, IProgressChannel pch, RoleMappedData data, RoleMappedData validData, FieldAwareFactorizationMachinePredictor predictor)
        {
            Host.AssertValue(ch);
            Host.AssertValue(pch);

            data.CheckBinaryLabel();
            var featureColumns = data.Schema.GetColumns(RoleMappedSchema.ColumnRole.Feature);
            int fieldCount = featureColumns.Count;
            int totalFeatureCount = 0;
            int[] fieldColumnIndexes = new int[fieldCount];
            for (int f = 0; f < fieldCount; f++)
            {
                var col = featureColumns[f];
                Host.Assert(col.Type.AsVector.VectorSize > 0);
                if (col == null)
                    throw ch.ExceptParam(nameof(data), "Empty feature column not allowed");
                Host.Assert(!data.Schema.Schema.IsHidden(col.Index));
                if (!col.Type.IsKnownSizeVector || col.Type.ItemType != NumberType.Float)
                    throw ch.ExceptParam(nameof(data), "Training feature column '{0}' must be a known-size vector of R4, but has type: {1}.", col.Name, col.Type);
                fieldColumnIndexes[f] = col.Index;
                totalFeatureCount += col.Type.AsVector.VectorSize;
            }
            ch.Check(checked(totalFeatureCount * fieldCount * _latentDimAligned) <= Utils.ArrayMaxSize, "Latent dimension or the number of fields too large");
            if (predictor != null)
            {
                ch.Check(predictor.FeatureCount == totalFeatureCount, "Input model's feature count mismatches training feature count");
                ch.Check(predictor.LatentDim == _latentDim, "Input model's latent dimension mismatches trainer's");
            }
            if (validData != null)
            {
                validData.CheckBinaryLabel();
                var validFeatureColumns = data.Schema.GetColumns(RoleMappedSchema.ColumnRole.Feature);
                Host.Assert(fieldCount == validFeatureColumns.Count);
                for (int f = 0; f < fieldCount; f++)
                    Host.Assert(featureColumns[f] == validFeatureColumns[f]);
            }
            bool shuffle = _shuffle;
            if (shuffle && !data.Data.CanShuffle)
            {
                ch.Warning("Training data does not support shuffling, so ignoring request to shuffle");
                shuffle = false;
            }
            var rng = shuffle ? Host.Rand : null;
            var featureGetters = new ValueGetter<VBuffer<float>>[fieldCount];
            var featureBuffer = new VBuffer<float>();
            var featureValueBuffer = new float[totalFeatureCount];
            var featureIndexBuffer = new int[totalFeatureCount];
            var featureFieldBuffer = new int[totalFeatureCount];
            var latentSum = new AlignedArray(fieldCount * fieldCount * _latentDimAligned, 16);
            var metricNames = new List<string>() { "Training-loss" };
            if (validData != null)
                metricNames.Add("Validation-loss");
            int iter = 0;
            long exampleCount = 0;
            long badExampleCount = 0;
            long validBadExampleCount = 0;
            double loss = 0;
            double validLoss = 0;
            pch.SetHeader(new ProgressHeader(metricNames.ToArray(), new string[] { "iterations", "examples" }), entry =>
            {
                entry.SetProgress(0, iter, _numIterations);
                entry.SetProgress(1, exampleCount);
            });
            Func<int, bool> pred = c => fieldColumnIndexes.Contains(c) || c == data.Schema.Label.Index || (data.Schema.Weight != null && c == data.Schema.Weight.Index);
            InitializeTrainingState(fieldCount, totalFeatureCount, predictor, out float[] linearWeights,
                out AlignedArray latentWeightsAligned, out float[] linearAccSqGrads, out AlignedArray latentAccSqGradsAligned);

            // refer to Algorithm 3 in https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
            while (iter++ < _numIterations)
            {
                using (var cursor = data.Data.GetRowCursor(pred, rng))
                {
                    var labelGetter = RowCursorUtils.GetLabelGetter(cursor, data.Schema.Label.Index);
                    var weightGetter = data.Schema.Weight == null ? null : RowCursorUtils.GetGetterAs<float>(NumberType.R4, cursor, data.Schema.Weight.Index);
                    for (int i = 0; i < fieldCount; i++)
                        featureGetters[i] = cursor.GetGetter<VBuffer<float>>(fieldColumnIndexes[i]);
                    loss = 0;
                    exampleCount = 0;
                    badExampleCount = 0;
                    while (cursor.MoveNext())
                    {
                        float label = 0;
                        float weight = 1;
                        int count = 0;
                        float modelResponse = 0;
                        labelGetter(ref label);
                        weightGetter?.Invoke(ref weight);
                        float annihilation = label - label + weight - weight;
                        if (!FloatUtils.IsFinite(annihilation))
                        {
                            badExampleCount++;
                            continue;
                        }
                        if (!FieldAwareFactorizationMachineUtils.LoadOneExampleIntoBuffer(featureGetters, featureBuffer, _norm, ref count,
                            featureFieldBuffer, featureIndexBuffer, featureValueBuffer))
                        {
                            badExampleCount++;
                            continue;
                        }

                        // refer to Algorithm 1 in [3] https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
                        FieldAwareFactorizationMachineInterface.CalculateIntermediateVariables(fieldCount, _latentDimAligned, count,
                            featureFieldBuffer, featureIndexBuffer, featureValueBuffer, linearWeights, latentWeightsAligned, latentSum, ref modelResponse);
                        var slope = CalculateLossSlope(label, modelResponse);

                        // refer to Algorithm 2 in [3] https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
                        FieldAwareFactorizationMachineInterface.CalculateGradientAndUpdate(_lambdaLinear, _lambdaLatent, _learningRate, fieldCount, _latentDimAligned, weight, count,
                            featureFieldBuffer, featureIndexBuffer, featureValueBuffer, latentSum, slope, linearWeights, latentWeightsAligned, linearAccSqGrads, latentAccSqGradsAligned);
                        loss += weight * CalculateLoss(label, modelResponse);
                        exampleCount++;
                    }
                    loss /= exampleCount;
                }

                if (_verbose)
                {
                    if (validData == null)
                        pch.Checkpoint(loss, iter, exampleCount);
                    else
                    {
                        validLoss = CalculateAvgLoss(ch, validData, _norm, linearWeights, latentWeightsAligned, _latentDimAligned, latentSum,
                            featureFieldBuffer, featureIndexBuffer, featureValueBuffer, featureBuffer, ref validBadExampleCount);
                        pch.Checkpoint(loss, validLoss, iter, exampleCount);
                    }
                }
            }
            if (badExampleCount != 0)
                ch.Warning($"Skipped {badExampleCount} examples with bad label/weight/features in training set");
            if (validBadExampleCount != 0)
                ch.Warning($"Skipped {validBadExampleCount} examples with bad label/weight/features in validation set");
            _pred = new FieldAwareFactorizationMachinePredictor(Host, _norm, fieldCount, totalFeatureCount, _latentDim, linearWeights, latentWeightsAligned);
        }

        public override void Train(RoleMappedData data)
        {
            Host.CheckValue(data, nameof(data));
            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                TrainCore(ch, pch, data, null, null);
                ch.Done();
            }
        }

        public void Train(RoleMappedData data, RoleMappedData validData)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckValue(validData, nameof(validData));
            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                TrainCore(ch, pch, data, validData, null);
                ch.Done();
            }
        }

        public void Train(RoleMappedData data, FieldAwareFactorizationMachinePredictor predictor)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckValue(predictor, nameof(predictor));
            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                TrainCore(ch, pch, data, null, predictor);
                ch.Done();
            }
        }

        public void Train(RoleMappedData data, RoleMappedData validData, FieldAwareFactorizationMachinePredictor predictor)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckValue(data, nameof(validData));
            Host.CheckValue(predictor, nameof(predictor));
            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                TrainCore(ch, pch, data, validData, predictor);
                ch.Done();
            }
        }

        public override FieldAwareFactorizationMachinePredictor CreatePredictor()
        {
            Host.Check(_pred != null, nameof(Train) + " has not yet been called");
            return _pred;
        }

        [TlcModule.EntryPoint(Name = "Trainers.FieldAwareFactorizationMachineBinaryClassifier",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/FactorizationMachine/doc.xml' path='doc/members/member[@name=""FieldAwareFactorizationMachineBinaryClassifier""]/*' />" })]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Train a field-aware factorization machine");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input, () => new FieldAwareFactorizationMachineTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }
}
