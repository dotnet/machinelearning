﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.FactorizationMachine;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Training;

[assembly: LoadableClass(FieldAwareFactorizationMachineTrainer.Summary, typeof(FieldAwareFactorizationMachineTrainer),
    typeof(FieldAwareFactorizationMachineTrainer.Options), new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) }
    , FieldAwareFactorizationMachineTrainer.UserName, FieldAwareFactorizationMachineTrainer.LoadName,
    FieldAwareFactorizationMachineTrainer.ShortName, DocName = "trainer/FactorizationMachine.md")]

[assembly: LoadableClass(typeof(void), typeof(FieldAwareFactorizationMachineTrainer), null, typeof(SignatureEntryPointModule), FieldAwareFactorizationMachineTrainer.LoadName)]

namespace Microsoft.ML.FactorizationMachine
{
    /*
     Train a field-aware factorization machine using ADAGRAD (an advanced stochastic gradient method). See references below
     for details. This trainer is essentially faster the one introduced in [2] because of some implementation tricks[3].
     [1] http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
     [2] https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
     [3] https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
    */
    /// <include file='doc.xml' path='doc/members/member[@name="FieldAwareFactorizationMachineBinaryClassifier"]/*' />
    public sealed class FieldAwareFactorizationMachineTrainer : TrainerBase<FieldAwareFactorizationMachineModelParameters>,
        IEstimator<FieldAwareFactorizationMachinePredictionTransformer>
    {
        internal const string Summary = "Train a field-aware factorization machine for binary classification";
        internal const string UserName = "Field-aware Factorization Machine";
        internal const string LoadName = "FieldAwareFactorizationMachine";
        internal const string ShortName = "ffm";

        public sealed class Options : LearnerInputBaseWithWeight
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

            /// <summary>
            /// Extra feature column names. The column named <see cref="LearnerInputBase.FeatureColumn"/> stores features from the first field.
            /// The i-th string in <see cref="ExtraFeatureColumns"/> stores the name of the (i+1)-th field's feature column.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Extra columns to use for feature vectors. The i-th specified string denotes the column containing features form the (i+1)-th field." +
                " Note that the first field is specified by \"feat\" instead of \"exfeat\".",
                ShortName = "exfeat", SortOrder = 7)]
            public string[] ExtraFeatureColumns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to shuffle for each training iteration", ShortName = "shuf", SortOrder = 90)]
            public bool Shuffle = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Report traning progress or not", ShortName = "verbose", SortOrder = 91)]
            public bool Verbose = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Radius of initial latent factors", ShortName = "rad", SortOrder = 110)]
            [TlcModule.SweepableFloatParam(0.1f, 1f)]
            public float Radius = 0.5f;
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        /// <summary>
        /// The feature column that the trainer expects.
        /// </summary>
        internal readonly SchemaShape.Column[] FeatureColumns;

        /// <summary>
        /// The label column that the trainer expects. Can be <c>null</c>, which indicates that label
        /// is not used for training.
        /// </summary>
        internal readonly SchemaShape.Column LabelColumn;

        /// <summary>
        /// The weight column that the trainer expects. Can be <c>null</c>, which indicates that weight is
        /// not used for training.
        /// </summary>
        internal readonly SchemaShape.Column WeightColumn;

        /// <summary>
        /// The <see cref="TrainerInfo"/> containing at least the training data for this trainer.
        /// </summary>
        public override TrainerInfo Info { get; }

        private int _latentDim;
        private int _latentDimAligned;
        private float _lambdaLinear;
        private float _lambdaLatent;
        private float _learningRate;
        private int _numIterations;
        private bool _norm;
        private bool _shuffle;
        private bool _verbose;
        private float _radius;

        /// <summary>
        /// Initializes a new instance of <see cref="FieldAwareFactorizationMachineTrainer"/> through the <see cref="Options"/> class.
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="options">An instance of the legacy <see cref="Options"/> to apply advanced parameters to the algorithm.</param>
        [BestFriend]
        internal FieldAwareFactorizationMachineTrainer(IHostEnvironment env, Options options)
            : base(env, LoadName)
        {
            Initialize(env, options);
            Info = new TrainerInfo(supportValid: true, supportIncrementalTrain: true);
            var extraColumnLength = (options.ExtraFeatureColumns != null ? options.ExtraFeatureColumns.Length : 0);
            // There can be multiple feature columns in FFM, jointly specified by args.FeatureColumn and args.ExtraFeatureColumns.
            FeatureColumns = new SchemaShape.Column[1 + extraColumnLength];

            // Treat the default feature column as the 1st field.
            FeatureColumns[0] = new SchemaShape.Column(options.FeatureColumn, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);

            // Add 2nd, 3rd, and other fields from a FFM-specific argument, args.ExtraFeatureColumns.
            for (int i = 0; i < extraColumnLength; i++)
                FeatureColumns[i + 1] = new SchemaShape.Column(options.ExtraFeatureColumns[i], SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);

            LabelColumn = new SchemaShape.Column(options.LabelColumn, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false);
            WeightColumn = options.WeightColumn.IsExplicit ? new SchemaShape.Column(options.WeightColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false) : default;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FieldAwareFactorizationMachineTrainer"/>.
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="featureColumns">The name of column hosting the features. The i-th element stores feature column of the i-th field.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="weights">The name of the optional weights' column.</param>
        [BestFriend]
        internal FieldAwareFactorizationMachineTrainer(IHostEnvironment env,
            string[] featureColumns,
            string labelColumn = DefaultColumnNames.Label,
            string weights = null)
            : base(env, LoadName)
        {
            var args = new Options();

            Initialize(env, args);
            Info = new TrainerInfo(supportValid: true, supportIncrementalTrain: true);

            FeatureColumns = new SchemaShape.Column[featureColumns.Length];

            for (int i = 0; i < featureColumns.Length; i++)
                FeatureColumns[i] = new SchemaShape.Column(featureColumns[i], SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);

            LabelColumn = new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false);
            WeightColumn = weights != null ? new SchemaShape.Column(weights, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false) : default;
        }

        /// <summary>
        /// Initializes the instance. Shared between the two constructors.
        /// REVIEW: Once the legacy constructor goes away, this can move to the only constructor and most of the fields can be back to readonly.
        /// </summary>
        /// <param name="env"></param>
        /// <param name="options"></param>
        private void Initialize(IHostEnvironment env, Options options)
        {
            Host.CheckUserArg(options.LatentDim > 0, nameof(options.LatentDim), "Must be positive");
            Host.CheckUserArg(options.LambdaLinear >= 0, nameof(options.LambdaLinear), "Must be non-negative");
            Host.CheckUserArg(options.LambdaLatent >= 0, nameof(options.LambdaLatent), "Must be non-negative");
            Host.CheckUserArg(options.LearningRate > 0, nameof(options.LearningRate), "Must be positive");
            Host.CheckUserArg(options.Iters >= 0, nameof(options.Iters), "Must be non-negative");
            _latentDim = options.LatentDim;
            _latentDimAligned = FieldAwareFactorizationMachineUtils.GetAlignedVectorLength(_latentDim);
            _lambdaLinear = options.LambdaLinear;
            _lambdaLatent = options.LambdaLatent;
            _learningRate = options.LearningRate;
            _numIterations = options.Iters;
            _norm = options.Norm;
            _shuffle = options.Shuffle;
            _verbose = options.Verbose;
            _radius = options.Radius;
        }

        private void InitializeTrainingState(int fieldCount, int featureCount, FieldAwareFactorizationMachineModelParameters predictor, out float[] linearWeights,
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
            var getters = new ValueGetter<VBuffer<float>>[featureColumns.Count];
            float label = 0;
            float weight = 1;
            double loss = 0;
            float modelResponse = 0;
            long exampleCount = 0;
            badExampleCount = 0;
            int count = 0;

            var columns = featureColumns.Append(data.Schema.Label.Value);
            if (data.Schema.Weight != null)
                columns.Append(data.Schema.Weight.Value);

            using (var cursor = data.Data.GetRowCursor(columns))
            {
                var labelGetter = RowCursorUtils.GetLabelGetter(cursor, data.Schema.Label.Value.Index);
                var weightGetter = data.Schema.Weight?.Index is int weightIdx ? cursor.GetGetter<float>(weightIdx) : null;
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

        private FieldAwareFactorizationMachineModelParameters TrainCore(IChannel ch, IProgressChannel pch, RoleMappedData data,
            RoleMappedData validData = null, FieldAwareFactorizationMachineModelParameters predictor = null)
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
                Host.Assert(!col.IsHidden);
                if (!(col.Type is VectorType vectorType) ||
                    !vectorType.IsKnownSize ||
                    vectorType.ItemType != NumberType.Float)
                    throw ch.ExceptParam(nameof(data), "Training feature column '{0}' must be a known-size vector of R4, but has type: {1}.", col.Name, col.Type);
                Host.Assert(vectorType.Size > 0);
                fieldColumnIndexes[f] = col.Index;
                totalFeatureCount += vectorType.Size;
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
                {
                    var featCol = featureColumns[f];
                    var validFeatCol = validFeatureColumns[f];
                    Host.Assert(featCol.Name == validFeatCol.Name);
                    Host.Assert(featCol.Type == validFeatCol.Type);
                }
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

            var columns = data.Schema.Schema.Where(x => fieldColumnIndexes.Contains(x.Index)).ToList();
            columns.Add(data.Schema.Label.Value);
            if (data.Schema.Weight != null)
                columns.Add(data.Schema.Weight.Value);

            InitializeTrainingState(fieldCount, totalFeatureCount, predictor, out float[] linearWeights,
                out AlignedArray latentWeightsAligned, out float[] linearAccSqGrads, out AlignedArray latentAccSqGradsAligned);

            // refer to Algorithm 3 in https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
            while (iter++ < _numIterations)
            {
                using (var cursor = data.Data.GetRowCursor(columns, rng))
                {
                    var labelGetter = RowCursorUtils.GetLabelGetter(cursor, data.Schema.Label.Value.Index);
                    var weightGetter = data.Schema.Weight?.Index is int weightIdx ? RowCursorUtils.GetGetterAs<float>(NumberType.R4, cursor, weightIdx) : null;
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

            return new FieldAwareFactorizationMachineModelParameters(Host, _norm, fieldCount, totalFeatureCount, _latentDim, linearWeights, latentWeightsAligned);
        }

        private protected override FieldAwareFactorizationMachineModelParameters Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var initPredictor = context.InitialPredictor as FieldAwareFactorizationMachineModelParameters;
            Host.CheckParam(context.InitialPredictor == null || initPredictor != null, nameof(context),
                "Initial predictor should have been " + nameof(FieldAwareFactorizationMachineModelParameters));

            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                return TrainCore(ch, pch, context.TrainingSet, context.ValidationSet, initPredictor);
            }
        }

        [TlcModule.EntryPoint(Name = "Trainers.FieldAwareFactorizationMachineBinaryClassifier",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName)]
        internal static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Train a field-aware factorization machine");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            return LearnerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, input, () => new FieldAwareFactorizationMachineTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }

        public FieldAwareFactorizationMachinePredictionTransformer Train(IDataView trainData,
            IDataView validationData = null, FieldAwareFactorizationMachineModelParameters initialPredictor = null)
        {
            FieldAwareFactorizationMachineModelParameters model = null;

            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            foreach (var feat in FeatureColumns)
                roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, feat.Name));

            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Label, LabelColumn.Name));

            if (WeightColumn.IsValid)
                roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, WeightColumn.Name));

            var trainingData = new RoleMappedData(trainData, roles);
            var validData = validationData == null ? null : new RoleMappedData(validationData, roles);

            using (var ch = Host.Start("Training"))
            using (var pch = Host.StartProgressChannel("Training"))
            {
                model = TrainCore(ch, pch, trainingData, validData, initialPredictor as FieldAwareFactorizationMachineModelParameters);
            }

            return new FieldAwareFactorizationMachinePredictionTransformer(Host, model, trainData.Schema, FeatureColumns.Select(x => x.Name).ToArray());
        }

        public FieldAwareFactorizationMachinePredictionTransformer Fit(IDataView input) => Train(input);

        /// <summary>
        /// Schema propagation for transformers. Returns the output schema of the data, if
        /// the input schema is like the one provided.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {

            Host.CheckValue(inputSchema, nameof(inputSchema));

            void CheckColumnsCompatible(SchemaShape.Column column, string columnRole)
            {

                if (!inputSchema.TryFindColumn(column.Name, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), columnRole, column.Name);

                if (!column.IsCompatibleWith(col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), columnRole, column.Name,
                        column.GetTypeString(), col.GetTypeString());
            }

            CheckColumnsCompatible(LabelColumn, "label");

            foreach (var feat in FeatureColumns)
            {
                CheckColumnsCompatible(feat, "feature");
            }

            if (WeightColumn.IsValid)
                CheckColumnsCompatible(WeightColumn, "weight");

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            foreach (var col in GetOutputColumnsCore(inputSchema))
                outColumns[col.Name] = col;

            return new SchemaShape(outColumns.Values);
        }

        private SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }
    }
}
