// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.CpuMath.FactorizationMachine;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(FieldAwareFactorizationMachineTrainer.Summary, typeof(FieldAwareFactorizationMachineTrainer),
    typeof(FieldAwareFactorizationMachineTrainer.Options), new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) }
    , FieldAwareFactorizationMachineTrainer.UserName, FieldAwareFactorizationMachineTrainer.LoadName,
    FieldAwareFactorizationMachineTrainer.ShortName, DocName = "trainer/FactorizationMachine.md")]

[assembly: LoadableClass(typeof(void), typeof(FieldAwareFactorizationMachineTrainer), null, typeof(SignatureEntryPointModule), FieldAwareFactorizationMachineTrainer.LoadName)]

namespace Microsoft.ML.Trainers
{
    /*
     Train a field-aware factorization machine using ADAGRAD (an advanced stochastic gradient method). See references below
     for details. This trainer is essentially faster than the one introduced in [2] because of some implementation tricks in [3].
     [1] http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
     [2] https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
     [3] https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
    */
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> to predict a target using a field-aware factorization machine model trained using a stochastic gradient method.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-binary-classification.md)]
    /// To create this trainer, use [FieldAwareFactorizationMachine](xref:Microsoft.ML.FactorizationMachineExtensions.FieldAwareFactorizationMachine(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,System.String,System.String,System.String))
    /// [FieldAwareFactorizationMachine](xref:Microsoft.ML.FactorizationMachineExtensions.FieldAwareFactorizationMachine(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,System.String[],System.String,System.String)),
    /// or [FieldAwareFactorizationMachine(Options)](xref:Microsoft.ML.FactorizationMachineExtensions.FieldAwareFactorizationMachine(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,Microsoft.ML.Trainers.FieldAwareFactorizationMachineTrainer.Options)).
    ///
    /// In contrast to other binary classifiers, which can only support one feature column, field-aware factorization machine can consume multiple feature columns.
    /// Each column is viewed as a container of some features and such a container is called a field.
    /// Note that all feature columns must be float vectors but their dimensions can be different.
    /// The motivation of splitting features into different fields is to model features from different distributions independently.
    /// For example, in online game store, features created from user profile and those from game profile can be assigned to two different fields.
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Binary classification |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | No |
    ///
    /// ### Background
    /// Factorization machine family is a powerful model group for supervised learning problems.
    /// It was first introduced in Steffen Rendle's [Factorization Machines](http://ieeexplore.ieee.org/document/5694074/?reload=true) paper in 2010.
    /// Later, one of its generalized versions, field-aware factorization machine, became an important predictive module in recent recommender systems and click-through rate prediction contests.
    /// For examples, see winning solutions in Steffen Rendle's KDD-Cup 2012 ([Track 1](http://www.kdd.org/kdd-cup/view/kdd-cup-2012-track-1) and [Track 2](http://www.kdd.org/kdd-cup/view/kdd-cup-2012-track-2)),
    /// [Criteo's](https://www.kaggle.com/c/criteo-display-ad-challenge), [Avazu's](https://www.kaggle.com/c/avazu-ctr-prediction), and [Outbrain's](https://www.kaggle.com/c/outbrain-click-prediction) click prediction challenges on Kaggle.
    ///
    /// Factorization machines are especially powerful when feature conjunctions are extremely correlated to the signal you want to predict.
    /// An example of feature pairs which can form important conjunctions is user ID and music ID in music recommendation.
    /// When a dataset consists of only dense numerical features, usage of factorization machine is not recommended or some featurizations should be performed.
    ///
    /// ### Scoring Function
    /// Field-aware factorization machine is a scoring function which maps feature vectors from different fields to a scalar score.
    /// Assume that all $m$ feature columns are concatenated into a long feature vector $\textbf{x} \in {\mathbb R}^n$ and ${\mathcal F}(j)$ denotes the $j$-th feature's field indentifier.
    /// The corresponding score is $\hat{y}(\textbf{x}) = \langle \textbf{w}, \textbf{x} \rangle + \sum_{j = 1}^n \sum_{j' = j + 1}^n \langle \textbf{v}\_{j, {\mathcal F}(j')}, \textbf{v}\_{j', {\mathcal F}(j)} \rangle x_j x_{j'}$,
    /// where $\langle \cdot, \cdot \rangle$ is the inner product operator, $\textbf{w} \in {\mathbb R}^n$ stores the linear coefficients, and $\textbf{v}_{j, f}\in {\mathbb R}^k$ is the $j$-th feature's representation in the $f$-th field's latent space.
    /// Note that $k$ is the latent dimension specified by the user.
    ///
    /// The predicted label is the sign of $\hat{y}$. If $\hat{y} > 0$, this model predicts true. Otherwise, it predicts false.
    ///
    /// For a systematic introduction to field-aware factorization machine, please see [this paper](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
    ///
    /// ### Training Algorithm Details
    /// The algorithm implemented in <xref:Microsoft.ML.Trainers.FieldAwareFactorizationMachineTrainer> is based on [a stochastic gradient method](http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
    /// Algorithm details is described in Algorithm 3 in [this online document](https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf).
    /// The minimized loss function is [logistic loss](https://en.wikipedia.org/wiki/Loss_functions_for_classification), so the trained model can be viewed as a non-linear logistic regression.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.FactorizationMachineExtensions.FieldAwareFactorizationMachine(BinaryClassificationCatalog.BinaryClassificationTrainers, string, string, string)"/>
    /// <seealso cref="Microsoft.ML.FactorizationMachineExtensions.FieldAwareFactorizationMachine(BinaryClassificationCatalog.BinaryClassificationTrainers, string[], string, string)"/>
    /// <seealso cref="Microsoft.ML.FactorizationMachineExtensions.FieldAwareFactorizationMachine(BinaryClassificationCatalog.BinaryClassificationTrainers, FieldAwareFactorizationMachineTrainer.Options)"/>
    /// <seealso cref="FieldAwareFactorizationMachineTrainer.Options"/>
    public sealed class FieldAwareFactorizationMachineTrainer : ITrainer<FieldAwareFactorizationMachineModelParameters>,
        IEstimator<FieldAwareFactorizationMachinePredictionTransformer>
    {
        internal const string Summary = "Train a field-aware factorization machine for binary classification";
        internal const string UserName = "Field-aware Factorization Machine";
        internal const string LoadName = "FieldAwareFactorizationMachine";
        internal const string ShortName = "ffm";

        /// <summary>
        /// <see cref="Options"/> for <see cref="FieldAwareFactorizationMachineTrainer"/> as used in
        /// <see cref="Microsoft.ML.FactorizationMachineExtensions.FieldAwareFactorizationMachine(BinaryClassificationCatalog.BinaryClassificationTrainers, FieldAwareFactorizationMachineTrainer.Options)"/>.
        /// </summary>
        public sealed class Options : TrainerInputBaseWithWeight
        {
            /// <summary>
            /// Initial learning rate.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial learning rate", ShortName = "lr", SortOrder = 1)]
            [TlcModule.SweepableFloatParam(0.001f, 1.0f, isLogScale: true)]
            public float LearningRate = (float)0.1;

            /// <summary>
            /// Number of training iterations.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of training iterations", ShortName = "iters,iter", SortOrder = 2)]
            [TlcModule.SweepableLongParam(1, 100)]
            public int NumberOfIterations = 5;

            /// <summary>
            /// Latent space dimension.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Latent space dimension", ShortName = "d", SortOrder = 3)]
            [TlcModule.SweepableLongParam(4, 100)]
            public int LatentDimension = 20;

            /// <summary>
            /// Regularization coefficient of linear weights.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularization coefficient of linear weights", ShortName = "lambdaLinear", SortOrder = 4)]
            [TlcModule.SweepableFloatParam(1e-8f, 1f, isLogScale: true)]
            public float LambdaLinear = 0.0001f;

            /// <summary>
            /// Regularization coefficient of latent weights.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regularization coefficient of latent weights", ShortName = "lambdaLatent", SortOrder = 5)]
            [TlcModule.SweepableFloatParam(1e-8f, 1f, isLogScale: true)]
            public float LambdaLatent = 0.0001f;

            /// <summary>
            /// Whether to normalize the input vectors so that the concatenation of all fields' feature vectors is unit-length.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to normalize the input vectors so that the concatenation of all fields' feature vectors is unit-length", ShortName = "norm", SortOrder = 6)]
            public new bool NormalizeFeatures = true;

            /// <summary>
            /// Extra feature column names. The column named <see cref="TrainerInputBase.FeatureColumnName"/> stores features from the first field.
            /// The i-th string in <see cref="ExtraFeatureColumns"/> stores the name of the (i+1)-th field's feature column.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Extra columns to use for feature vectors. The i-th specified string denotes the column containing features form the (i+1)-th field." +
                " Note that the first field is specified by \"feat\" instead of \"exfeat\".",
                ShortName = "exfeat", SortOrder = 7)]
            public string[] ExtraFeatureColumns;

            /// <summary>
            /// Whether to shuffle for each training iteration.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to shuffle for each training iteration", ShortName = "shuf", SortOrder = 90)]
            public bool Shuffle = true;

            /// <summary>
            /// Report traning progress or not.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Report traning progress or not", ShortName = "verbose", SortOrder = 91)]
            public bool Verbose = true;

            /// <summary>
            /// Radius of initial latent factors.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Radius of initial latent factors", ShortName = "rad", SortOrder = 110)]
            [TlcModule.SweepableFloatParam(0.1f, 1f)]
            public float Radius = 0.5f;
        }

        private readonly IHost _host;

        PredictionKind ITrainer.PredictionKind => PredictionKind.BinaryClassification;

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
        TrainerInfo ITrainer.Info => _info;
        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, supportValid: true, supportIncrementalTrain: true);

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
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);

            Initialize(env, options);
            var extraColumnLength = (options.ExtraFeatureColumns != null ? options.ExtraFeatureColumns.Length : 0);
            // There can be multiple feature columns in FFM, jointly specified by args.FeatureColumnName and args.ExtraFeatureColumns.
            FeatureColumns = new SchemaShape.Column[1 + extraColumnLength];

            // Treat the default feature column as the 1st field.
            FeatureColumns[0] = new SchemaShape.Column(options.FeatureColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            // Add 2nd, 3rd, and other fields from a FFM-specific argument, args.ExtraFeatureColumns.
            for (int i = 0; i < extraColumnLength; i++)
                FeatureColumns[i + 1] = new SchemaShape.Column(options.ExtraFeatureColumns[i], SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            LabelColumn = new SchemaShape.Column(options.LabelColumnName, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false);
            WeightColumn = options.ExampleWeightColumnName != null ? new SchemaShape.Column(options.ExampleWeightColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false) : default;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FieldAwareFactorizationMachineTrainer"/>.
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="featureColumnNames">The name of column hosting the features. The i-th element stores feature column of the i-th field.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="exampleWeightColumnName">The name of the weight column (optional).</param>
        [BestFriend]
        internal FieldAwareFactorizationMachineTrainer(IHostEnvironment env,
            string[] featureColumnNames,
            string labelColumnName = DefaultColumnNames.Label,
            string exampleWeightColumnName = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);

            var args = new Options();

            Initialize(env, args);

            FeatureColumns = new SchemaShape.Column[featureColumnNames.Length];

            for (int i = 0; i < featureColumnNames.Length; i++)
                FeatureColumns[i] = new SchemaShape.Column(featureColumnNames[i], SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            LabelColumn = new SchemaShape.Column(labelColumnName, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false);
            WeightColumn = exampleWeightColumnName != null ? new SchemaShape.Column(exampleWeightColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false) : default;
        }

        /// <summary>
        /// Initializes the instance. Shared between the two constructors.
        /// REVIEW: Once the legacy constructor goes away, this can move to the only constructor and most of the fields can be back to readonly.
        /// </summary>
        /// <param name="env"></param>
        /// <param name="options"></param>
        private void Initialize(IHostEnvironment env, Options options)
        {
            _host.CheckUserArg(options.LatentDimension > 0, nameof(options.LatentDimension), "Must be positive");
            _host.CheckUserArg(options.LambdaLinear >= 0, nameof(options.LambdaLinear), "Must be non-negative");
            _host.CheckUserArg(options.LambdaLatent >= 0, nameof(options.LambdaLatent), "Must be non-negative");
            _host.CheckUserArg(options.LearningRate > 0, nameof(options.LearningRate), "Must be positive");
            _host.CheckUserArg(options.NumberOfIterations >= 0, nameof(options.NumberOfIterations), "Must be non-negative");
            _latentDim = options.LatentDimension;
            _latentDimAligned = FieldAwareFactorizationMachineUtils.GetAlignedVectorLength(_latentDim);
            _lambdaLinear = options.LambdaLinear;
            _lambdaLatent = options.LambdaLatent;
            _learningRate = options.LearningRate;
            _numIterations = options.NumberOfIterations;
            _norm = options.NormalizeFeatures;
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
                var rng = _host.Rand;
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

            var columns = new List<DataViewSchema.Column>(featureColumns);
            columns.Add(data.Schema.Label.Value);
            if (data.Schema.Weight != null)
                columns.Add(data.Schema.Weight.Value);

            using (var cursor = data.Data.GetRowCursor(columns))
            {
                var labelGetter = RowCursorUtils.GetLabelGetter(cursor, data.Schema.Label.Value.Index);
                var weightGetter = data.Schema.Weight.HasValue ? cursor.GetGetter<float>(data.Schema.Weight.Value) : null;
                for (int f = 0; f < featureColumns.Count; f++)
                    getters[f] = cursor.GetGetter<VBuffer<float>>(featureColumns[f]);
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
            _host.AssertValue(ch);
            _host.AssertValue(pch);

            data.CheckBinaryLabel();
            var featureColumns = data.Schema.GetColumns(RoleMappedSchema.ColumnRole.Feature);
            int fieldCount = featureColumns.Count;
            int totalFeatureCount = 0;
            int[] fieldColumnIndexes = new int[fieldCount];
            for (int f = 0; f < fieldCount; f++)
            {
                var col = featureColumns[f];
                _host.Assert(!col.IsHidden);
                if (!(col.Type is VectorDataViewType vectorType) ||
                    !vectorType.IsKnownSize ||
                    vectorType.ItemType != NumberDataViewType.Single)
                    throw ch.ExceptParam(nameof(data), "Training feature column '{0}' must be a known-size vector of Single, but has type: {1}.", col.Name, col.Type);
                _host.Assert(vectorType.Size > 0);
                fieldColumnIndexes[f] = col.Index;
                totalFeatureCount += vectorType.Size;
            }
            ch.Check(checked(totalFeatureCount * fieldCount * _latentDimAligned) <= Utils.ArrayMaxSize, "Latent dimension or the number of fields too large");
            if (predictor != null)
            {
                ch.Check(predictor.FeatureCount == totalFeatureCount, "Input model's feature count mismatches training feature count");
                ch.Check(predictor.LatentDimension == _latentDim, "Input model's latent dimension mismatches trainer's");
            }
            if (validData != null)
            {
                validData.CheckBinaryLabel();
                var validFeatureColumns = data.Schema.GetColumns(RoleMappedSchema.ColumnRole.Feature);
                _host.Assert(fieldCount == validFeatureColumns.Count);
                for (int f = 0; f < fieldCount; f++)
                {
                    var featCol = featureColumns[f];
                    var validFeatCol = validFeatureColumns[f];
                    _host.Assert(featCol.Name == validFeatCol.Name);
                    _host.Assert(featCol.Type == validFeatCol.Type);
                }
            }
            bool shuffle = _shuffle;
            if (shuffle && !data.Data.CanShuffle)
            {
                ch.Warning("Training data does not support shuffling, so ignoring request to shuffle");
                shuffle = false;
            }
            var rng = shuffle ? _host.Rand : null;
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
                    var weightGetter = data.Schema.Weight?.Index is int weightIdx ? RowCursorUtils.GetGetterAs<float>(NumberDataViewType.Single, cursor, weightIdx) : null;
                    for (int i = 0; i < fieldCount; i++)
                        featureGetters[i] = cursor.GetGetter<VBuffer<float>>(cursor.Schema[fieldColumnIndexes[i]]);
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

            return new FieldAwareFactorizationMachineModelParameters(_host, _norm, fieldCount, totalFeatureCount, _latentDim, linearWeights, latentWeightsAligned);
        }

        private FieldAwareFactorizationMachineModelParameters Train(TrainContext context)
        {
            _host.CheckValue(context, nameof(context));
            var initPredictor = context.InitialPredictor as FieldAwareFactorizationMachineModelParameters;
            _host.CheckParam(context.InitialPredictor == null || initPredictor != null, nameof(context),
                "Initial predictor should have been " + nameof(FieldAwareFactorizationMachineModelParameters));

            using (var ch = _host.Start("Training"))
            using (var pch = _host.StartProgressChannel("Training"))
            {
                return TrainCore(ch, pch, context.TrainingSet, context.ValidationSet, initPredictor);
            }
        }

        IPredictor ITrainer.Train(TrainContext context) => Train(context);
        FieldAwareFactorizationMachineModelParameters ITrainer<FieldAwareFactorizationMachineModelParameters>.Train(TrainContext context) => Train(context);

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
            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, input, () => new FieldAwareFactorizationMachineTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName));
        }

        /// <summary>
        /// Continues the training of a <see cref="FieldAwareFactorizationMachineTrainer"/> using an already trained <paramref name="modelParameters"/> and/or validation data,
        /// and returns a <see cref="FieldAwareFactorizationMachinePredictionTransformer"/>.
        /// </summary>
        public FieldAwareFactorizationMachinePredictionTransformer Fit(IDataView trainData,
            IDataView validationData = null, FieldAwareFactorizationMachineModelParameters modelParameters = null)
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

            using (var ch = _host.Start("Training"))
            using (var pch = _host.StartProgressChannel("Training"))
            {
                model = TrainCore(ch, pch, trainingData, validData, modelParameters);
            }

            return new FieldAwareFactorizationMachinePredictionTransformer(_host, model, trainData.Schema, FeatureColumns.Select(x => x.Name).ToArray());
        }

        /// <summary> Trains and returns a <see cref="FieldAwareFactorizationMachinePredictionTransformer"/>.</summary>
        public FieldAwareFactorizationMachinePredictionTransformer Fit(IDataView input) => Fit(input, null, null);

        /// <summary>
        /// Schema propagation for transformers. Returns the output schema of the data, if
        /// the input schema is like the one provided.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {

            _host.CheckValue(inputSchema, nameof(inputSchema));

            void CheckColumnsCompatible(SchemaShape.Column column, string columnRole)
            {

                if (!inputSchema.TryFindColumn(column.Name, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), columnRole, column.Name);

                if (!column.IsCompatibleWith(col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), columnRole, column.Name,
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
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }
    }
}
