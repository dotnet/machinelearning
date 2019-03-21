// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(FastForestRegressionTrainer.Summary, typeof(FastForestRegressionTrainer), typeof(FastForestRegressionTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastForestRegressionTrainer.UserNameValue,
    FastForestRegressionTrainer.LoadNameValue,
    FastForestRegressionTrainer.ShortName)]

[assembly: LoadableClass(typeof(FastForestRegressionModelParameters), null, typeof(SignatureLoadModel),
    "FastForest Regression Executor",
    FastForestRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    public sealed class FastForestRegressionModelParameters :
        TreeEnsembleModelParametersBasedOnQuantileRegressionTree,
        IQuantileValueMapper,
        IQuantileRegressionPredictor
    {
        private sealed class QuantileStatistics
        {
            private readonly float[] _data;
            private readonly float[] _weights;

            //This holds the cumulative sum of _weights to search the rank easily by binary search.
            private float[] _weightedSums;
            private SummaryStatistics _summaryStatistics;

            /// <summary>
            /// data array will be modified because of sorting if it is not already sorted yet and this class owns the data.
            /// Modifying the data outside will lead to erroneous output by this class
            /// </summary>
            public QuantileStatistics(float[] data, float[] weights = null, bool isSorted = false)
            {
                Contracts.CheckValue(data, nameof(data));
                Contracts.Check(weights == null || weights.Length == data.Length, "weights");

                _data = data;
                _weights = weights;

                if (!isSorted)
                    Array.Sort(_data);
                else
                    Contracts.Assert(Utils.IsMonotonicallyIncreasing(_data));
            }

            /// <summary>
            /// There are many ways to estimate quantile. This implementations is based on R-8, SciPy-(1/3,1/3)
            /// https://en.wikipedia.org/wiki/Quantile#Estimating_the_quantiles_of_a_population
            /// </summary>
            public float GetQuantile(float p)
            {
                Contracts.CheckParam(0 <= p && p <= 1, nameof(p), "Probablity argument for Quantile function should be between 0 to 1 inclusive");

                if (_data.Length == 0)
                    return float.NaN;

                if (p == 0 || _data.Length == 1)
                    return _data[0];
                if (p == 1)
                    return _data[_data.Length - 1];

                float h = GetRank(p);

                if (h <= 1)
                    return _data[0];

                if (h >= _data.Length)
                    return _data[_data.Length - 1];

                var hf = (int)h;
                return (float)(_data[hf - 1] + (h - hf) * (_data[hf] - _data[hf - 1]));
            }

            private float GetRank(float p)
            {
                const float oneThird = (float)1 / 3;

                // holds length of the _data array if the weights is null or holds the sum of weights
                float weightedLength = _data.Length;

                if (_weights != null)
                {
                    if (_weightedSums == null)
                    {
                        _weightedSums = new float[_weights.Length];
                        _weightedSums[0] = _weights[0];
                        for (int i = 1; i < _weights.Length; i++)
                            _weightedSums[i] = _weights[i] + _weightedSums[i - 1];
                    }

                    weightedLength = _weightedSums[_weightedSums.Length - 1];
                }

                // This implementations is based on R-8, SciPy-(1/3,1/3)
                // https://en.wikipedia.org/wiki/Quantile#Estimating_the_quantiles_of_a_population
                var h = (_weights == null) ? (weightedLength + oneThird) * p + oneThird : weightedLength * p;

                if (_weights == null)
                    return h;

                return _weightedSums.FindIndexSorted(h);
            }

            private SummaryStatistics SummaryStatistics
            {
                get
                {
                    if (_summaryStatistics == null)
                    {
                        _summaryStatistics = new SummaryStatistics();
                        if (_weights != null)
                        {
                            for (int i = 0; i < _data.Length; i++)
                                _summaryStatistics.Add(_data[i], _weights[i]);
                        }
                        else
                        {
                            for (int i = 0; i < _data.Length; i++)
                                _summaryStatistics.Add(_data[i]);
                        }
                    }

                    return _summaryStatistics;
                }
            }
        }

        private readonly int _quantileSampleCount;

        internal const string LoaderSignature = "FastForestRegressionExec";
        internal const string RegistrationName = "FastForestRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FFORE RE",
                // verWrittenCur: 0x00010001, Initial
                // verWrittenCur: 0x00010002, // InstanceWeights are part of QuantileRegression Tree to support weighted intances
                // verWrittenCur: 0x00010003, // _numFeatures serialized
                // verWrittenCur: 0x00010004, // Ini content out of predictor
                // verWrittenCur: 0x00010005, // Add _defaultValueForMissing
                verWrittenCur: 0x00010006, // Categorical splits.
                verReadableCur: 0x00010005,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FastForestRegressionModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010003;

        private protected override uint VerDefaultValueSerialized => 0x00010005;

        private protected override uint VerCategoricalSplitSerialized => 0x00010006;

        internal FastForestRegressionModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs, int samplesCount)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
            _quantileSampleCount = samplesCount;
        }

        private FastForestRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
            // *** Binary format ***
            // bool: should be always true
            // int: Quantile sample count
            Contracts.Check(ctx.Reader.ReadBoolByte());
            _quantileSampleCount = ctx.Reader.ReadInt32();
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bool: always true
            // int: Quantile sample count
            // Previously we store quantileEnabled parameter here,
            // but this paramater always should be true for regression.
            // If you update model version feel free to delete it.
            ctx.Writer.WriteBoolByte(true);
            ctx.Writer.Write(_quantileSampleCount);
        }

        private static FastForestRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new FastForestRegressionModelParameters(env, ctx);
        }

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        private protected override void Map(in VBuffer<float> src, ref float dst)
        {
            int inputVectorSize = InputType.GetVectorSize();
            if (inputVectorSize > 0)
                Host.Check(src.Length == inputVectorSize);
            else
                Host.Check(src.Length > MaxSplitFeatIdx);

            dst = (float)TrainedEnsemble.GetOutput(in src) / TrainedEnsemble.NumTrees;
        }

        ValueMapper<VBuffer<float>, VBuffer<float>> IQuantileValueMapper.GetMapper(float[] quantiles)
        {
            return
                (in VBuffer<float> src, ref VBuffer<float> dst) =>
                {
                    // REVIEW: Should make this more efficient - it repeatedly allocates too much stuff.
                    float[] weights = null;
                    var distribution = TrainedEnsemble.GetDistribution(in src, _quantileSampleCount, out weights);
                    QuantileStatistics qdist = new QuantileStatistics(distribution, weights);

                    var editor = VBufferEditor.Create(ref dst, quantiles.Length);
                    for (int i = 0; i < quantiles.Length; i++)
                        editor.Values[i] = qdist.GetQuantile((float)quantiles[i]);
                    dst = editor.Commit();
                };
        }

        ISchemaBindableMapper IQuantileRegressionPredictor.CreateMapper(Double[] quantiles)
        {
            Host.CheckNonEmpty(quantiles, nameof(quantiles));
            return new SchemaBindableQuantileRegressionPredictor(this, quantiles);
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a decision tree regression model using Fast Forest.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="FastForest_remarks"]/*' />
    public sealed partial class FastForestRegressionTrainer
        : RandomForestTrainerBase<FastForestRegressionTrainer.Options, RegressionPredictionTransformer<FastForestRegressionModelParameters>, FastForestRegressionModelParameters>
    {
        /// <summary>
        /// Options for the <see cref="FastForestRegressionTrainer"/>.
        /// </summary>
        public sealed class Options : FastForestOptionsBase
        {
            /// <summary>
            /// Whether to shuffle the labels on every iteration.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Shuffle the labels on every iteration. " +
                "Useful probably only if using this tree as a tree leaf featurizer for multiclass.")]
            public bool ShuffleLabels;
        }

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        internal const string Summary = "Trains a random forest to fit target values using least-squares.";
        internal const string LoadNameValue = "FastForestRegression";
        internal const string UserNameValue = "Fast Forest Regression";
        internal const string ShortName = "ffr";

        /// <summary>
        /// Initializes a new instance of <see cref="FastForestRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The optional name for the column containing the example weight.</param>
        /// <param name="numberOfLeaves">The max number of leaves in each regression tree.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of documents allowed in a leaf of a regression tree, out of the subsampled data.</param>
        internal FastForestRegressionTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf)
            : base(env, TrainerUtils.MakeR4ScalarColumn(labelColumnName), featureColumnName, exampleWeightColumnName, null, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf)
        {
            Host.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Host.CheckNonEmpty(featureColumnName, nameof(featureColumnName));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastForestRegressionTrainer"/> by using the <see cref="Options"/> class.
        /// </summary>
        /// <param name="env">The instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        internal FastForestRegressionTrainer(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName), true)
        {
        }

        private protected override FastForestRegressionModelParameters TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;
            ValidData = context.ValidationSet;
            TestData = context.TestSet;

            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(trainData, nameof(trainData));
                trainData.CheckRegressionLabel();
                trainData.CheckFeatureFloatVector();
                trainData.CheckOptFloatWeight();
                FeatureCount = trainData.Schema.Feature.Value.Type.GetValueCount();
                ConvertData(trainData);
                TrainCore(ch);
            }
            return new FastForestRegressionModelParameters(Host, TrainedEnsemble, FeatureCount, InnerOptions, FastTreeTrainerOptions.NumberOfQuantileSamples);
        }

        private protected override void PrepareLabels(IChannel ch)
        {
        }

        private protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch)
        {
            return ObjectiveFunctionImplBase.Create(TrainSet, FastTreeTrainerOptions);
        }

        private protected override Test ConstructTestForTrainingData()
        {
            return new RegressionTest(ConstructScoreTracker(TrainSet));
        }

        private protected override RegressionPredictionTransformer<FastForestRegressionModelParameters> MakeTransformer(FastForestRegressionModelParameters model, DataViewSchema trainSchema)
         => new RegressionPredictionTransformer<FastForestRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="FastForestRegressionTrainer"/> using both training and validation data, returns
        /// a <see cref="RegressionPredictionTransformer{FastForestRegressionModelParameters}"/>.
        /// </summary>
        public RegressionPredictionTransformer<FastForestRegressionModelParameters> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private abstract class ObjectiveFunctionImplBase : RandomForestObjectiveFunction
        {
            private readonly float[] _labels;

            public static ObjectiveFunctionImplBase Create(Dataset trainData, Options options)
            {
                if (options.ShuffleLabels)
                    return new ShuffleImpl(trainData, options);
                return new BasicImpl(trainData, options);
            }

            private ObjectiveFunctionImplBase(Dataset trainData, Options options)
                : base(trainData, options, double.MaxValue) // No notion of maximum step size.
            {
                _labels = FastTreeRegressionTrainer.GetDatasetRegressionLabels(trainData);
                Contracts.Assert(_labels.Length == trainData.NumDocs);
            }

            protected override void GetGradientInOneQuery(int query, int threadIndex)
            {
                int begin = Dataset.Boundaries[query];
                int end = Dataset.Boundaries[query + 1];
                for (int i = begin; i < end; ++i)
                    Gradient[i] = _labels[i];
            }

            private sealed class ShuffleImpl : ObjectiveFunctionImplBase
            {
                private readonly Random _rgen;
                private readonly int _labelLim;

                public ShuffleImpl(Dataset trainData, Options options)
                    : base(trainData, options)
                {
                    Contracts.AssertValue(options);
                    Contracts.Assert(options.ShuffleLabels);

                    _rgen = new Random(0); // Ideally we'd get this from the host.

                    for (int i = 0; i < _labels.Length; ++i)
                    {
                        var lab = _labels[i];
                        if (!(0 <= lab && lab < Utils.ArrayMaxSize))
                        {
                            throw Contracts.ExceptUserArg(nameof(options.ShuffleLabels),
                                "Label {0} for example {1} outside of allowed range" +
                                "[0,{2}) when doing shuffled labels", lab, i, Utils.ArrayMaxSize);
                        }
                        int lim = (int)lab + 1;
                        Contracts.Assert(1 <= lim && lim <= Utils.ArrayMaxSize);
                        if (lim > _labelLim)
                            _labelLim = lim;
                    }
                }

                public override double[] GetGradient(IChannel ch, double[] scores)
                {
                    // Each time we get the gradient in random forest regression, it means
                    // we are building a new tree. Shuffle the targets!!
                    int[] map = Utils.GetRandomPermutation(_rgen, _labelLim);
                    for (int i = 0; i < _labels.Length; ++i)
                        _labels[i] = map[(int)_labels[i]];

                    return base.GetGradient(ch, scores);
                }
            }

            private sealed class BasicImpl : ObjectiveFunctionImplBase
            {
                public BasicImpl(Dataset trainData, Options options)
                    : base(trainData, options)
                {
                }
            }
        }
    }

    internal static partial class FastForest
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastForestRegressor",
            Desc = FastForestRegressionTrainer.Summary,
            UserName = FastForestRegressionTrainer.LoadNameValue,
            ShortName = FastForestRegressionTrainer.ShortName)]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, FastForestRegressionTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainFastForest");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<FastForestRegressionTrainer.Options, CommonOutputs.RegressionOutput>(host, input,
                () => new FastForestRegressionTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.RowGroupColumnName));
        }
    }
}
