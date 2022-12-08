// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.OneDal;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(FastForestBinaryTrainer.Summary, typeof(FastForestBinaryTrainer), typeof(FastForestBinaryTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastForestBinaryTrainer.UserNameValue,
    FastForestBinaryTrainer.LoadNameValue,
    "FastForest",
    FastForestBinaryTrainer.ShortName,
    "ffc")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(FastForestBinaryModelParameters), null, typeof(SignatureLoadModel),
    "FastForest Binary Executor",
    FastForestBinaryModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(FastForest), null, typeof(SignatureEntryPointModule), "FastForest")]

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// Base class for fast forest trainer options.
    /// </summary>
    public abstract class FastForestOptionsBase : TreeOptions
    {
        /// <summary>
        /// The number of data points to be sampled from each leaf to find the distribution of labels.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of labels to be sampled from each leaf to make the distribution", ShortName = "qsc")]
        public int NumberOfQuantileSamples = 100;

        internal FastForestOptionsBase()
        {
            FeatureFraction = 0.7;
            BaggingSize = 1;
            FeatureFractionPerSplit = 0.7;
        }
    }

    /// <summary>
    /// Model parameters for <see cref="FastForestBinaryTrainer"/>.
    /// </summary>
    public sealed class FastForestBinaryModelParameters :
        TreeEnsembleModelParametersBasedOnQuantileRegressionTree
    {
        internal const string LoaderSignature = "FastForestBinaryExec";
        internal const string RegistrationName = "FastForestClassificationPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FFORE BC",
                // verWrittenCur: 0x00010001, Initial
                // verWrittenCur: 0x00010002, // InstanceWeights are part of QuantileRegression Tree to support weighted instances
                // verWrittenCur: 0x00010003, // _numFeatures serialized
                // verWrittenCur: 0x00010004, // Ini content out of predictor
                // verWrittenCur: 0x00010005, // Add _defaultValueForMissing
                verWrittenCur: 0x00010006, // Categorical splits.
                verReadableCur: 0x00010005,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FastForestBinaryModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010003;

        private protected override uint VerDefaultValueSerialized => 0x00010005;

        private protected override uint VerCategoricalSplitSerialized => 0x00010006;

        /// <summary>
        /// The type of prediction for this trainer.
        /// </summary>
        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        internal FastForestBinaryModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        { }

        private FastForestBinaryModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        internal static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new FastForestBinaryModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new SchemaBindableCalibratedModelParameters<FastForestBinaryModelParameters, ICalibrator>(env, predictor, calibrator);
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a decision tree binary classification model using Fast Forest.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [FastForest](xref:Microsoft.ML.TreeExtensions.FastForest(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,System.String,System.String,System.String,System.Int32,System.Int32,System.Int32))
    /// or [FastForest(Options)](xref:Microsoft.ML.TreeExtensions.FastForest(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-binary-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Binary classification |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.FastTree |
    /// | Exportable to ONNX | Yes |
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/algo-details-fastforest.md)]
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FastForest(BinaryClassificationCatalog.BinaryClassificationTrainers, string, string, string, int, int, int)"/>
    /// <seealso cref="TreeExtensions.FastForest(BinaryClassificationCatalog.BinaryClassificationTrainers, FastForestBinaryTrainer.Options)"/>
    /// <seealso cref="Options"/>
    public sealed partial class FastForestBinaryTrainer :
        RandomForestTrainerBase<FastForestBinaryTrainer.Options, BinaryPredictionTransformer<FastForestBinaryModelParameters>, FastForestBinaryModelParameters>
    {
        /// <summary>
        /// Options for the <see cref="FastForestBinaryTrainer"/> as used in
        /// [FastForest(Options)](xref:Microsoft.ML.TreeExtensions.FastForest(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer.Options)).
        /// </summary>
        public sealed class Options : FastForestOptionsBase
        {
            /// <summary>
            /// The upper bound on the absolute value of a single tree output.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Upper bound on absolute value of single tree output", ShortName = "mo")]
            public Double MaximumOutputMagnitudePerTree = 100;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal int MaxCalibrationExamples = 1000000;
        }

        internal const string LoadNameValue = "FastForestClassification";
        internal const string UserNameValue = "Fast Forest Classification";
        internal const string Summary = "Uses a random forest learner to perform binary classification.";
        internal const string ShortName = "ff";

        private bool[] _trainSetLabels;

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        private protected override bool NeedCalibration => true;

        /// <summary>
        /// Initializes a new instance of <see cref="FastForestBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name for the column containing the example weight.</param>
        /// <param name="numberOfLeaves">The max number of leaves in each regression tree.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of documents allowed in a leaf of a regression tree, out of the subsampled data.</param>
        internal FastForestBinaryTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf)
            : base(env, TrainerUtils.MakeBoolScalarLabel(labelColumnName), featureColumnName, exampleWeightColumnName, null, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf)
        {
            Host.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Host.CheckNonEmpty(featureColumnName, nameof(featureColumnName));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastForestBinaryTrainer"/> by using the <see cref="Options"/> class.
        /// </summary>
        /// <param name="env">The instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        internal FastForestBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
        }

        private protected override FastForestBinaryModelParameters TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;
            ValidData = context.ValidationSet;
            TestData = context.TestSet;

            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(trainData, nameof(trainData));
                trainData.CheckBinaryLabel();
                trainData.CheckFeatureFloatVector();
                trainData.CheckOptFloatWeight();
                FeatureCount = trainData.Schema.Feature.Value.Type.GetValueCount();
                ConvertData(trainData);

                if (!trainData.Schema.Weight.HasValue && IsDispatchingToOneDalEnabled())
                {
                    if (FastTreeTrainerOptions.FeatureFraction != 1.0)
                    {
                        ch.Warning($"oneDAL decision forest doesn't support 'FeatureFraction'[per tree] != 1.0, changing it from {FastTreeTrainerOptions.FeatureFraction} to 1.0");
                        FastTreeTrainerOptions.FeatureFraction = 1.0;
                    }
                    CursOpt cursorOpt = CursOpt.Label | CursOpt.Features;
                    var cursorFactory = new FloatLabelCursor.Factory(trainData, cursorOpt);
                    TrainCoreOneDal(ch, cursorFactory, FeatureCount);
                    if (FeatureMap != null)
                        TrainedEnsemble.RemapFeatures(FeatureMap);
                }
                else
                {
                    TrainCore(ch);
                }
            }
            // LogitBoost is naturally calibrated to
            // output probabilities when transformed using
            // the logistic function, so if we have trained no
            // calibrator, transform the scores using that.

            // REVIEW: Need a way to signal the outside world that we prefer simple sigmoid?
            return new FastForestBinaryModelParameters(Host, TrainedEnsemble, FeatureCount, InnerOptions);
        }

        internal static class OneDal
        {
            private const string OneDalLibPath = "OneDalNative";

            [DllImport(OneDalLibPath, EntryPoint = "decisionForestClassificationCompute")]
            public static extern unsafe int DecisionForestClassificationCompute(
                void* featuresPtr, void* labelsPtr, long nRows, int nColumns, int nClasses, int numberOfThreads,
                float featureFractionPerSplit, int numberOfTrees, int numberOfLeaves, int minimumExampleCountPerLeaf, int maxBins,
                void* lteChildPtr, void* gtChildPtr, void* splitFeaturePtr, void* featureThresholdPtr, void* leafValuesPtr, void* modelPtr);
        }

        [BestFriend]
        private bool IsDispatchingToOneDalEnabled()
        {
            return OneDalUtils.IsDispatchingEnabled();
        }

        [BestFriend]
        private void TrainCoreOneDal(IChannel ch, FloatLabelCursor.Factory cursorFactory, int featureCount)
        {
            CheckOptions(ch);
            Initialize(ch);

            List<float> featuresList = new List<float>();
            List<float> labelsList = new List<float>();
            int nClasses = 2;
            int numberOfLeaves = FastTreeTrainerOptions.NumberOfLeaves;
            int numberOfTrees = FastTreeTrainerOptions.NumberOfTrees;

            int numberOfThreads = 0;
            if (FastTreeTrainerOptions.NumberOfThreads.HasValue)
                numberOfThreads = FastTreeTrainerOptions.NumberOfThreads.Value;

            long n = OneDalUtils.GetTrainData(ch, cursorFactory, ref featuresList, ref labelsList, featureCount);

            float[] featuresArray = featuresList.ToArray();
            float[] labelsArray = labelsList.ToArray();

            int[] lteChildArray = new int[(numberOfLeaves - 1) * numberOfTrees];
            int[] gtChildArray = new int[(numberOfLeaves - 1) * numberOfTrees];
            int[] splitFeatureArray = new int[(numberOfLeaves - 1) * numberOfTrees];
            float[] featureThresholdArray = new float[(numberOfLeaves - 1) * numberOfTrees];
            float[] leafValuesArray = new float[numberOfLeaves * numberOfTrees];

            int oneDalModelSize = -1;
            int projectedOneDalModelSize = 96 * nClasses * numberOfLeaves * numberOfTrees + 4096 * 16;
            byte[] oneDalModel = new byte[projectedOneDalModelSize];

            unsafe
            {
#pragma warning disable MSML_SingleVariableDeclaration // Have only a single variable present per declaration
                fixed (void* featuresPtr = &featuresArray[0], labelsPtr = &labelsArray[0],
                    lteChildPtr = &lteChildArray[0], gtChildPtr = &gtChildArray[0], splitFeaturePtr = &splitFeatureArray[0],
                    featureThresholdPtr = &featureThresholdArray[0], leafValuesPtr = &leafValuesArray[0], oneDalModelPtr = &oneDalModel[0])
#pragma warning restore MSML_SingleVariableDeclaration // Have only a single variable present per declaration
                {
                    oneDalModelSize = OneDal.DecisionForestClassificationCompute(featuresPtr, labelsPtr, n, featureCount, nClasses,
                        numberOfThreads, (float)FastTreeTrainerOptions.FeatureFractionPerSplit, numberOfTrees,
                        numberOfLeaves, FastTreeTrainerOptions.MinimumExampleCountPerLeaf, FastTreeTrainerOptions.MaximumBinCountPerFeature,
                        lteChildPtr, gtChildPtr, splitFeaturePtr, featureThresholdPtr, leafValuesPtr, oneDalModelPtr
                    );
                }
            }
            TrainedEnsemble = new InternalTreeEnsemble();
            for (int i = 0; i < numberOfTrees; ++i)
            {
                int[] lteChildArrayPerTree = new int[numberOfLeaves - 1];
                int[] gtChildArrayPerTree = new int[numberOfLeaves - 1];
                int[] splitFeatureArrayPerTree = new int[numberOfLeaves - 1];
                float[] featureThresholdArrayPerTree = new float[numberOfLeaves - 1];
                double[] leafValuesArrayPerTree = new double[numberOfLeaves];

                int[][] categoricalSplitFeaturesPerTree = new int[numberOfLeaves - 1][];
                bool[] categoricalSplitPerTree = new bool[numberOfLeaves - 1];
                double[] splitGainPerTree = new double[numberOfLeaves - 1];
                float[] defaultValueForMissingPerTree = new float[numberOfLeaves - 1];

                for (int j = 0; j < numberOfLeaves - 1; ++j)
                {
                    lteChildArrayPerTree[j] = lteChildArray[(numberOfLeaves - 1) * i + j];
                    gtChildArrayPerTree[j] = gtChildArray[(numberOfLeaves - 1) * i + j];
                    splitFeatureArrayPerTree[j] = splitFeatureArray[(numberOfLeaves - 1) * i + j];
                    featureThresholdArrayPerTree[j] = featureThresholdArray[(numberOfLeaves - 1) * i + j];
                    leafValuesArrayPerTree[j] = leafValuesArray[numberOfLeaves * i + j];

                    categoricalSplitFeaturesPerTree[j] = null;
                    categoricalSplitPerTree[j] = false;
                    splitGainPerTree[j] = 0.0;
                    defaultValueForMissingPerTree[j] = 0.0f;
                }
                leafValuesArrayPerTree[numberOfLeaves - 1] = leafValuesArray[numberOfLeaves * i + numberOfLeaves - 1];

                InternalQuantileRegressionTree newTree = new InternalQuantileRegressionTree(splitFeatureArrayPerTree, splitGainPerTree, null,
                    featureThresholdArrayPerTree, defaultValueForMissingPerTree, lteChildArrayPerTree, gtChildArrayPerTree, leafValuesArrayPerTree,
                    categoricalSplitFeaturesPerTree, categoricalSplitPerTree);
                newTree.PopulateThresholds(TrainSet);
                TrainedEnsemble.AddTree(newTree);
            }
        }

        private protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch)
        {
            return new ObjectiveFunctionImpl(TrainSet, _trainSetLabels, FastTreeTrainerOptions);
        }

        private protected override void PrepareLabels(IChannel ch)
        {
            // REVIEW: Historically FastTree has this test as >= 1. TLC however
            // generally uses > 0. Consider changing FastTree to be consistent.
            _trainSetLabels = TrainSet.Ratings.Select(x => x >= 1).ToArray(TrainSet.NumDocs);
        }

        private protected override Test ConstructTestForTrainingData()
        {
            return new BinaryClassificationTest(ConstructScoreTracker(TrainSet), _trainSetLabels, 1);
        }

        private protected override BinaryPredictionTransformer<FastForestBinaryModelParameters> MakeTransformer(FastForestBinaryModelParameters model, DataViewSchema trainSchema)
         => new BinaryPredictionTransformer<FastForestBinaryModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="FastForestBinaryTrainer"/> using both training and validation data, returns
        /// a <see cref="BinaryPredictionTransformer{FastForestClassificationModelParameters}"/>.
        /// </summary>
        public BinaryPredictionTransformer<FastForestBinaryModelParameters> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private sealed class ObjectiveFunctionImpl : RandomForestObjectiveFunction
        {
            private readonly bool[] _labels;

            public ObjectiveFunctionImpl(Dataset trainSet, bool[] trainSetLabels, Options options)
                : base(trainSet, options, options.MaximumOutputMagnitudePerTree)
            {
                _labels = trainSetLabels;
            }

            protected override void GetGradientInOneQuery(int query, int threadIndex)
            {
                int begin = Dataset.Boundaries[query];
                int end = Dataset.Boundaries[query + 1];
                for (int i = begin; i < end; ++i)
                    Gradient[i] = _labels[i] ? 1 : -1;
            }
        }
    }

    internal static partial class FastForest
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastForestBinaryClassifier",
            Desc = FastForestBinaryTrainer.Summary,
            UserName = FastForestBinaryTrainer.UserNameValue,
            ShortName = FastForestBinaryTrainer.ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, FastForestBinaryTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainFastForest");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<FastForestBinaryTrainer.Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new FastForestBinaryTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.RowGroupColumnName),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);

        }
    }
}
