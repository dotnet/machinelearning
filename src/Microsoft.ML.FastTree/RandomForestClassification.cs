// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;
using System.Linq;

[assembly: LoadableClass(FastForestClassification.Summary, typeof(FastForestClassification), typeof(FastForestClassification.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastForestClassification.UserNameValue,
    FastForestClassification.LoadNameValue,
    "FastForest",
    FastForestClassification.ShortName,
    "ffc")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(FastForestClassificationPredictor), null, typeof(SignatureLoadModel),
    "FastForest Binary Executor",
    FastForestClassificationPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(FastForest), null, typeof(SignatureEntryPointModule), "FastForest")]

namespace Microsoft.ML.Runtime.FastTree
{
    public abstract class FastForestArgumentsBase : TreeArgs
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of labels to be sampled from each leaf to make the distribtuion", ShortName = "qsc")]
        public int QuantileSampleCount = 100;

        public FastForestArgumentsBase()
        {
            FeatureFraction = 0.7;
            BaggingSize = 1;
            SplitFraction = 0.7;
        }
    }

    public sealed class FastForestClassificationPredictor :
        FastTreePredictionWrapper
    {
        public const string LoaderSignature = "FastForestBinaryExec";
        public const string RegistrationName = "FastForestClassificationPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FFORE BC",
                // verWrittenCur: 0x00010001, Initial
                // verWrittenCur: 0x00010002, // InstanceWeights are part of QuantileRegression Tree to support weighted intances
                // verWrittenCur: 0x00010003, // _numFeatures serialized
                // verWrittenCur: 0x00010004, // Ini content out of predictor
                // verWrittenCur: 0x00010005, // Add _defaultValueForMissing
                verWrittenCur: 0x00010006, // Categorical splits.
                verReadableCur: 0x00010005,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010003;

        protected override uint VerDefaultValueSerialized => 0x00010005;

        protected override uint VerCategoricalSplitSerialized => 0x00010006;

        /// <summary>
        /// The type of prediction for this trainer.
        /// </summary>
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public FastForestClassificationPredictor(IHostEnvironment env, Ensemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {  }

        private FastForestClassificationPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new FastForestClassificationPredictor(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new SchemaBindableCalibratedPredictor(env, predictor, calibrator);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="FastForest"]/*' />
    public sealed partial class FastForestClassification :
        RandomForestTrainerBase<FastForestClassification.Arguments, BinaryPredictionTransformer<IPredictorWithFeatureWeights<float>>, IPredictorWithFeatureWeights<float>>
    {
        public sealed class Arguments : FastForestArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Upper bound on absolute value of single tree output", ShortName = "mo")]
            public Double MaxTreeOutput = 100;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public int MaxCalibrationExamples = 1000000;
        }

        internal const string LoadNameValue = "FastForestClassification";
        internal const string UserNameValue = "Fast Forest Classification";
        internal const string Summary = "Uses a random forest learner to perform binary classification.";
        internal const string ShortName = "ff";

        private bool[] _trainSetLabels;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        private protected override bool NeedCalibration => true;

        /// <summary>
        /// Initializes a new instance of <see cref="FastForestClassification"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="groupIdColumn">The name for the column containing the group ID.</param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public FastForestClassification(IHostEnvironment env, string labelColumn, string featureColumn,
            string groupIdColumn = null, string weightColumn = null, Action<Arguments> advancedSettings = null)
            : base(env, TrainerUtils.MakeBoolScalarLabel(labelColumn), featureColumn, weightColumn, groupIdColumn, advancedSettings: advancedSettings)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastForestClassification"/> by using the legacy <see cref="Arguments"/> class.
        /// </summary>
        public FastForestClassification(IHostEnvironment env, Arguments args)
            : base(env, args, TrainerUtils.MakeBoolScalarLabel(args.LabelColumn))
        {
        }

        protected override IPredictorWithFeatureWeights<float> TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;
            ValidData = context.ValidationSet;

            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(trainData, nameof(trainData));
                trainData.CheckBinaryLabel();
                trainData.CheckFeatureFloatVector();
                trainData.CheckOptFloatWeight();
                FeatureCount = trainData.Schema.Feature.Type.ValueCount;
                ConvertData(trainData);
                TrainCore(ch);
                ch.Done();
            }
            // LogitBoost is naturally calibrated to
            // output probabilities when transformed using
            // the logistic function, so if we have trained no
            // calibrator, transform the scores using that.

            // REVIEW: Need a way to signal the outside world that we prefer simple sigmoid?
            return new FastForestClassificationPredictor(Host, TrainedEnsemble, FeatureCount, InnerArgs);
        }

        protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch)
        {
            return new ObjectiveFunctionImpl(TrainSet, _trainSetLabels, Args);
        }

        protected override void PrepareLabels(IChannel ch)
        {
            // REVIEW: Historically FastTree has this test as >= 1. TLC however
            // generally uses > 0. Consider changing FastTree to be consistent.
            _trainSetLabels = TrainSet.Ratings.Select(x => x >= 1).ToArray(TrainSet.NumDocs);
        }

        protected override Test ConstructTestForTrainingData()
        {
            return new BinaryClassificationTest(ConstructScoreTracker(TrainSet), _trainSetLabels, 1);
        }

        protected override BinaryPredictionTransformer<IPredictorWithFeatureWeights<float>> MakeTransformer(IPredictorWithFeatureWeights<float> model, ISchema trainSchema)
         => new BinaryPredictionTransformer<IPredictorWithFeatureWeights<float>>(Host, model, trainSchema, FeatureColumn.Name);

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        private sealed class ObjectiveFunctionImpl : RandomForestObjectiveFunction
        {
            private readonly bool[] _labels;

            public ObjectiveFunctionImpl(Dataset trainSet, bool[] trainSetLabels, Arguments args)
                : base(trainSet, args, args.MaxTreeOutput)
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

    public static partial class FastForest
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastForestBinaryClassifier",
            Desc = FastForestClassification.Summary,
            UserName = FastForestClassification.UserNameValue,
            ShortName = FastForestClassification.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/member[@name=""FastForest""]/*' />",
                                 @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/example[@name=""FastForestBinaryClassifier""]/*' />"})]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, FastForestClassification.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainFastForest");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<FastForestClassification.Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new FastForestClassification(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.GroupIdColumn),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);

        }
    }
}
