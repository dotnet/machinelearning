// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;

[assembly: LoadableClass(LightGbmBinaryTrainer.Summary, typeof(LightGbmBinaryTrainer), typeof(LightGbmArguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmBinaryTrainer.UserName, LightGbmBinaryTrainer.LoadNameValue, LightGbmBinaryTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(LightGbmBinaryPredictor), null, typeof(SignatureLoadModel),
    "LightGBM Binary Executor",
    LightGbmBinaryPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(LightGbm), null, typeof(SignatureEntryPointModule), "LightGBM")]

namespace Microsoft.ML.Runtime.LightGBM
{
    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmBinaryPredictor : FastTreePredictionWrapper
    {
        internal const string LoaderSignature = "LightGBMBinaryExec";
        internal const string RegistrationName = "LightGBMBinaryPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBBINCL",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                //verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LightGbmBinaryPredictor).Assembly.FullName);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010002;
        protected override uint VerDefaultValueSerialized => 0x00010004;
        protected override uint VerCategoricalSplitSerialized => 0x00010005;
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        internal LightGbmBinaryPredictor(IHostEnvironment env, FastTree.Internal.Ensemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmBinaryPredictor(IHostEnvironment env, ModelLoadContext ctx)
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
            var predictor = new LightGbmBinaryPredictor(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new CalibratedPredictor(env, predictor, calibrator);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmBinaryTrainer : LightGbmTrainerBase<float, BinaryPredictionTransformer<IPredictorWithFeatureWeights<float>>, IPredictorWithFeatureWeights<float>>
    {
        internal const string UserName = "LightGBM Binary Classifier";
        internal const string LoadNameValue = "LightGBMBinary";
        internal const string ShortName = "LightGBM";
        internal const string Summary = "Train a LightGBM binary classification model.";

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        internal LightGbmBinaryTrainer(IHostEnvironment env, LightGbmArguments args)
             : base(env, LoadNameValue, args, TrainerUtils.MakeBoolScalarLabel(args.LabelColumn))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="groupIdColumn">The name for the column containing the group ID. </param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public LightGbmBinaryTrainer(IHostEnvironment env, string labelColumn, string featureColumn,
            string groupIdColumn = null, string weightColumn = null, Action<LightGbmArguments> advancedSettings = null)
            : base(env, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(labelColumn), featureColumn, weightColumn, groupIdColumn, advancedSettings)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
        }

        private protected override IPredictorWithFeatureWeights<float> CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            var pred = new LightGbmBinaryPredictor(Host, TrainedEnsemble, FeatureCount, innerArgs);
            var cali = new PlattCalibrator(Host, -0.5, 0);
            return new FeatureWeightsCalibratedPredictor(Host, pred, cali);
        }

        protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Type;
            if (!(labelType.IsBool || labelType.IsKey || labelType == NumberType.R4))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Name}' is of type '{labelType}', but must be key, boolean or R4.");
            }
        }

        protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        {
            Options["objective"] = "binary";
            // Add default metric.
            if (!Options.ContainsKey("metric"))
                Options["metric"] = "binary_logloss";
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema) {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override BinaryPredictionTransformer<IPredictorWithFeatureWeights<float>> MakeTransformer(IPredictorWithFeatureWeights<float> model, ISchema trainSchema)
         => new BinaryPredictionTransformer<IPredictorWithFeatureWeights<float>>(Host, model, trainSchema, FeatureColumn.Name);
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    public static partial class LightGbm
    {
        [TlcModule.EntryPoint(
            Name = "Trainers.LightGbmBinaryClassifier",
            Desc = LightGbmBinaryTrainer.Summary,
            UserName = LightGbmBinaryTrainer.UserName,
            ShortName = LightGbmBinaryTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.LightGBM/doc.xml' path='doc/members/member[@name=""LightGBM""]/*' />",
                                 @"<include file='../Microsoft.ML.LightGBM/doc.xml' path='doc/members/example[@name=""LightGbmBinaryClassifier""]/*' />"})]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, LightGbmArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<LightGbmArguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LightGbmBinaryTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
