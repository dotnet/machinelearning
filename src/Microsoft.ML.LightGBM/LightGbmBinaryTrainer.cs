// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(LightGbmBinaryTrainer.Summary, typeof(LightGbmBinaryTrainer), typeof(LightGbmArguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmBinaryTrainer.UserName, LightGbmBinaryTrainer.LoadNameValue, LightGbmBinaryTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(LightGbmBinaryPredictor), null, typeof(SignatureLoadModel),
    "LightGBM Binary Executor",
    LightGbmBinaryPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(LightGbm), null, typeof(SignatureEntryPointModule), "LightGBM")]

namespace Microsoft.ML.Runtime.LightGBM
{
    public sealed class LightGbmBinaryPredictor : FastTreePredictionWrapper
    {
        public const string LoaderSignature = "LightGBMBinaryExec";
        public const string RegistrationName = "LightGBMBinaryPredictor";
        
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
                loaderSignature: LoaderSignature);
        }

        protected override uint VerNumFeaturesSerialized { get { return 0x00010002; } }

        protected override uint VerDefaultValueSerialized { get { return 0x00010004; } }

        protected override uint VerCategoricalSplitSerialized { get { return 0x00010005; } }

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

        public override PredictionKind PredictionKind { get { return PredictionKind.BinaryClassification; } }
    }

    /// <include file='../../docs/code/xmlIncludes/Learners.xml' path='docs/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmBinaryTrainer : LightGbmTrainerBase<float, IPredictorWithFeatureWeights<float>>
    {
        internal const string UserName = "LightGBM Binary Classifier";
        internal const string LoadNameValue = "LightGBMBinary";
        internal const string ShortName = "LightGBM";
        internal const string Summary = "Train a LightGBM binary classification model.";

        public LightGbmBinaryTrainer(IHostEnvironment env, LightGbmArguments args)
            : base(env, args, PredictionKind.BinaryClassification, "LGBBINCL")
        {
        }

        public override IPredictorWithFeatureWeights<float> CreatePredictor()
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
            XmlInclude = new[] { @"<include file='../../docs/code/xmlIncludes/Learners.xml' path='docs/members/member[@name=""LightGBM""]/*' />" })]
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
