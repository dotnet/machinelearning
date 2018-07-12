// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(LightGbmRegressorTrainer.Summary, typeof(LightGbmRegressorTrainer), typeof(LightGbmArguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmRegressorTrainer.UserNameValue, LightGbmRegressorTrainer.LoadNameValue, LightGbmRegressorTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(LightGbmRegressionPredictor), null, typeof(SignatureLoadModel),
    "LightGBM Regression Executor",
    LightGbmRegressionPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.LightGBM
{
    /// <include file='./doc.xml' path='docs/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmRegressionPredictor : FastTreePredictionWrapper
    {
        public const string LoaderSignature = "LightGBMRegressionExec";
        public const string RegistrationName = "LightGBMRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW tfinley(guoke): can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBSIREG",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010002;
        protected override uint VerDefaultValueSerialized => 0x00010004;
        protected override uint VerCategoricalSplitSerialized => 0x00010005;
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        internal LightGbmRegressionPredictor(IHostEnvironment env, FastTree.Internal.Ensemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmRegressionPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public static LightGbmRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LightGbmRegressionPredictor(env, ctx);
        }
    }

    public sealed class LightGbmRegressorTrainer : LightGbmTrainerBase<float, LightGbmRegressionPredictor>
    {
        public const string Summary = "LightGBM Regression";
        public const string LoadNameValue = "LightGBMRegression";
        public const string ShortName = "LightGBMR";
        public const string UserNameValue = "LightGBM Regressor";

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public LightGbmRegressorTrainer(IHostEnvironment env, LightGbmArguments args)
            : base(env, args, LoadNameValue)
        {
        }

        protected internal override LightGbmRegressionPredictor CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null,
                "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            return new LightGbmRegressionPredictor(Host, TrainedEnsemble, FeatureCount, innerArgs);
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
            Options["objective"] = "regression";
            // Add default metric.
            if (!Options.ContainsKey("metric"))
                Options["metric"] = "l2";
        }
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    public static partial class LightGbm
    {
        [TlcModule.EntryPoint(Name = "Trainers.LightGbmRegressor", 
            Desc = LightGbmRegressorTrainer.Summary, 
            UserName = LightGbmRegressorTrainer.UserNameValue, 
            ShortName = LightGbmRegressorTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.LightGBM/doc.xml' path='docs/members/member[@name=""LightGBM""]/*' />" })]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, LightGbmArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<LightGbmArguments, CommonOutputs.RegressionOutput>(host, input,
                () => new LightGbmRegressorTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
