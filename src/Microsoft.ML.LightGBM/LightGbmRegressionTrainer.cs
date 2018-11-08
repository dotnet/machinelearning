// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.FastTree.Internal;
using System;

[assembly: LoadableClass(LightGbmRegressorTrainer.Summary, typeof(LightGbmRegressorTrainer), typeof(LightGbmArguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmRegressorTrainer.UserNameValue, LightGbmRegressorTrainer.LoadNameValue, LightGbmRegressorTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(LightGbmRegressionPredictor), null, typeof(SignatureLoadModel),
    "LightGBM Regression Executor",
    LightGbmRegressionPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.LightGBM
{
    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmRegressionPredictor : FastTreePredictionWrapper
    {
        public const string LoaderSignature = "LightGBMRegressionExec";
        public const string RegistrationName = "LightGBMRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBSIREG",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LightGbmRegressionPredictor).Assembly.FullName);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010002;
        protected override uint VerDefaultValueSerialized => 0x00010004;
        protected override uint VerCategoricalSplitSerialized => 0x00010005;
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        internal LightGbmRegressionPredictor(IHostEnvironment env, TreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
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

    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmRegressorTrainer : LightGbmTrainerBase<float, RegressionPredictionTransformer<LightGbmRegressionPredictor>, LightGbmRegressionPredictor>
    {
        internal const string Summary = "LightGBM Regression";
        internal const string LoadNameValue = "LightGBMRegression";
        internal const string ShortName = "LightGBMR";
        internal const string UserNameValue = "LightGBM Regressor";

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmRegressorTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public LightGbmRegressorTrainer(IHostEnvironment env, string labelColumn, string featureColumn,
            string weightColumn = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null)
            : base(env, LoadNameValue, TrainerUtils.MakeR4ScalarLabel(labelColumn), featureColumn, weightColumn, null, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings)
        {
        }

        internal LightGbmRegressorTrainer(IHostEnvironment env, LightGbmArguments args)
             : base(env, LoadNameValue, args, TrainerUtils.MakeR4ScalarLabel(args.LabelColumn))
        {
        }

        private protected override LightGbmRegressionPredictor CreatePredictor()
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

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override RegressionPredictionTransformer<LightGbmRegressionPredictor> MakeTransformer(LightGbmRegressionPredictor model, Schema trainSchema)
            => new RegressionPredictionTransformer<LightGbmRegressionPredictor>(Host, model, trainSchema, FeatureColumn.Name);
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
            XmlInclude = new[] { @"<include file='../Microsoft.ML.LightGBM/doc.xml' path='doc/members/member[@name=""LightGBM""]/*' />",
                                 @"<include file='../Microsoft.ML.LightGBM/doc.xml' path='doc/members/example[@name=""LightGbmRegressor""]/*' />"})]
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
