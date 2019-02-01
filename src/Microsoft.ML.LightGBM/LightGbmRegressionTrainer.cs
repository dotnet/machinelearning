// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.LightGBM;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.FastTree.Internal;
using Microsoft.ML.Training;

[assembly: LoadableClass(LightGbmRegressorTrainer.Summary, typeof(LightGbmRegressorTrainer), typeof(Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmRegressorTrainer.UserNameValue, LightGbmRegressorTrainer.LoadNameValue, LightGbmRegressorTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(LightGbmRegressionModelParameters), null, typeof(SignatureLoadModel),
    "LightGBM Regression Executor",
    LightGbmRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.LightGBM
{
    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmRegressionModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "LightGBMRegressionExec";
        internal const string RegistrationName = "LightGBMRegressionPredictor";

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
                loaderAssemblyName: typeof(LightGbmRegressionModelParameters).Assembly.FullName);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010002;
        protected override uint VerDefaultValueSerialized => 0x00010004;
        protected override uint VerCategoricalSplitSerialized => 0x00010005;
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        internal LightGbmRegressionModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static LightGbmRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LightGbmRegressionModelParameters(env, ctx);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmRegressorTrainer : LightGbmTrainerBase<float, RegressionPredictionTransformer<LightGbmRegressionModelParameters>, LightGbmRegressionModelParameters>
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
        /// <param name="weights">The name for the column containing the initial weight.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        internal LightGbmRegressorTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGBM.Options.Defaults.NumBoostRound)
            : base(env, LoadNameValue, TrainerUtils.MakeR4ScalarColumn(labelColumn), featureColumn, weights, null, numLeaves, minDataPerLeaf, learningRate, numBoostRound)
        {
        }

        internal LightGbmRegressorTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumn))
        {
        }

        private protected override LightGbmRegressionModelParameters CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null,
                "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            return new LightGbmRegressionModelParameters(Host, TrainedEnsemble, FeatureCount, innerArgs);
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BoolType || labelType is KeyType || labelType == NumberType.R4))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType}', but must be key, boolean or R4.");
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
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

        protected override RegressionPredictionTransformer<LightGbmRegressionModelParameters> MakeTransformer(LightGbmRegressionModelParameters model, Schema trainSchema)
            => new RegressionPredictionTransformer<LightGbmRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        public RegressionPredictionTransformer<LightGbmRegressionModelParameters> Train(IDataView trainData, IDataView validationData = null)
            => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    internal static partial class LightGbm
    {
        [TlcModule.EntryPoint(Name = "Trainers.LightGbmRegressor",
            Desc = LightGbmRegressorTrainer.Summary,
            UserName = LightGbmRegressorTrainer.UserNameValue,
            ShortName = LightGbmRegressorTrainer.ShortName)]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Options, CommonOutputs.RegressionOutput>(host, input,
                () => new LightGbmRegressorTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
