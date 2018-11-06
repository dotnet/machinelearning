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

[assembly: LoadableClass(LightGbmRankingTrainer.UserName, typeof(LightGbmRankingTrainer), typeof(LightGbmArguments),
    new[] { typeof(SignatureRankerTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    "LightGBM Ranking", LightGbmRankingTrainer.LoadNameValue, LightGbmRankingTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(LightGbmRankingPredictor), null, typeof(SignatureLoadModel),
    "LightGBM Ranking Executor",
    LightGbmRankingPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.LightGBM
{

    public sealed class LightGbmRankingPredictor : FastTreePredictionWrapper
    {
        public const string LoaderSignature = "LightGBMRankerExec";
        public const string RegistrationName = "LightGBMRankingPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW tfinley(guoke): can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBMRANK",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LightGbmRankingPredictor).Assembly.FullName);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010002;
        protected override uint VerDefaultValueSerialized => 0x00010004;
        protected override uint VerCategoricalSplitSerialized => 0x00010005;
        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        internal LightGbmRankingPredictor(IHostEnvironment env, Ensemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmRankingPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static LightGbmRankingPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            return new LightGbmRankingPredictor(env, ctx);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmRankingTrainer : LightGbmTrainerBase<float, RankingPredictionTransformer<LightGbmRankingPredictor>, LightGbmRankingPredictor>
    {
        public const string UserName = "LightGBM Ranking";
        public const string LoadNameValue = "LightGBMRanking";
        public const string ShortName = "LightGBMRank";

        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        internal LightGbmRankingTrainer(IHostEnvironment env, LightGbmArguments args)
             : base(env, LoadNameValue, args, TrainerUtils.MakeR4ScalarLabel(args.LabelColumn))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmRankingTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="groupId">The name of the column containing the group ID. </param>
        /// <param name="weights">The name of the optional column containing the initial weights.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public LightGbmRankingTrainer(IHostEnvironment env,
            string label,
            string features,
            string groupId,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null)
            : base(env, LoadNameValue, TrainerUtils.MakeR4ScalarLabel(label), features, weights, groupId, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings)
        {
            Host.CheckNonEmpty(groupId, nameof(groupId));
        }

        protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            // Check label types.
            var labelType = data.Schema.Label.Type;
            if (!(labelType.IsKey || labelType == NumberType.R4))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Name}' is of type '{labelType}', but must be key or R4.");
            }
            // Check group types.
            var groupType = data.Schema.Group.Type;
            if (!(groupType == NumberType.U4 || groupType.IsKey))
            {
                throw ch.ExceptParam(nameof(data),
                   $"Group column '{data.Schema.Group.Name}' is of type '{groupType}', but must be U4 or a Key.");
            }
        }

        private protected override LightGbmRankingPredictor CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            return new LightGbmRankingPredictor(Host, TrainedEnsemble, FeatureCount, innerArgs);
        }

        protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        {
            Host.AssertValue(ch);
            Options["objective"] = "lambdarank";
            ch.CheckValue(groups, nameof(groups));
            // Add default metric.
            if (!Options.ContainsKey("metric"))
                Options["metric"] = "ndcg";
            // Only output one ndcg score.
            Options["eval_at"] = "5";
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
           {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override RankingPredictionTransformer<LightGbmRankingPredictor> MakeTransformer(LightGbmRankingPredictor model, Schema trainSchema)
         => new RankingPredictionTransformer<LightGbmRankingPredictor>(Host, model, trainSchema, FeatureColumn.Name);
    }

    /// <summary>
    /// The entry point for the LightGbmRankingTrainer.
    /// </summary>
    public static partial class LightGbm
    {
        [TlcModule.EntryPoint(Name = "Trainers.LightGbmRanker",
            Desc = "Train a LightGBM ranking model.",
            UserName = LightGbmRankingTrainer.UserName,
            ShortName = LightGbmRankingTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.LightGBM/doc.xml' path='doc/members/member[@name=""LightGBM""]/*' />",
                                 @"<include file='../Microsoft.ML.LightGBM/doc.xml' path='doc/members/example[@name=""LightGbmRanker""]/*' />"})]
        public static CommonOutputs.RankingOutput TrainRanking(IHostEnvironment env, LightGbmArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<LightGbmArguments, CommonOutputs.RankingOutput>(host, input,
                () => new LightGbmRankingTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn),
                getGroup: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.GroupIdColumn));
        }
    }
}
