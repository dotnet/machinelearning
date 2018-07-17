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
                loaderSignature: LoaderSignature);
        }

        protected override uint VerNumFeaturesSerialized { get { return 0x00010002; } }

        protected override uint VerDefaultValueSerialized { get { return 0x00010004; } }

        protected override uint VerCategoricalSplitSerialized { get { return 0x00010005; } }

        internal LightGbmRankingPredictor(IHostEnvironment env, FastTree.Internal.Ensemble trainedEnsemble, int featureCount, string innerArgs)
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

        public static LightGbmRankingPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            return new LightGbmRankingPredictor(env, ctx);
        }

        public override PredictionKind PredictionKind { get { return PredictionKind.Ranking; } }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmRankingTrainer : LightGbmTrainerBase<float, LightGbmRankingPredictor>
    {
        public const string UserName = "LightGBM Ranking";
        public const string LoadNameValue = "LightGBMRanking";
        public const string ShortName = "LightGBMRank";

        public LightGbmRankingTrainer(IHostEnvironment env, LightGbmArguments args)
            : base(env, args, PredictionKind.Ranking, "LightGBMRanking")
        {
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

        public override LightGbmRankingPredictor CreatePredictor()
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
