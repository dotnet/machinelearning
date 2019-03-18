// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

[assembly: LoadableClass(LightGbmRankingTrainer.UserName, typeof(LightGbmRankingTrainer), typeof(Options),
    new[] { typeof(SignatureRankerTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    "LightGBM Ranking", LightGbmRankingTrainer.LoadNameValue, LightGbmRankingTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(LightGbmRankingModelParameters), null, typeof(SignatureLoadModel),
    "LightGBM Ranking Executor",
    LightGbmRankingModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.LightGbm
{
    public sealed class LightGbmRankingModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "LightGBMRankerExec";
        internal const string RegistrationName = "LightGBMRankingPredictor";

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
                loaderAssemblyName: typeof(LightGbmRankingModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010002;
        private protected override uint VerDefaultValueSerialized => 0x00010004;
        private protected override uint VerCategoricalSplitSerialized => 0x00010005;
        private protected override PredictionKind PredictionKind => PredictionKind.Ranking;

        internal LightGbmRankingModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmRankingModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static LightGbmRankingModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            return new LightGbmRankingModelParameters(env, ctx);
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a boosted decision tree ranking model using LightGBM.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM_remarks"]/*' />
    public sealed class LightGbmRankingTrainer : LightGbmTrainerBase<float, RankingPredictionTransformer<LightGbmRankingModelParameters>, LightGbmRankingModelParameters>
    {
        internal const string UserName = "LightGBM Ranking";
        internal const string LoadNameValue = "LightGBMRanking";
        internal const string ShortName = "LightGBMRank";

        private protected override PredictionKind PredictionKind => PredictionKind.Ranking;

        internal LightGbmRankingTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmRankingTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupdColumnName">The name of the column containing the group ID. </param>
        /// <param name="weightsColumnName">The name of the optional column containing the initial weights.</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="numberOfIterations">The number of iterations to use.</param>
        internal LightGbmRankingTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupdColumnName = DefaultColumnNames.GroupId,
            string weightsColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Trainers.LightGbm.Options.Defaults.NumberOfIterations)
            : base(env, LoadNameValue, TrainerUtils.MakeR4ScalarColumn(labelColumnName),
                    featureColumnName, weightsColumnName, rowGroupdColumnName, numberOfLeaves,
                    minimumExampleCountPerLeaf, learningRate, numberOfIterations)
        {
            Host.CheckNonEmpty(rowGroupdColumnName, nameof(rowGroupdColumnName));
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            // Check label types.
            var labelCol = data.Schema.Label.Value;
            var labelType = labelCol.Type;
            if (!(labelType is KeyType || labelType == NumberDataViewType.Single))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{labelCol.Name}' is of type '{labelType.RawType}', but must be unsigned int or float.");
            }
            // Check group types.
            ch.CheckParam(data.Schema.Group.HasValue, nameof(data), "Need a group column.");
            var groupCol = data.Schema.Group.Value;
            var groupType = groupCol.Type;
            if (!(groupType == NumberDataViewType.UInt32 || groupType is KeyType))
            {
                throw ch.ExceptParam(nameof(data),
                   $"Group column '{groupCol.Name}' is of type '{groupType.RawType}', but must be unsigned int.");
            }
        }

        private protected override void CheckLabelCompatible(SchemaShape.Column labelCol)
        {
            Contracts.Assert(labelCol.IsValid);

            Action error =
                () => throw Host.ExceptSchemaMismatch(nameof(labelCol), "label", labelCol.Name, "float or unsigned int", labelCol.GetTypeString());

            if (labelCol.Kind != SchemaShape.Column.VectorKind.Scalar)
                error();
            if (!labelCol.IsKey && labelCol.ItemType != NumberDataViewType.Single)
                error();
        }

        private protected override LightGbmRankingModelParameters CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            return new LightGbmRankingModelParameters(Host, TrainedEnsemble, FeatureCount, innerArgs);
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
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

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
           {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override RankingPredictionTransformer<LightGbmRankingModelParameters> MakeTransformer(LightGbmRankingModelParameters model, DataViewSchema trainSchema)
         => new RankingPredictionTransformer<LightGbmRankingModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="LightGbmRankingTrainer"/> using both training and validation data, returns
        /// a <see cref="RankingPredictionTransformer{LightGbmRankingModelParameters}"/>.
        /// </summary>
        public RankingPredictionTransformer<LightGbmRankingModelParameters> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// The entry point for the LightGbmRankingTrainer.
    /// </summary>
    internal static partial class LightGbm
    {
        [TlcModule.EntryPoint(Name = "Trainers.LightGbmRanker",
            Desc = "Train a LightGBM ranking model.",
            UserName = LightGbmRankingTrainer.UserName,
            ShortName = LightGbmRankingTrainer.ShortName)]
        public static CommonOutputs.RankingOutput TrainRanking(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.RankingOutput>(host, input,
                () => new LightGbmRankingTrainer(host, input),
                getLabel: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                getWeight: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName),
                getGroup: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.RowGroupColumnName));
        }
    }
}
