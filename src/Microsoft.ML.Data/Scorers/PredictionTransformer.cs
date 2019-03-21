// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(typeof(BinaryPredictionTransformer<IPredictorProducing<float>>), typeof(BinaryPredictionTransformer), null, typeof(SignatureLoadModel),
    "", BinaryPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(MulticlassPredictionTransformer<IPredictorProducing<VBuffer<float>>>), typeof(MulticlassPredictionTransformer), null, typeof(SignatureLoadModel),
    "", MulticlassPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(RegressionPredictionTransformer<IPredictorProducing<float>>), typeof(RegressionPredictionTransformer), null, typeof(SignatureLoadModel),
    "", RegressionPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(RankingPredictionTransformer<IPredictorProducing<float>>), typeof(RankingPredictionTransformer), null, typeof(SignatureLoadModel),
    "", RankingPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(AnomalyPredictionTransformer<IPredictorProducing<float>>), typeof(AnomalyPredictionTransformer), null, typeof(SignatureLoadModel),
    "", AnomalyPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ClusteringPredictionTransformer<IPredictorProducing<VBuffer<float>>>), typeof(ClusteringPredictionTransformer), null, typeof(SignatureLoadModel),
    "", ClusteringPredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.Data
{

    /// <summary>
    /// Base class for transformers with no feature column, or more than one feature columns.
    /// </summary>
    /// <typeparam name="TModel">The type of the model parameters used by this prediction transformer.</typeparam>
    public abstract class PredictionTransformerBase<TModel> : IPredictionTransformer<TModel>
        where TModel : class
    {
        /// <summary>
        /// The model.
        /// </summary>
        public TModel Model { get; }

        private protected IPredictor ModelAsPredictor => (IPredictor)Model;

        [BestFriend]
        private protected const string DirModel = "Model";
        [BestFriend]
        private protected const string DirTransSchema = "TrainSchema";
        [BestFriend]
        private protected readonly IHost Host;
        [BestFriend]
        private protected ISchemaBindableMapper BindableMapper;
        [BestFriend]
        private protected DataViewSchema TrainSchema;

        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => true;

        /// <summary>
        /// This class is more or less a thin wrapper over the <see cref="IDataScorerTransform"/> implementing
        /// <see cref="RowToRowScorerBase"/>, which publicly is a deprecated concept as far as the public API is
        /// concerned. Nonetheless, until we move all internal infrastructure to be truely transform based, we
        /// retain this as a wrapper. Even though it is mutable, subclasses of this should set this only in
        /// their constructor.
        /// </summary>
        [BestFriend]
        private protected RowToRowScorerBase Scorer { get; set; }

        [BestFriend]
        private protected PredictionTransformerBase(IHost host, TModel model, DataViewSchema trainSchema)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;

            Host.CheckValue(model, nameof(model));
            Host.CheckParam(model is IPredictor, nameof(model));
            Model = model;

            Host.CheckValue(trainSchema, nameof(trainSchema));
            TrainSchema = trainSchema;
        }

        [BestFriend]
        private protected PredictionTransformerBase(IHost host, ModelLoadContext ctx)
        {
            Host = host;

            // *** Binary format ***
            // model: prediction model.
            // stream: empty data view that contains train schema.
            // id of string: feature column.

            ctx.LoadModel<TModel, SignatureLoadModel>(host, out TModel model, DirModel);
            Model = model;

            // Clone the stream with the schema into memory.
            var ms = new MemoryStream();
            ctx.TryLoadBinaryStream(DirTransSchema, reader =>
            {
                reader.BaseStream.CopyTo(ms);
            });

            ms.Position = 0;
            var loader = new BinaryLoader(host, new BinaryLoader.Arguments(), ms);
            TrainSchema = loader.Schema;
        }

        /// <summary>
        /// Gets the output schema resulting from the <see cref="Transform(IDataView)"/>
        /// </summary>
        /// <param name="inputSchema">The <see cref="DataViewSchema"/> of the input data.</param>
        /// <returns>The resulting <see cref="DataViewSchema"/>.</returns>
        public abstract DataViewSchema GetOutputSchema(DataViewSchema inputSchema);

        /// <summary>
        /// Transforms the input data.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <returns>The transformed <see cref="IDataView"/></returns>
        public IDataView Transform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return Scorer.ApplyToData(Host, input);
        }

        /// <summary>
        /// Gets a IRowToRowMapper instance.
        /// </summary>
        /// <param name="inputSchema"></param>
        /// <returns></returns>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return (IRowToRowMapper)Scorer.ApplyToData(Host, new EmptyDataView(Host, inputSchema));
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected abstract void SaveModel(ModelSaveContext ctx);

        [BestFriend]
        private protected void SaveModelCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // <base info>
            // stream: empty data view that contains train schema.

            ctx.SaveModel(Model, DirModel);
            ctx.SaveBinaryStream(DirTransSchema, writer =>
            {
                using (var ch = Host.Start("Saving train schema"))
                {
                    var saver = new BinarySaver(Host, new BinarySaver.Arguments { Silent = true });
                    DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(Host, TrainSchema), writer.BaseStream);
                }
            });
        }
    }

    /// <summary>
    /// The base class for all the transformers implementing the <see cref="ISingleFeaturePredictionTransformer{TModel}"/>.
    /// Those are all the transformers that work with one feature column.
    /// </summary>
    /// <typeparam name="TModel">The model used to transform the data.</typeparam>
    public abstract class SingleFeaturePredictionTransformerBase<TModel> : PredictionTransformerBase<TModel>, ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        /// <summary>
        /// The name of the feature column used by the prediction transformer.
        /// </summary>
        public string FeatureColumnName { get; }

        /// <summary>
        /// The type of the prediction transformer
        /// </summary>
        public DataViewType FeatureColumnType { get; }

        /// <summary>
        /// Initializes a new reference of <see cref="SingleFeaturePredictionTransformerBase{TModel}"/>.
        /// </summary>
        /// <param name="host">The local instance of <see cref="IHost"/>.</param>
        /// <param name="model">The model used for scoring.</param>
        /// <param name="trainSchema">The schema of the training data.</param>
        /// <param name="featureColumn">The feature column name.</param>
        private protected SingleFeaturePredictionTransformerBase(IHost host, TModel model, DataViewSchema trainSchema, string featureColumn)
            : base(host, model, trainSchema)
        {
            FeatureColumnName = featureColumn;
            if (featureColumn == null)
                FeatureColumnType = null;
            else if (!trainSchema.TryGetColumnIndex(featureColumn, out int col))
                throw Host.ExceptSchemaMismatch(nameof(featureColumn), "feature", featureColumn);
            else
                FeatureColumnType = trainSchema[col].Type;

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, ModelAsPredictor);
        }

        private protected SingleFeaturePredictionTransformerBase(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            FeatureColumnName = ctx.LoadStringOrNull();

            if (FeatureColumnName == null)
                FeatureColumnType = null;
            else if (!TrainSchema.TryGetColumnIndex(FeatureColumnName, out int col))
                throw Host.ExceptSchemaMismatch(nameof(FeatureColumnName), "feature", FeatureColumnName);
            else
                FeatureColumnType = TrainSchema[col].Type;

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, ModelAsPredictor);
        }

        /// <summary>
        ///  Schema propagation for this prediction transformer.
        /// </summary>
        /// <param name="inputSchema">The input schema to attempt to map.</param>
        /// <returns>The output schema of the data, given an input schema like <paramref name="inputSchema"/>.</returns>
        public sealed override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (FeatureColumnName != null)
            {
                if (!inputSchema.TryGetColumnIndex(FeatureColumnName, out int col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", FeatureColumnName);
                if (!inputSchema[col].Type.Equals(FeatureColumnType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", FeatureColumnName, FeatureColumnType.ToString(), inputSchema[col].Type.ToString());
            }

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        private protected sealed override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        private protected virtual void SaveCore(ModelSaveContext ctx)
        {
            SaveModelCore(ctx);
            ctx.SaveStringOrNull(FeatureColumnName);
        }

        private protected GenericScorer GetGenericScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumnName);
            return new GenericScorer(Host, new GenericScorer.Arguments(), new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on anomaly detection tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class AnomalyPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel>
        where TModel : class
    {
        internal readonly string ThresholdColumn;
        internal readonly float Threshold;

        [BestFriend]
        internal AnomalyPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn,
            float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(AnomalyPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            Threshold = threshold;
            ThresholdColumn = thresholdColumn;

            SetScorer();
        }

        internal AnomalyPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(AnomalyPredictionTransformer<TModel>)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // float: scorer threshold
            // id of string: scorer threshold column

            Threshold = ctx.Reader.ReadSingle();
            ThresholdColumn = ctx.LoadString();
            SetScorer();
        }

        private void SetScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumnName);
            var args = new BinaryClassifierScorer.Arguments { Threshold = Threshold, ThresholdColumn = ThresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // float: scorer threshold
            // id of string: scorer threshold column
            base.SaveCore(ctx);

            ctx.Writer.Write(Threshold);
            ctx.SaveString(ThresholdColumn);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ANOMPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: AnomalyPredictionTransformer.LoaderSignature,
                loaderAssemblyName: typeof(AnomalyPredictionTransformer<>).Assembly.FullName);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on binary classification tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class BinaryPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel>
        where TModel : class
    {
        internal readonly string ThresholdColumn;
        internal readonly float Threshold;

        [BestFriend]
        internal BinaryPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn,
            float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            Threshold = threshold;
            ThresholdColumn = thresholdColumn;

            SetScorer();
        }

        internal BinaryPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<TModel>)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // float: scorer threshold
            // id of string: scorer threshold column

            Threshold = ctx.Reader.ReadSingle();
            ThresholdColumn = ctx.LoadString();
            SetScorer();
        }

        private void SetScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumnName);
            var args = new BinaryClassifierScorer.Arguments { Threshold = Threshold, ThresholdColumn = ThresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // float: scorer threshold
            // id of string: scorer threshold column
            base.SaveCore(ctx);

            ctx.Writer.Write(Threshold);
            ctx.SaveString(ThresholdColumn);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BIN PRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: BinaryPredictionTransformer.LoaderSignature,
                loaderAssemblyName: typeof(BinaryPredictionTransformer<>).Assembly.FullName);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on multi-class classification tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class MulticlassPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel>
        where TModel : class
    {
        private readonly string _trainLabelColumn;

        [BestFriend]
        internal MulticlassPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn, string labelColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckValueOrNull(labelColumn);

            _trainLabelColumn = labelColumn;
            SetScorer();
        }

        internal MulticlassPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<TModel>)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // id of string: train label column

            _trainLabelColumn = ctx.LoadStringOrNull();
            SetScorer();
        }

        private void SetScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, _trainLabelColumn, FeatureColumnName);
            var args = new MulticlassClassificationScorer.Arguments();
            Scorer = new MulticlassClassificationScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // id of string: train label column
            base.SaveCore(ctx);

            ctx.SaveStringOrNull(_trainLabelColumn);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MC  PRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: MulticlassPredictionTransformer.LoaderSignature,
                loaderAssemblyName: typeof(MulticlassPredictionTransformer<>).Assembly.FullName);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on regression tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class RegressionPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel>
        where TModel : class
    {
        [BestFriend]
        internal RegressionPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Scorer = GetGenericScorer();
        }

        internal RegressionPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), ctx)
        {
            Scorer = GetGenericScorer();
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            base.SaveCore(ctx);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "REG PRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: RegressionPredictionTransformer.LoaderSignature,
                loaderAssemblyName: typeof(RegressionPredictionTransformer<>).Assembly.FullName);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on ranking tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class RankingPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel>
    where TModel : class
    {
        [BestFriend]
        internal RankingPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RankingPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Scorer = GetGenericScorer();
        }

        internal RankingPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RankingPredictionTransformer<TModel>)), ctx)
        {
            Scorer = GetGenericScorer();
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            base.SaveCore(ctx);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RANKPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: RankingPredictionTransformer.LoaderSignature,
                loaderAssemblyName: typeof(RankingPredictionTransformer<>).Assembly.FullName);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on clustering tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class ClusteringPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel>
        where TModel : class
    {
        [BestFriend]
        internal ClusteringPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn,
            float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ClusteringPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            var schema = new RoleMappedSchema(inputSchema, null, featureColumn);

            var args = new ClusteringScorer.Arguments();
            Scorer = new ClusteringScorer(Host, args, new EmptyDataView(Host, inputSchema), BindableMapper.Bind(Host, schema), schema);
        }

        internal ClusteringPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ClusteringPredictionTransformer<TModel>)), ctx)
        {
            // *** Binary format ***
            // <base info>

            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumnName);
            var args = new ClusteringScorer.Arguments();
            Scorer = new ClusteringScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // id of string: scorer threshold column
            base.SaveCore(ctx);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CLUSPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: ClusteringPredictionTransformer.LoaderSignature,
                loaderAssemblyName: typeof(ClusteringPredictionTransformer<>).Assembly.FullName);
        }
    }

    internal static class BinaryPredictionTransformer
    {
        public const string LoaderSignature = "BinaryPredXfer";

        public static BinaryPredictionTransformer<IPredictorProducing<float>> Create(IHostEnvironment env, ModelLoadContext ctx)
            => new BinaryPredictionTransformer<IPredictorProducing<float>>(env, ctx);
    }

    internal static class MulticlassPredictionTransformer
    {
        public const string LoaderSignature = "MulticlassPredXfer";

        public static MulticlassPredictionTransformer<IPredictorProducing<VBuffer<float>>> Create(IHostEnvironment env, ModelLoadContext ctx)
            => new MulticlassPredictionTransformer<IPredictorProducing<VBuffer<float>>>(env, ctx);
    }

    internal static class RegressionPredictionTransformer
    {
        public const string LoaderSignature = "RegressionPredXfer";

        public static RegressionPredictionTransformer<IPredictorProducing<float>> Create(IHostEnvironment env, ModelLoadContext ctx)
            => new RegressionPredictionTransformer<IPredictorProducing<float>>(env, ctx);
    }

    internal static class RankingPredictionTransformer
    {
        public const string LoaderSignature = "RankingPredXfer";

        public static RankingPredictionTransformer<IPredictorProducing<float>> Create(IHostEnvironment env, ModelLoadContext ctx)
            => new RankingPredictionTransformer<IPredictorProducing<float>>(env, ctx);
    }

    internal static class AnomalyPredictionTransformer
    {
        public const string LoaderSignature = "AnomalyPredXfer";

        public static AnomalyPredictionTransformer<IPredictorProducing<float>> Create(IHostEnvironment env, ModelLoadContext ctx)
            => new AnomalyPredictionTransformer<IPredictorProducing<float>>(env, ctx);
    }

    internal static class ClusteringPredictionTransformer
    {
        public const string LoaderSignature = "ClusteringPredXfer";

        public static ClusteringPredictionTransformer<IPredictorProducing<VBuffer<float>>> Create(IHostEnvironment env, ModelLoadContext ctx)
            => new ClusteringPredictionTransformer<IPredictorProducing<VBuffer<float>>>(env, ctx);
    }
}
