// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;

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

namespace Microsoft.ML.Runtime.Data
{

    /// <summary>
    /// Base class for transformers with no feature column, or more than one feature columns.
    /// </summary>
    /// <typeparam name="TModel"></typeparam>
    /// <typeparam name="TScorer">The Scorer used by this <see cref="IPredictionTransformer{TModel}"/></typeparam>
    public abstract class PredictionTransformerBase<TModel, TScorer> : IPredictionTransformer<TModel>
        where TScorer : RowToRowScorerBase
        where TModel : class, IPredictor
    {
        /// <summary>
        /// The model.
        /// </summary>
        public TModel Model { get; }

        protected const string DirModel = "Model";
        protected const string DirTransSchema = "TrainSchema";
        protected readonly IHost Host;
        protected ISchemaBindableMapper BindableMapper;
        protected ISchema TrainSchema;

        public bool IsRowToRowMapper => true;

        protected abstract TScorer Scorer { get; set; }

        protected PredictionTransformerBase(IHost host, TModel model, ISchema trainSchema)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;

            Host.CheckValue(trainSchema, nameof(trainSchema));
            Model = model;

            Host.CheckValue(trainSchema, nameof(trainSchema));
            TrainSchema = trainSchema;
        }

        protected PredictionTransformerBase(IHost host, ModelLoadContext ctx)
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
        /// <param name="inputSchema">The <see cref="ISchema"/> of the input data.</param>
        /// <returns>The resulting <see cref="ISchema"/>.</returns>
        public abstract ISchema GetOutputSchema(ISchema inputSchema);

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
        public IRowToRowMapper GetRowToRowMapper(ISchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return (IRowToRowMapper)Scorer.ApplyToData(Host, new EmptyDataView(Host, inputSchema));
        }

        protected void SaveModel(ModelSaveContext ctx)
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
    /// <typeparam name="TScorer">The scorer used on this PredictionTransformer.</typeparam>
    public abstract class SingleFeaturePredictionTransformerBase<TModel, TScorer> : PredictionTransformerBase<TModel, TScorer>, ISingleFeaturePredictionTransformer<TModel>, ICanSaveModel
        where TModel : class, IPredictor
        where TScorer: RowToRowScorerBase
    {
        /// <summary>
        /// The name of the feature column used by the prediction transformer.
        /// </summary>
        public string FeatureColumn { get; }

        /// <summary>
        /// The type of the prediction transformer
        /// </summary>
        public ColumnType FeatureColumnType { get; }

        protected override TScorer Scorer { get; set; }

        /// <summary>
        /// Initializes a new reference of <see cref="SingleFeaturePredictionTransformerBase{TModel, TScorer}"/>.
        /// </summary>
        /// <param name="host">The local instance of <see cref="IHost"/>.</param>
        /// <param name="model">The model used for scoring.</param>
        /// <param name="trainSchema">The schema of the training data.</param>
        /// <param name="featureColumn">The feature column name.</param>
        public SingleFeaturePredictionTransformerBase(IHost host, TModel model, ISchema trainSchema, string featureColumn)
            : base(host, model, trainSchema)
        {
            FeatureColumn = featureColumn;

            FeatureColumn = featureColumn;
            if (featureColumn == null)
                FeatureColumnType = null;
            else if (!trainSchema.TryGetColumnIndex(featureColumn, out int col))
                throw Host.ExceptSchemaMismatch(nameof(featureColumn), RoleMappedSchema.ColumnRole.Feature.Value, featureColumn);
            else
                FeatureColumnType = trainSchema.GetColumnType(col);

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, model);
        }

        internal SingleFeaturePredictionTransformerBase(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            FeatureColumn = ctx.LoadStringOrNull();

            if (FeatureColumn == null)
                FeatureColumnType = null;
            else if (!TrainSchema.TryGetColumnIndex(FeatureColumn, out int col))
                throw Host.ExceptSchemaMismatch(nameof(FeatureColumn), RoleMappedSchema.ColumnRole.Feature.Value, FeatureColumn);
            else
                FeatureColumnType = TrainSchema.GetColumnType(col);

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, Model);
        }

        public override ISchema GetOutputSchema(ISchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (FeatureColumn != null)
            {
                if (!inputSchema.TryGetColumnIndex(FeatureColumn, out int col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), RoleMappedSchema.ColumnRole.Feature.Value, FeatureColumn, FeatureColumnType.ToString(), null);
                if (!inputSchema.GetColumnType(col).Equals(FeatureColumnType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), RoleMappedSchema.ColumnRole.Feature.Value, FeatureColumn, FeatureColumnType.ToString(), inputSchema.GetColumnType(col).ToString());
            }

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        protected virtual void SaveCore(ModelSaveContext ctx)
        {
            SaveModel(ctx);
            ctx.SaveStringOrNull(FeatureColumn);
        }

        protected virtual GenericScorer GetGenericScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumn);
            return new GenericScorer(Host, new GenericScorer.Arguments(), new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on anomaly detection tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class AnomalyPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel, BinaryClassifierScorer>
        where TModel : class, IPredictorProducing<float>
    {
        public readonly string ThresholdColumn;
        public readonly float Threshold;

        public AnomalyPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn,
            float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            Threshold = threshold;
            ThresholdColumn = thresholdColumn;

            SetScorer();
        }

        public AnomalyPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
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
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumn);
            var args = new BinaryClassifierScorer.Arguments { Threshold = Threshold, ThresholdColumn = ThresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        protected override void SaveCore(ModelSaveContext ctx)
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
                loaderSignature: AnomalyPredictionTransformer.LoaderSignature);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on binary classification tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class BinaryPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel, BinaryClassifierScorer>
        where TModel : class, IPredictorProducing<float>
    {
        public readonly string ThresholdColumn;
        public readonly float Threshold;

        public BinaryPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn,
            float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            Threshold = threshold;
            ThresholdColumn = thresholdColumn;

            SetScorer();
        }

        public BinaryPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
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
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumn);
            var args = new BinaryClassifierScorer.Arguments { Threshold = Threshold, ThresholdColumn = ThresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        protected override void SaveCore(ModelSaveContext ctx)
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
                loaderSignature: BinaryPredictionTransformer.LoaderSignature);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on multi-class classification tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class MulticlassPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel, MultiClassClassifierScorer>
        where TModel : class, IPredictorProducing<VBuffer<float>>
    {
        private readonly string _trainLabelColumn;

        public MulticlassPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn, string labelColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckValueOrNull(labelColumn);

            _trainLabelColumn = labelColumn;
            SetScorer();
        }

        public MulticlassPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
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
            var schema = new RoleMappedSchema(TrainSchema, _trainLabelColumn, FeatureColumn);
            var args = new MultiClassClassifierScorer.Arguments();
            Scorer = new MultiClassClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        protected override void SaveCore(ModelSaveContext ctx)
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
                loaderSignature: MulticlassPredictionTransformer.LoaderSignature);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on regression tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class RegressionPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel, GenericScorer>
        where TModel : class, IPredictorProducing<float>
    {
        public RegressionPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Scorer = GetGenericScorer();
        }

        internal RegressionPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), ctx)
        {
            Scorer = GetGenericScorer();
        }

        protected override void SaveCore(ModelSaveContext ctx)
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
                loaderSignature: RegressionPredictionTransformer.LoaderSignature);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on ranking tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class RankingPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel, GenericScorer>
    where TModel : class, IPredictorProducing<float>
    {
        public RankingPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RankingPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Scorer = GetGenericScorer();
        }

        internal RankingPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RankingPredictionTransformer<TModel>)), ctx)
        {
            Scorer = GetGenericScorer();
        }

        protected override void SaveCore(ModelSaveContext ctx)
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
                modelSignature: "RANK PRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: RankingPredictionTransformer.LoaderSignature);
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
}
