// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(typeof(ISingleFeaturePredictionTransformer<object>), typeof(BinaryPredictionTransformer), null, typeof(SignatureLoadModel),
    "", BinaryPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ISingleFeaturePredictionTransformer<object>), typeof(MulticlassPredictionTransformer), null, typeof(SignatureLoadModel),
    "", MulticlassPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ISingleFeaturePredictionTransformer<object>), typeof(RegressionPredictionTransformer), null, typeof(SignatureLoadModel),
    "", RegressionPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ISingleFeaturePredictionTransformer<object>), typeof(RankingPredictionTransformer), null, typeof(SignatureLoadModel),
    "", RankingPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(AnomalyPredictionTransformer<IPredictorProducing<float>>), typeof(AnomalyPredictionTransformer), null, typeof(SignatureLoadModel),
    "", AnomalyPredictionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ClusteringPredictionTransformer<IPredictorProducing<VBuffer<float>>>), typeof(ClusteringPredictionTransformer), null, typeof(SignatureLoadModel),
    "", ClusteringPredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.Data
{
    internal static class PredictionTransformerBase
    {
        internal const string DirModel = "Model";
    }

    /// <summary>
    /// Base class for transformers with no feature column, or more than one feature columns.
    /// </summary>
    /// <typeparam name="TModel">The type of the model parameters used by this prediction transformer.</typeparam>
    public abstract class PredictionTransformerBase<TModel> : IPredictionTransformer<TModel>, IDisposable
        where TModel : class
    {
        /// <summary>
        /// The model.
        /// </summary>
        public TModel Model { get; }

        private protected IPredictor ModelAsPredictor => (IPredictor)Model;

        [BestFriend]
        private protected const string DirModel = PredictionTransformerBase.DirModel;
        [BestFriend]
        private protected const string DirTransSchema = "TrainSchema";
        [BestFriend]
        private protected readonly IHost Host;
        [BestFriend]
        private protected ISchemaBindableMapper BindableMapper;
        [BestFriend]
        internal DataViewSchema TrainSchema;

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

            ctx.LoadModel<TModel, SignatureLoadModel>(host, out TModel model, DirModel);
            Model = model;

            InitializeLogic(host, ctx);
        }

        [BestFriend]
        private protected PredictionTransformerBase(IHost host, ModelLoadContext ctx, TModel model)
        {
            Host = host;
            Model = model; // prediction model
            InitializeLogic(host, ctx);
        }

        private void InitializeLogic(IHost host, ModelLoadContext ctx)
        {
            // *** Binary format ***
            // stream: empty data view that contains train schema.
            // id of string: feature column.

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

        #region IDisposable Support
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
                return;

            (Model as IDisposable)?.Dispose();
            (BindableMapper as IDisposable)?.Dispose();
            (Scorer as IDisposable)?.Dispose();

            _disposed = true;
        }
        #endregion
    }

    /// <summary>
    /// The base class for all the transformers implementing the <see cref="ISingleFeaturePredictionTransformer{TModel}"/>.
    /// Those are all the transformers that work with one feature column.
    /// </summary>
    /// <typeparam name="TModel">The model used to transform the data.</typeparam>
    public abstract class SingleFeaturePredictionTransformerBase<TModel> : PredictionTransformerBase<TModel>, ISingleFeaturePredictionTransformer<TModel>, ISingleFeaturePredictionTransformer
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

        private protected SingleFeaturePredictionTransformerBase(IHost host, ModelLoadContext ctx, TModel model)
            : base(host, ctx, model)
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
            float threshold = 0.5f, string thresholdColumn = DefaultColumnNames.Score)
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
        internal readonly string LabelColumnName;

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

        internal BinaryPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn, string labelColumn,
            float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            Threshold = threshold;
            ThresholdColumn = thresholdColumn;
            LabelColumnName = labelColumn;

            SetScorer();
        }
        internal BinaryPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<TModel>)), ctx)
        {
            InitializationLogic(ctx, out Threshold, out ThresholdColumn);
        }

        internal BinaryPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx, IHost host, TModel model)
            : base(host, ctx, model)
        {
            InitializationLogic(ctx, out Threshold, out ThresholdColumn);
        }

        private void InitializationLogic(ModelLoadContext ctx, out float threshold, out string thresholdcolumn)
        {
            // *** Binary format ***
            // <base info>
            // float: scorer threshold
            // id of string: scorer threshold column

            threshold = ctx.Reader.ReadSingle();
            thresholdcolumn = ctx.LoadString();
            SetScorer();
        }

        private void SetScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, LabelColumnName, FeatureColumnName);
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
        private readonly string _scoreColumn;
        private readonly string _predictedLabelColumn;

        [BestFriend]
        internal MulticlassPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn, string labelColumn,
            string scoreColumn = AnnotationUtils.Const.ScoreValueKind.Score, string predictedLabel = DefaultColumnNames.PredictedLabel) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckValueOrNull(labelColumn);

            _trainLabelColumn = labelColumn;
            _scoreColumn = scoreColumn;
            _predictedLabelColumn = predictedLabel;
            SetScorer();
        }

        internal MulticlassPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<TModel>)), ctx)
        {
            InitializationLogic(ctx, out _trainLabelColumn, out _scoreColumn, out _predictedLabelColumn);
        }

        internal MulticlassPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx, IHost host, TModel model)
            : base(host, ctx, model)
        {

            InitializationLogic(ctx, out _trainLabelColumn, out _scoreColumn, out _predictedLabelColumn);
        }

        private void InitializationLogic(ModelLoadContext ctx, out string trainLabelColumn, out string scoreColumn, out string predictedLabelColumn)
        {
            // *** Binary format ***
            // <base info>
            // id of string: train label column

            trainLabelColumn = ctx.LoadStringOrNull();
            if (ctx.Header.ModelVerWritten >= 0x00010002)
            {
                scoreColumn = ctx.LoadStringOrNull();
                predictedLabelColumn = ctx.LoadStringOrNull();
            }
            else
            {
                scoreColumn = AnnotationUtils.Const.ScoreValueKind.Score;
                predictedLabelColumn = DefaultColumnNames.PredictedLabel;
            }

            SetScorer();
        }

        private void SetScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, _trainLabelColumn, FeatureColumnName);
            var args = new MulticlassClassificationScorer.Arguments() { ScoreColumnName = _scoreColumn, PredictedLabelColumnName = _predictedLabelColumn };
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
            ctx.SaveStringOrNull(_scoreColumn);
            ctx.SaveStringOrNull(_predictedLabelColumn);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MC  PRED",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Score and Predicted Label column names.
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

        internal RegressionPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx, IHost host, TModel model)
            : base(host, ctx, model)
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

        internal RankingPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx, IHost host, TModel model)
            : base(host, ctx, model)
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
        private const string DirModel = PredictionTransformerBase.DirModel;

        public static ISingleFeaturePredictionTransformer<object> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            // Load internal model
            var host = Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<IPredictorProducing<float>>));
            ctx.LoadModel<IPredictorProducing<float>, SignatureLoadModel>(host, out IPredictorProducing<float> model, DirModel);

            // Returns prediction transformer using the right TModel from the previously loaded model
            Type predictionTransformerType = typeof(BinaryPredictionTransformer<>);
            return (ISingleFeaturePredictionTransformer<object>)CreatePredictionTransformer.Create(env, ctx, host, model, predictionTransformerType);
        }
    }

    internal static class MulticlassPredictionTransformer
    {
        public const string LoaderSignature = "MulticlassPredXfer";
        private const string DirModel = PredictionTransformerBase.DirModel;

        public static ISingleFeaturePredictionTransformer<object> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            // Load internal model
            var host = Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<IPredictorProducing<VBuffer<float>>>));
            ctx.LoadModel<IPredictorProducing<VBuffer<float>>, SignatureLoadModel>(host, out IPredictorProducing<VBuffer<float>> model, DirModel);

            // Returns prediction transformer using the right TModel from the previously loaded model
            Type predictionTransformerType = typeof(MulticlassPredictionTransformer<>);
            return (ISingleFeaturePredictionTransformer<object>)CreatePredictionTransformer.Create(env, ctx, host, model, predictionTransformerType);
        }
    }

    internal static class RegressionPredictionTransformer
    {
        public const string LoaderSignature = "RegressionPredXfer";
        private const string DirModel = PredictionTransformerBase.DirModel;

        public static ISingleFeaturePredictionTransformer<object> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            // Load internal model
            var host = Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<IPredictorProducing<float>>));
            ctx.LoadModel<IPredictorProducing<float>, SignatureLoadModel>(host, out IPredictorProducing<float> model, DirModel);

            // Returns prediction transformer using the right TModel from the previously loaded model
            Type predictionTransformerType = typeof(RegressionPredictionTransformer<>);
            return (ISingleFeaturePredictionTransformer<object>)CreatePredictionTransformer.Create(env, ctx, host, model, predictionTransformerType);

        }
    }

    internal static class RankingPredictionTransformer
    {
        public const string LoaderSignature = "RankingPredXfer";
        private const string DirModel = PredictionTransformerBase.DirModel;

        public static ISingleFeaturePredictionTransformer<object> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            // Load internal model
            var host = Contracts.CheckRef(env, nameof(env)).Register(nameof(RankingPredictionTransformer<IPredictorProducing<float>>));
            ctx.LoadModel<IPredictorProducing<float>, SignatureLoadModel>(host, out IPredictorProducing<float> model, DirModel);

            // Returns prediction transformer using the right TModel from the previously loaded model
            Type predictionTransformerType = typeof(RankingPredictionTransformer<>);
            return (ISingleFeaturePredictionTransformer<object>)CreatePredictionTransformer.Create(env, ctx, host, model, predictionTransformerType);
        }
    }

    internal static class CreatePredictionTransformer
    {
        internal static object Create(IHostEnvironment env, ModelLoadContext ctx, IHost host, IPredictorProducing<float> model, Type predictionTransformerType)
        {
            // Create generic type of the prediction transformer using the correct TModel.
            // Return an instance of that type, passing the previously loaded model to the constructor

            var genericCtor = CreateConstructor(model.GetType(), predictionTransformerType);
            var genericInstance = genericCtor.Invoke(new object[] { env, ctx, host, model });

            return genericInstance;
        }

        internal static object Create(IHostEnvironment env, ModelLoadContext ctx, IHost host, IPredictorProducing<VBuffer<float>> model, Type predictionTransformerType)
        {
            // Create generic type of the prediction transformer using the correct TModel.
            // Return an instance of that type, passing the previously loaded model to the constructor

            var genericCtor = CreateConstructor(model.GetType(), predictionTransformerType);
            var genericInstance = genericCtor.Invoke(new object[] { env, ctx, host, model });

            return genericInstance;
        }

        private static ConstructorInfo CreateConstructor(Type modelType, Type predictionTransformerType)
        {
            Type modelLoadType = GetLoadType(modelType);
            Type[] genericTypeArgs = { modelLoadType };
            Type constructedType = predictionTransformerType.MakeGenericType(genericTypeArgs);

            Type[] constructorArgs = {
                typeof(IHostEnvironment),
                typeof(ModelLoadContext),
                typeof(IHost),
                modelLoadType
            };

            var genericCtor = constructedType.GetConstructor(BindingFlags.NonPublic | BindingFlags.Instance, null, constructorArgs, null);
            return genericCtor;
        }

        private static Type GetLoadType(Type modelType)
        {
            // Returns the type that should be assigned as TModel of the Prediction Transformer being loaded
            var att = modelType.GetCustomAttribute(typeof(PredictionTransformerLoadTypeAttribute)) as PredictionTransformerLoadTypeAttribute;
            if (att != null)
            {
                if (att.LoadType.IsGenericType && att.LoadType.GetGenericArguments().Length == modelType.GetGenericArguments().Length)
                {
                    // This assumes that if att.LoadType and modelType have the same number of type parameters
                    // Then they should get the same type parameters.
                    // This is the case for CalibratedModelParametersBase and its children generic clases.
                    // But might break if other classes begin using the PredictionTransformerLoadTypeAttribute in the future.
                    Type[] typeArguments = modelType.GetGenericArguments();
                    Type genericType = att.LoadType;
                    return genericType.MakeGenericType(typeArguments);
                }
            }

            return modelType;
        }
    }

    [AttributeUsage(AttributeTargets.Class)]
    internal class PredictionTransformerLoadTypeAttribute : Attribute
    {
        internal Type LoadType { get; }
        internal PredictionTransformerLoadTypeAttribute(Type loadtype)
        {
            LoadType = loadtype;
        }

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
