// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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

namespace Microsoft.ML.Runtime.Data
{
    public abstract class PredictionTransformerBase<TModel> : IPredictionTransformer<TModel>, ICanSaveModel
        where TModel : class, IPredictor
    {
        private const string DirModel = "Model";
        private const string DirTransSchema = "TrainSchema";

        protected readonly IHost Host;
        protected readonly ISchemaBindableMapper BindableMapper;
        protected readonly ISchema TrainSchema;

        public string FeatureColumn { get; }

        public ColumnType FeatureColumnType { get; }

        public TModel Model { get; }

        public PredictionTransformerBase(IHost host, TModel model, ISchema trainSchema, string featureColumn)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;
            Host.CheckValue(trainSchema, nameof(trainSchema));

            Model = model;
            FeatureColumn = featureColumn;
            if (!trainSchema.TryGetColumnIndex(featureColumn, out int col))
                throw Host.ExceptSchemaMismatch(nameof(featureColumn), RoleMappedSchema.ColumnRole.Feature.Value, featureColumn);
            FeatureColumnType = trainSchema.GetColumnType(col);

            TrainSchema = trainSchema;
            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, model);
        }

        internal PredictionTransformerBase(IHost host, ModelLoadContext ctx)
        {
            Host = host;

            ctx.LoadModel<TModel, SignatureLoadModel>(host, out TModel model, DirModel);
            Model = model;

            // *** Binary format ***
            // model: prediction model.
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

            FeatureColumn = ctx.LoadString();
            if (!TrainSchema.TryGetColumnIndex(FeatureColumn, out int col))
                throw Host.ExceptSchemaMismatch(nameof(FeatureColumn), RoleMappedSchema.ColumnRole.Feature.Value, FeatureColumn);
            FeatureColumnType = TrainSchema.GetColumnType(col);

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, model);
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryGetColumnIndex(FeatureColumn, out int col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), RoleMappedSchema.ColumnRole.Feature.Value, FeatureColumn, FeatureColumnType.ToString(), null);
            if (!inputSchema.GetColumnType(col).Equals(FeatureColumnType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), RoleMappedSchema.ColumnRole.Feature.Value, FeatureColumn, FeatureColumnType.ToString(), inputSchema.GetColumnType(col).ToString());

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        public abstract IDataView Transform(IDataView input);

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        protected virtual void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // model: prediction model.
            // stream: empty data view that contains train schema.
            // id of string: feature column.

            ctx.SaveModel(Model, DirModel);
            ctx.SaveBinaryStream(DirTransSchema, writer =>
            {
                using (var ch = Host.Start("Saving train schema"))
                {
                    var saver = new BinarySaver(Host, new BinarySaver.Arguments { Silent = true });
                    DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(Host, TrainSchema), writer.BaseStream);
                }
            });

            ctx.SaveString(FeatureColumn);
        }
    }

    public sealed class BinaryPredictionTransformer<TModel> : PredictionTransformerBase<TModel>
        where TModel : class, IPredictorProducing<float>
    {
        private readonly BinaryClassifierScorer _scorer;

        public readonly string ThresholdColumn;
        public readonly float Threshold;

        public BinaryPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn,
            float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(BinaryPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            var schema = new RoleMappedSchema(inputSchema, null, featureColumn);
            Threshold = threshold;
            ThresholdColumn = thresholdColumn;

            var args = new BinaryClassifierScorer.Arguments { Threshold = Threshold, ThresholdColumn = ThresholdColumn };
            _scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, inputSchema), BindableMapper.Bind(Host, schema), schema);
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

            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumn);
            var args = new BinaryClassifierScorer.Arguments { Threshold = Threshold, ThresholdColumn = ThresholdColumn };
            _scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public override IDataView Transform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return _scorer.ApplyToData(Host, input);
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

    public sealed class MulticlassPredictionTransformer<TModel> : PredictionTransformerBase<TModel>
        where TModel : class, IPredictorProducing<VBuffer<float>>
    {
        private readonly MultiClassClassifierScorer _scorer;
        private readonly string _trainLabelColumn;

        public MulticlassPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn, string labelColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Host.CheckValueOrNull(labelColumn);

            _trainLabelColumn = labelColumn;
            var schema = new RoleMappedSchema(inputSchema, labelColumn, featureColumn);
            var args = new MultiClassClassifierScorer.Arguments();
            _scorer = new MultiClassClassifierScorer(Host, args, new EmptyDataView(Host, inputSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public MulticlassPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MulticlassPredictionTransformer<TModel>)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // id of string: train label column

            _trainLabelColumn = ctx.LoadStringOrNull();

            var schema = new RoleMappedSchema(TrainSchema, _trainLabelColumn, FeatureColumn);
            var args = new MultiClassClassifierScorer.Arguments();
            _scorer = new MultiClassClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public override IDataView Transform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return _scorer.ApplyToData(Host, input);
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

    public sealed class RegressionPredictionTransformer<TModel> : PredictionTransformerBase<TModel>
        where TModel : class, IPredictorProducing<float>
    {
        private readonly GenericScorer _scorer;

        public RegressionPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            var schema = new RoleMappedSchema(inputSchema, null, featureColumn);
            _scorer = new GenericScorer(Host, new GenericScorer.Arguments(), new EmptyDataView(Host, inputSchema), BindableMapper.Bind(Host, schema), schema);
        }

        internal RegressionPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), ctx)
        {
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumn);
            _scorer = new GenericScorer(Host, new GenericScorer.Arguments(), new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public override IDataView Transform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return _scorer.ApplyToData(Host, input);
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
                modelSignature: "MC  PRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: RegressionPredictionTransformer.LoaderSignature);
        }
    }

    public sealed class RankingPredictionTransformer<TModel> : PredictionTransformerBase<TModel>
    where TModel : class, IPredictorProducing<float>
    {
        private readonly GenericScorer _scorer;

        public RankingPredictionTransformer(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            var schema = new RoleMappedSchema(inputSchema, null, featureColumn);
            _scorer = new GenericScorer(Host, new GenericScorer.Arguments(), new EmptyDataView(Host, inputSchema), BindableMapper.Bind(Host, schema), schema);
        }

        internal RankingPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), ctx)
        {
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumn);
            _scorer = new GenericScorer(Host, new GenericScorer.Arguments(), new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public override IDataView Transform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return _scorer.ApplyToData(Host, input);
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
                modelSignature: "MC  RANK",
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
}
