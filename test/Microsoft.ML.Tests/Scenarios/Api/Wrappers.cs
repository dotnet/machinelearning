using Microsoft.ML.Core.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Tests.Scenarios.Api;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

[assembly: LoadableClass(typeof(TransformWrapper), null, typeof(SignatureLoadModel),
    "Transform wrapper", TransformWrapper.LoaderSignature)]
[assembly: LoadableClass(typeof(LoaderWrapper), null, typeof(SignatureLoadModel),
    "Loader wrapper", LoaderWrapper.LoaderSignature)]

namespace Microsoft.ML.Tests.Scenarios.Api
{
    using LinearModel = LinearPredictor;

    public sealed class LoaderWrapper : IDataReader<IMultiStreamSource>, ICanSaveModel
    {
        public const string LoaderSignature = "LoaderWrapper";

        private readonly IHostEnvironment _env;
        private readonly Func<IMultiStreamSource, IDataLoader> _loaderFactory;

        public LoaderWrapper(IHostEnvironment env, Func<IMultiStreamSource, IDataLoader> loaderFactory)
        {
            _env = env;
            _loaderFactory = loaderFactory;
        }

        public ISchema GetOutputSchema()
        {
            var emptyData = Read(new MultiFileSource(null));
            return emptyData.Schema;
        }

        public IDataView Read(IMultiStreamSource input) => _loaderFactory(input);

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            var ldr = Read(new MultiFileSource(null));
            ctx.SaveModel(ldr, "Loader");
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LDR WRPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public LoaderWrapper(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.CheckAtModel(GetVersionInfo());
            ctx.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));

            var loaderStream = new MemoryStream();
            using (var rep = RepositoryWriter.CreateNew(loaderStream))
            {
                ModelSaveContext.SaveModel(rep, loader, "Loader");
                rep.Commit();
            }

            _env = env;
            _loaderFactory = (IMultiStreamSource source) =>
            {
                using (var rep = RepositoryReader.Open(loaderStream))
                {
                    ModelLoadContext.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var ldr, rep, "Loader", source);
                    return ldr;
                }
            };

        }
    }

    public class TransformWrapper : ITransformer, ICanSaveModel
    {
        public const string LoaderSignature = "TransformWrapper";
        private const string TransformDirTemplate = "Step_{0:000}";

        protected readonly IHostEnvironment _env;
        protected readonly IDataView _xf;

        public TransformWrapper(IHostEnvironment env, IDataView xf)
        {
            _env = env;
            _xf = xf;
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            var dv = new EmptyDataView(_env, inputSchema);
            var output = ApplyTransformUtils.ApplyAllTransformsToData(_env, _xf, dv);
            return output.Schema;
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            var dataPipe = _xf;
            var transforms = new List<IDataTransform>();
            while (dataPipe is IDataTransform xf)
            {
                // REVIEW: a malicious user could construct a loop in the Source chain, that would
                // cause this method to iterate forever (and throw something when the list overflows). There's
                // no way to insulate from ALL malicious behavior.
                transforms.Add(xf);
                dataPipe = xf.Source;
                Contracts.AssertValue(dataPipe);
            }
            transforms.Reverse();

            ctx.SaveSubModel("Loader", c => BinaryLoader.SaveInstance(_env, c, dataPipe.Schema));

            ctx.Writer.Write(transforms.Count);
            for (int i = 0; i < transforms.Count; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.SaveModel(transforms[i], dirName);
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "XF  WRPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public TransformWrapper(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.CheckAtModel(GetVersionInfo());
            int n = ctx.Reader.ReadInt32();

            ctx.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));

            IDataView data = loader;
            for (int i = 0; i < n; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out var xf, dirName, data);
                data = xf;
            }

            _env = env;
            _xf = data;
        }

        public IDataView Transform(IDataView input) => ApplyTransformUtils.ApplyAllTransformsToData(_env, _xf, input);
    }

    public interface IPredictorTransformer<out TModel> : ITransformer
    {
        TModel InnerModel { get; }
    }

    public class ScorerWrapper<TModel> : TransformWrapper, IPredictorTransformer<TModel>
        where TModel : IPredictor
    {
        protected readonly string _featureColumn;

        public ScorerWrapper(IHostEnvironment env, IDataView scorer, TModel trainedModel, string featureColumn)
            : base(env, scorer)
        {
            _featureColumn = featureColumn;
            InnerModel = trainedModel;
        }

        public TModel InnerModel { get; }
    }

    public class BinaryScorerWrapper<TModel> : ScorerWrapper<TModel>
        where TModel : IPredictor
    {
        public BinaryScorerWrapper(IHostEnvironment env, TModel model, ISchema inputSchema, string featureColumn, BinaryClassifierScorer.Arguments args)
            : base(env, MakeScorer(env, inputSchema, featureColumn, model, args), model, featureColumn)
        {
        }

        private static IDataView MakeScorer(IHostEnvironment env, ISchema schema, string featureColumn, TModel model, BinaryClassifierScorer.Arguments args)
        {
            var settings = $"Binary{{{CmdParser.GetSettings(env, args, new BinaryClassifierScorer.Arguments())}}}";
            var mapper = ScoreUtils.GetSchemaBindableMapper(env, model, SubComponent.Parse<IDataScorerTransform, SignatureDataScorer>(settings));
            var edv = new EmptyDataView(env, schema);
            var data = new RoleMappedData(edv, "Label", featureColumn, opt: true);
            return new BinaryClassifierScorer(env, args, data.Data, mapper.Bind(env, data.Schema), data.Schema);
        }

        public BinaryScorerWrapper<TModel> Clone(BinaryClassifierScorer.Arguments scorerArgs)
        {
            var scorer = _xf as IDataScorerTransform;
            return new BinaryScorerWrapper<TModel>(_env, InnerModel, scorer.Source.Schema, _featureColumn, scorerArgs);
        }
    }

    public class MyTextLoader : IDataReaderEstimator<IMultiStreamSource, LoaderWrapper>
    {
        private readonly TextLoader.Arguments _args;
        private readonly IHostEnvironment _env;

        public MyTextLoader(IHostEnvironment env, TextLoader.Arguments args)
        {
            _env = env;
            _args = args;
        }

        public LoaderWrapper Fit(IMultiStreamSource input)
        {
            return new LoaderWrapper(_env, x => new TextLoader(_env, _args, x));
        }

        public SchemaShape GetOutputSchema()
        {
            var emptyData = new TextLoader(_env, _args, new MultiFileSource(null));
            return SchemaShape.Create(emptyData.Schema);
        }
    }

    public abstract class TrainerBase<TTransformer, TModel> : IEstimator<TTransformer>
        where TTransformer : ScorerWrapper<TModel>
        where TModel : IPredictor
    {
        protected readonly IHostEnvironment _env;
        protected readonly string _featureCol;
        protected readonly string _labelCol;
        private readonly bool _cache;
        private readonly bool _normalize;

        protected TrainerBase(IHostEnvironment env, bool cache, bool normalize, string featureColumn, string labelColumn)
        {
            _env = env;
            _cache = cache;
            _normalize = normalize;
            _featureCol = featureColumn;
            _labelCol = labelColumn;
        }

        public TTransformer Fit(IDataView input)
        {
            return TrainTransformer(input);
        }

        protected TTransformer TrainTransformer(IDataView trainSet,
            IDataView validationSet = null, IPredictor initPredictor = null)
        {
            var cachedTrain = _cache ? new CacheDataView(_env, trainSet, prefetch: null) : trainSet;

            var trainRoles = new RoleMappedData(cachedTrain, label: _labelCol, feature: _featureCol);
            var emptyData = new EmptyDataView(_env, trainSet.Schema);
            IDataView normalizer = emptyData;

            if (_normalize && trainRoles.Schema.FeaturesAreNormalized() == false)
            {
                var view = NormalizeTransform.CreateMinMaxNormalizer(_env, trainRoles.Data, name: trainRoles.Schema.Feature.Name);
                normalizer = ApplyTransformUtils.ApplyAllTransformsToData(_env, view, emptyData, cachedTrain);

                trainRoles = new RoleMappedData(view, trainRoles.Schema.GetColumnRoleNames());
            }

            RoleMappedData validRoles;

            if (validationSet == null)
                validRoles = null;
            else
            {
                var cachedValid = _cache ? new CacheDataView(_env, validationSet, prefetch: null) : validationSet;
                cachedValid = ApplyTransformUtils.ApplyAllTransformsToData(_env, normalizer, cachedValid);
                validRoles = new RoleMappedData(cachedValid, label: _labelCol, feature: _featureCol);
            }

            var pred = TrainCore(new TrainContext(trainRoles, validRoles, initPredictor));

            var scoreRoles = new RoleMappedData(normalizer, label: _labelCol, feature: _featureCol);
            return MakeScorer(pred, scoreRoles);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }

        protected abstract TModel TrainCore(TrainContext trainContext);

        protected abstract TTransformer MakeScorer(TModel predictor, RoleMappedData data);

        protected ScorerWrapper<TModel> MakeScorerBasic(TModel predictor, RoleMappedData data)
        {
            var scorer = ScoreUtils.GetScorer(predictor, data, _env, data.Schema);
            return (TTransformer)(new ScorerWrapper<TModel>(_env, scorer, predictor, data.Schema.Feature.Name));
        }
    }

    public class MyTextTransform : IEstimator<TransformWrapper>
    {
        private readonly IHostEnvironment _env;
        private readonly TextTransform.Arguments _args;

        public MyTextTransform(IHostEnvironment env, TextTransform.Arguments args)
        {
            _env = env;
            _args = args;
        }

        public TransformWrapper Fit(IDataView input)
        {
            var xf = TextTransform.Create(_env, _args, input);
            var empty = new EmptyDataView(_env, input.Schema);
            var chunk = ApplyTransformUtils.ApplyAllTransformsToData(_env, xf, empty, input);
            return new TransformWrapper(_env, chunk);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }

    public class MyTermTransform : IEstimator<TransformWrapper>
    {
        private readonly IHostEnvironment _env;
        private readonly string _column;
        private readonly string _srcColumn;

        public MyTermTransform(IHostEnvironment env, string column, string srcColumn = null)
        {
            _env = env;
            _column = column;
            _srcColumn = srcColumn;
        }

        public TransformWrapper Fit(IDataView input)
        {
            var xf = new TermTransform(_env, input, _column, _srcColumn);
            var empty = new EmptyDataView(_env, input.Schema);
            var chunk = ApplyTransformUtils.ApplyAllTransformsToData(_env, xf, empty, input);
            return new TransformWrapper(_env, chunk);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }

    public class MyConcatTransform : IEstimator<TransformWrapper>
    {
        private readonly IHostEnvironment _env;
        private readonly string _name;
        private readonly string[] _source;

        public MyConcatTransform(IHostEnvironment env, string name, params string[] source)
        {
            _env = env;
            _name = name;
            _source = source;
        }

        public TransformWrapper Fit(IDataView input)
        {
            var xf = new ConcatTransform(_env, input, _name, _source);
            var empty = new EmptyDataView(_env, input.Schema);
            var chunk = ApplyTransformUtils.ApplyAllTransformsToData(_env, xf, empty, input);
            return new TransformWrapper(_env, chunk);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }

    public class MyKeyToValueTransform : IEstimator<TransformWrapper>
    {
        private readonly IHostEnvironment _env;
        private readonly string _name;
        private readonly string _source;

        public MyKeyToValueTransform(IHostEnvironment env, string name, string source = null)
        {
            _env = env;
            _name = name;
            _source = source;
        }

        public TransformWrapper Fit(IDataView input)
        {
            var xf = new KeyToValueTransform(_env, input, _name, _source);
            var empty = new EmptyDataView(_env, input.Schema);
            var chunk = ApplyTransformUtils.ApplyAllTransformsToData(_env, xf, empty, input);
            return new TransformWrapper(_env, chunk);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }

    public sealed class MySdca : TrainerBase<BinaryScorerWrapper<IPredictor>, IPredictor>
    {
        private readonly LinearClassificationTrainer.Arguments _args;

        public MySdca(IHostEnvironment env, LinearClassificationTrainer.Arguments args, string featureCol, string labelCol)
            : base(env, true, true, featureCol, labelCol)
        {
            _args = args;
        }

        protected override IPredictor TrainCore(TrainContext context) => new LinearClassificationTrainer(_env, _args).Train(context);

        public ITransformer Train(IDataView trainData, IDataView validationData = null) => TrainTransformer(trainData, validationData);

        protected override BinaryScorerWrapper<IPredictor> MakeScorer(IPredictor predictor, RoleMappedData data)
            => new BinaryScorerWrapper<IPredictor>(_env, predictor, data.Data.Schema, _featureCol, new BinaryClassifierScorer.Arguments());
    }

    public sealed class MySdcaMulticlass : TrainerBase<ScorerWrapper<IPredictor>, IPredictor>
    {
        private readonly SdcaMultiClassTrainer.Arguments _args;

        public MySdcaMulticlass(IHostEnvironment env, SdcaMultiClassTrainer.Arguments args, string featureCol, string labelCol)
            : base(env, true, true, featureCol, labelCol)
        {
            _args = args;
        }

        protected override ScorerWrapper<IPredictor> MakeScorer(IPredictor predictor, RoleMappedData data) => MakeScorerBasic(predictor, data);

        protected override IPredictor TrainCore(TrainContext context) => new SdcaMultiClassTrainer(_env, _args).Train(context);
    }

    public sealed class MyAveragedPerceptron : TrainerBase<BinaryScorerWrapper<IPredictor>, IPredictor>
    {
        private readonly AveragedPerceptronTrainer _trainer;

        public MyAveragedPerceptron(IHostEnvironment env, AveragedPerceptronTrainer.Arguments args, string featureCol, string labelCol)
            : base(env, false, true, featureCol, labelCol)
        {
            _trainer = new AveragedPerceptronTrainer(env, args);
        }

        protected override IPredictor TrainCore(TrainContext trainContext) => _trainer.Train(trainContext);

        public ITransformer Train(IDataView trainData, IPredictor initialPredictor)
        {
            return TrainTransformer(trainData, initPredictor: initialPredictor);
        }

        protected override BinaryScorerWrapper<IPredictor> MakeScorer(IPredictor predictor, RoleMappedData data)
            => new BinaryScorerWrapper<IPredictor>(_env, predictor, data.Data.Schema, _featureCol, new BinaryClassifierScorer.Arguments());
    }

    public sealed class MyPredictionEngine<TSrc, TDst>
                where TSrc : class
                where TDst : class, new()
    {
        private readonly PredictionEngine<TSrc, TDst> _engine;

        public MyPredictionEngine(IHostEnvironment env, ITransformer pipe)
        {
            IDataView dv = env.CreateDataView(new TSrc[0]);
            _engine = env.CreatePredictionEngine<TSrc, TDst>(pipe.Transform(dv));
        }

        public TDst Predict(TSrc example)
        {
            return _engine.Predict(example);
        }
    }

    public sealed class MyBinaryClassifierEvaluator
    {
        private readonly IHostEnvironment _env;
        private readonly BinaryClassifierEvaluator _evaluator;

        public MyBinaryClassifierEvaluator(IHostEnvironment env, BinaryClassifierEvaluator.Arguments args)
        {
            _env = env;
            _evaluator = new BinaryClassifierEvaluator(env, args);
        }

        public BinaryClassificationMetrics Evaluate(IDataView data, string labelColumn, string probabilityColumn)
        {
            var ci = EvaluateUtils.GetScoreColumnInfo(_env, data.Schema, null, "Score", MetadataUtils.Const.ScoreColumnKind.BinaryClassification);
            var map = new KeyValuePair<RoleMappedSchema.ColumnRole, string>[]
            {
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Probability, probabilityColumn),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, ci.Name)
            };
            var rmd = new RoleMappedData(data, labelColumn, "Features", opt: true, custom: map);

            var metricsDict = _evaluator.Evaluate(rmd);
            return BinaryClassificationMetrics.FromMetrics(_env, metricsDict["OverallMetrics"], metricsDict["ConfusionMatrix"]).Single();
        }
    }

    public static class MyCrossValidation
    {
        public sealed class BinaryCrossValidationMetrics
        {
            public readonly ITransformer[] FoldModels;
            public readonly BinaryClassificationMetrics[] FoldMetrics;

            public BinaryCrossValidationMetrics(ITransformer[] models, BinaryClassificationMetrics[] metrics)
            {
                FoldModels = models;
                FoldMetrics = metrics;
            }
        }

        public sealed class BinaryCrossValidator
        {
            private readonly IHostEnvironment _env;

            public int NumFolds { get; set; } = 2;

            public string StratificationColumn { get; set; }

            public string LabelColumn { get; set; } = DefaultColumnNames.Label;

            public BinaryCrossValidator(IHostEnvironment env)
            {
                _env = env;
            }

            public BinaryCrossValidationMetrics CrossValidate(IDataView trainData, IEstimator<ITransformer> estimator)
            {
                var models = new ITransformer[NumFolds];
                var metrics = new BinaryClassificationMetrics[NumFolds];

                if (StratificationColumn == null)
                {
                    StratificationColumn = "StratificationColumn";
                    var random = new GenerateNumberTransform(_env, trainData, StratificationColumn);
                    trainData = random;
                }
                else
                    throw new NotImplementedException();

                var evaluator = new MyBinaryClassifierEvaluator(_env, new BinaryClassifierEvaluator.Arguments() { });

                for (int fold = 0; fold < NumFolds; fold++)
                {
                    var trainFilter = new RangeFilter(_env, new RangeFilter.Arguments()
                    {
                        Column = StratificationColumn,
                        Min = (Double)fold / NumFolds,
                        Max = (Double)(fold + 1) / NumFolds,
                        Complement = true
                    }, trainData);
                    var testFilter = new RangeFilter(_env, new RangeFilter.Arguments()
                    {
                        Column = StratificationColumn,
                        Min = (Double)fold / NumFolds,
                        Max = (Double)(fold + 1) / NumFolds,
                        Complement = false
                    }, trainData);

                    models[fold] = estimator.Fit(trainFilter);
                    var scoredTest = models[fold].Transform(testFilter);
                    metrics[fold] = evaluator.Evaluate(scoredTest, labelColumn: LabelColumn, probabilityColumn: "Probability");
                }

                return new BinaryCrossValidationMetrics(models, metrics);

            }
        }
    }


    public static class MyHelperExtensions
    {
        public static void SaveAsBinary(this IDataView data, IHostEnvironment env, Stream stream)
        {
            var saver = new BinarySaver(env, new BinarySaver.Arguments());
            using (var ch = env.Start("SaveData"))
                DataSaverUtils.SaveDataView(ch, saver, data, stream);
        }

        public static IDataView FitAndTransform(this IEstimator<ITransformer> est, IDataView data) => est.Fit(data).Transform(data);

        public static IDataView FitAndRead<TSource>(this IDataReaderEstimator<TSource, IDataReader<TSource>> est, TSource source)
            => est.Fit(source).Read(source);
    }
}
