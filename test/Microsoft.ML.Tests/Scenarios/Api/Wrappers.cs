using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public sealed class LoaderWrapper : IDataReader<IMultiStreamSource>
    {
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
    }

    public class TransformWrapper : ITransformer
    {
        private readonly IHostEnvironment _env;
        private readonly IDataView _xf;

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

        public IDataView Transform(IDataView input) => ApplyTransformUtils.ApplyAllTransformsToData(_env, _xf, input);
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

    public abstract class TrainerBase : IEstimator<TransformWrapper>
    {
        protected readonly IHostEnvironment _env;
        private readonly string _featureCol;
        private readonly string _labelCol;
        private readonly bool _cache;

        protected TrainerBase(IHostEnvironment env, bool cache, string featureColumn, string labelColumn)
        {
            _env = env;
            _cache = cache;
            _featureCol = featureColumn;
            _labelCol = labelColumn;
        }

        public TransformWrapper Fit(IDataView input)
        {
            var cached = _cache ? new CacheDataView(_env, input, prefetch: null) : input;

            var trainRoles = new RoleMappedData(cached, label: _labelCol, feature: _featureCol);
            var pred = Train(trainRoles);

            var scoreRoles = new RoleMappedData(input, label: _labelCol, feature: _featureCol);
            IDataScorerTransform scorer = ScoreUtils.GetScorer(pred, scoreRoles, _env, trainRoles.Schema);
            return new TransformWrapper(_env, scorer);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }

        protected abstract IPredictor Train(RoleMappedData data);
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

    public sealed class MySdca : TrainerBase
    {
        private readonly LinearClassificationTrainer.Arguments _args;

        public MySdca(IHostEnvironment env, LinearClassificationTrainer.Arguments args, string featureCol, string labelCol)
            : base(env, true, featureCol, labelCol)
        {
            _args = args;
        }

        protected override IPredictor Train(RoleMappedData data) => new LinearClassificationTrainer(_env, _args).Train(data);
    }

    public sealed class MyPredictionEngine<TSrc, TDst>
                where TSrc : class
                where TDst : class, new()
    {
        private readonly PredictionEngine<TSrc, TDst> _engine;

        public MyPredictionEngine(IHostEnvironment env, ISchema inputSchema, ITransformer pipe)
        {
            IDataView dv = new EmptyDataView(env, inputSchema);
            _engine = env.CreatePredictionEngine<TSrc, TDst>(pipe.Transform(dv));
        }

        public TDst Predict(TSrc example)
        {
            return _engine.Predict(example);
        }
    }
}
