using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Multi-threaded prediction. A twist on "Simple train and predict", where we account that
        /// multiple threads may want predictions at the same time. Because we deliberately do not
        /// reallocate internal memory buffers on every single prediction, the PredictionEngine
        /// (or its estimator/transformer based successor) is, like most stateful .NET objects,
        /// fundamentally not thread safe. This is deliberate and as designed. However, some mechanism
        /// to enable multi-threaded scenarios (e.g., a web server servicing requests) should be possible
        /// and performant in the new API.
        /// </summary>
        [Fact]
        void MultithreadedPrediction()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(), loader);

                // Train
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                });

                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);

                // Create prediction engine and test predictions.
                var model = env.CreatePredictionEngine<SentimentData, SentimentPrediction>(scorer);

                // Take a couple examples out of the test data and run predictions on top.
                var testLoader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(SentimentTestPath)));
                var testData = testLoader.AsEnumerable<SentimentData>(env, false);
                var lockEngine = new LockBasedPredictionEngine(env, model);
                Parallel.ForEach(testData, (input) => lockEngine.Predict(input));
                int numThreads = 2;
                PredictionEngine<SentimentData, SentimentPrediction>[] models = new PredictionEngine<SentimentData, SentimentPrediction>[numThreads];
                using (var file = env.CreateTempFile())
                {
                    // Save model. 
                    using (var ch = env.Start("saving"))
                        TrainUtils.SaveModel(env, ch, file, predictor, scoreRoles);
                    var threadEngine = new ThreadLocalBasedPredictionEngine(env, file);
                    var poolEngine = new PoolBasedPredictionEngine(env, file);
                    Parallel.ForEach(testData, (input) => threadEngine.Predict(input));
                    Parallel.ForEach(testData, (input) => poolEngine.Predict(input));
                }
            }
        }


        /// <summary>
        /// This is a trivial implementation of a thread-safe prediction engine, where the underlying <see cref="SimplePredictionEngine"/>
        /// is just guarded by a lock.
        /// </summary>
        private sealed class LockBasedPredictionEngine
        {
            private readonly PredictionEngine<SentimentData, SentimentPrediction> _model;

            public LockBasedPredictionEngine(IHostEnvironment env, PredictionEngine<SentimentData, SentimentPrediction> model)
            {
                _model = model;
            }

            public SentimentPrediction Predict(SentimentData input)
            {
                lock (_model)
                {
                    return _model.Predict(input);
                }
            }
        }

        /// <summary>
        /// This is an implementation of a thread-safe prediction engine that works by instantiating one <see cref="SimplePredictionEngine"/>
        /// per the worker thread.
        /// </summary>
        private sealed class ThreadLocalBasedPredictionEngine
        {
            private readonly ThreadLocal<PredictionEngine<SentimentData, SentimentPrediction>> _engine;

            public ThreadLocalBasedPredictionEngine(IHostEnvironment env, IFileHandle fileHandle)
            {
                _engine = new ThreadLocal<PredictionEngine<SentimentData, SentimentPrediction>>(
                    () =>
                    {
                        using (var fs = fileHandle.OpenReadStream())
                            return env.CreatePredictionEngine<SentimentData, SentimentPrediction>(fs);
                    });
            }

            public SentimentPrediction Predict(SentimentData input)
            {
                return _engine.Value.Predict(input);
            }
        }

        /// <summary>
        /// This is an implementation of a thread-safe prediction engine that works by keeping a pool of allocated
        /// <see cref="SimplePredictionEngine"/> objects, that is grown as needed. 
        /// </summary>
        private sealed class PoolBasedPredictionEngine
        {
            private readonly MadeObjectPool<PredictionEngine<SentimentData, SentimentPrediction>> _enginePool;

            public PoolBasedPredictionEngine(IHostEnvironment env, IFileHandle fileHandle)
            {
                _enginePool = new MadeObjectPool<PredictionEngine<SentimentData, SentimentPrediction>>(
                    () =>
                    {
                        using (var fs = fileHandle.OpenReadStream())
                            return env.CreatePredictionEngine<SentimentData, SentimentPrediction>(fs);
                    });
            }

            public SentimentPrediction Predict(SentimentData features)
            {
                var engine = _enginePool.Get();
                try
                {
                    return engine.Predict(features);
                }
                finally
                {
                    _enginePool.Return(engine);
                }
            }
        }

    }
}
