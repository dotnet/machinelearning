// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.PipelineApi
{
    public partial class PipelineApiScenarioTests
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
            var testDataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();

            pipeline.Add(MakeSentimentTextLoader(dataPath));

            pipeline.Add(MakeSentimentTextTransform());

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            var modelName = "multithreadModel.zip";
            DeleteOutputPath(modelName);
            model.WriteAsync(modelName);
            var collection = new List<SentimentData>();
            int numExamples = 100;
            for (int i = 0; i < numExamples; i++)
            {
                collection.Add(new SentimentData() { SentimentText = "Let's predict this one!" });
            }

            var lockEngine = new LockBasedPredictionEngine(model);
            Parallel.ForEach(collection, (input) => lockEngine.Predict(input));
            var threadEngine = new ThreadLocalBasedPredictionEngine(modelName);
            var poolEngine = new PoolBasedPredictionEngine(modelName);
            Parallel.ForEach(collection, (input) => threadEngine.Predict(input));
            Parallel.ForEach(collection, (input) => poolEngine.Predict(input));
        }

        /// <summary>
        /// This is a trivial implementation of a thread-safe prediction engine is just guarded by a lock.
        /// </summary>
        private sealed class LockBasedPredictionEngine
        {
            private readonly PredictionModel<SentimentData, SentimentPrediction> _model;

            public LockBasedPredictionEngine(PredictionModel<SentimentData, SentimentPrediction> model)
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
        /// This is an implementation of a thread-safe prediction engine that works by instantiating one model per the worker thread.
        /// </summary>
        private sealed class ThreadLocalBasedPredictionEngine
        {
            private readonly ThreadLocal<PredictionModel<SentimentData, SentimentPrediction>> _engine;

            public ThreadLocalBasedPredictionEngine(string modelFile)
            {
                _engine = new ThreadLocal<PredictionModel<SentimentData, SentimentPrediction>>(
                    () =>
                    {
                        var model = PredictionModel.ReadAsync<SentimentData, SentimentPrediction>(modelFile);
                        model.Wait();
                        return model.Result;
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
            private readonly MadeObjectPool<PredictionModel<SentimentData, SentimentPrediction>> _enginePool;

            public PoolBasedPredictionEngine(string modelFile)
            {
                _enginePool = new MadeObjectPool<PredictionModel<SentimentData, SentimentPrediction>>(
                    () =>
                    {
                        var model = PredictionModel.ReadAsync<SentimentData, SentimentPrediction>(modelFile);
                        model.Wait();
                        return model.Result;
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
