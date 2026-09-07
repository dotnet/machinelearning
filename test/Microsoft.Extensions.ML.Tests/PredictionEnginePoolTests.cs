// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Extensions.ML
{
    public class PredictionEnginePoolTests : BaseTestClass
    {
        public PredictionEnginePoolTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void can_load_namedmodel()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();

            services.AddPredictionEnginePool<SentimentData, SentimentPrediction>()
               .FromFile(modelName: "model1", filePath: Path.Combine("TestModels", "SentimentModel.zip"), watchForChanges: false);

            var sp = services.BuildServiceProvider();

            var pool = sp.GetRequiredService<PredictionEnginePool<SentimentData, SentimentPrediction>>();
            var model = pool.GetModel("model1");

            Assert.NotNull(model);
        }

        [Fact]
        public void pool_serves_predictions_across_a_hot_swap()
        {
            var loader = CreateReloadableLoader();
            using var pool = CreatePool(loader);

            var inFlight = pool.GetPredictionEngine();
            Assert.NotNull(inFlight.Predict(new SentimentData { SentimentText = "great" }));

            loader.Reload();

            var afterSwap = pool.GetPredictionEngine();
            Assert.NotNull(afterSwap.Predict(new SentimentData { SentimentText = "terrible" }));
            pool.ReturnPredictionEngine(afterSwap);

            pool.ReturnPredictionEngine(inFlight);

            var reused = pool.GetPredictionEngine();
            Assert.NotSame(inFlight, reused);
            Assert.NotNull(reused.Predict(new SentimentData { SentimentText = "fine" }));
            pool.ReturnPredictionEngine(reused);
        }

        [Fact]
        public void disposing_pool_releases_loader_resources()
        {
            var loader = CreateReloadableLoader();
            var pool = CreatePool(loader);

            _ = pool.GetPredictionEngine();
            pool.Dispose();

            loader.Reload();
            pool.Dispose();
        }

        [Fact]
        public void pooled_engines_do_not_dispose_the_shared_model()
        {
            var context = new MLContext(seed: 1);
            using var stream = File.OpenRead(Path.Combine("TestModels", "SentimentModel.zip"));
            var innerModel = context.Model.Load(stream, out _);
            var model = new DisposeCountingTransformer(innerModel);
            var loader = new ReloadableModelLoader(model);
            using var pool = CreatePool(loader);

            var maximumRetained = Environment.ProcessorCount * 2;
            var rented = new PredictionEngine<SentimentData, SentimentPrediction>[maximumRetained + 2];
            for (var i = 0; i < rented.Length; i++)
            {
                rented[i] = pool.GetPredictionEngine();
            }

            foreach (var engine in rented)
            {
                pool.ReturnPredictionEngine(engine);
            }

            Assert.Equal(0, model.DisposeCount);

            var afterOverflow = pool.GetPredictionEngine();
            Assert.NotNull(afterOverflow.Predict(new SentimentData { SentimentText = "still works" }));
            pool.ReturnPredictionEngine(afterOverflow);
        }

        private static ReloadableModelLoader CreateReloadableLoader()
        {
            var context = new MLContext(seed: 1);
            using var stream = File.OpenRead(Path.Combine("TestModels", "SentimentModel.zip"));
            var model = context.Model.Load(stream, out _);
            return new ReloadableModelLoader(model);
        }

        private static PredictionEnginePool<SentimentData, SentimentPrediction> CreatePool(ModelLoader loader)
        {
            var services = new ServiceCollection().AddOptions().AddLogging();
            services.AddPredictionEnginePool<SentimentData, SentimentPrediction>();
            services.Configure<PredictionEnginePoolOptions<SentimentData, SentimentPrediction>>(
                string.Empty, o => o.ModelLoader = loader);

            var sp = services.BuildServiceProvider();
            return sp.GetRequiredService<PredictionEnginePool<SentimentData, SentimentPrediction>>();
        }

        private sealed class ReloadableModelLoader : ModelLoader
        {
            private readonly ITransformer _model;
            private ModelReloadToken _token = new ModelReloadToken();

            public ReloadableModelLoader(ITransformer model) => _model = model;

            public override IChangeToken GetReloadToken() => _token;

            public override ITransformer GetModel() => _model;

            public void Reload()
            {
                var previous = Interlocked.Exchange(ref _token, new ModelReloadToken());
                previous.OnReload();
            }
        }

        private sealed class DisposeCountingTransformer : ITransformer, IDisposable
        {
            private readonly ITransformer _inner;
            private int _disposeCount;

            public DisposeCountingTransformer(ITransformer inner) => _inner = inner;

            public int DisposeCount => _disposeCount;

            public bool IsRowToRowMapper => _inner.IsRowToRowMapper;

            public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => _inner.GetOutputSchema(inputSchema);

            public IDataView Transform(IDataView input) => _inner.Transform(input);

            public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => _inner.GetRowToRowMapper(inputSchema);

            public void Save(ModelSaveContext ctx) => _inner.Save(ctx);

            public void Dispose()
            {
                Interlocked.Increment(ref _disposeCount);
                (_inner as IDisposable)?.Dispose();
            }
        }

        public class SentimentData
        {
            [ColumnName("Label"), LoadColumn(0)]
            public bool Sentiment;

            [LoadColumn(1)]
            public string SentimentText;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Sentiment;

            public float Score;
        }
    }
}
