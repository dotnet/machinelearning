using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Train, save/load model, predict: 
        /// Serve the scenario where training and prediction happen in different processes (or even different machines). 
        /// The actual test will not run in different processes, but will simulate the idea that the 
        /// "communication pipe" is just a serialized model of some form.
        /// </summary>
        [Fact]
        public void TrainSaveModelAndPredict()
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

                PredictionEngine<SentimentData, SentimentPrediction> model;
                using (var file = env.CreateTempFile())
                {
                    // Save model. 
                    var scoreRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                    using (var ch = env.Start("saving"))
                        TrainUtils.SaveModel(env, ch, file, predictor, scoreRoles);

                    // Load model.
                    using (var fs = file.OpenReadStream())
                        model = env.CreatePredictionEngine<SentimentData, SentimentPrediction>(fs);
                }

                // Take a couple examples out of the test data and run predictions on top.
                var testLoader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(SentimentTestPath)));
                var testData = testLoader.AsEnumerable<SentimentData>(env, false);
                foreach (var input in testData.Take(5))
                {
                    var prediction = model.Predict(input);
                    // Verify that predictions match and scores are separated from zero.
                    Assert.Equal(input.Sentiment, prediction.Sentiment);
                    Assert.True(input.Sentiment && prediction.Score > 1 || !input.Sentiment && prediction.Score < -1);
                }
            }

        }

        /// <summary>
        /// Train, save/load model, predict: 
        /// Serve the scenario where training and prediction happen in different processes (or even different machines). 
        /// The actual test will not run in different processes, but will simulate the idea that the 
        /// "communication pipe" is just a serialized model of some form.
        /// </summary>
        [Fact]
        public void New_TrainSaveModelAndPredict()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var pipeline = new MyTextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new MyTextTransform(env, MakeSentimentTextTransformArgs()))
                    .Append(new MySdca(env, new LinearClassificationTrainer.Arguments { NumThreads = 1 }, "Features", "Label"));

                // Train.
                var model = pipeline.Fit(new MultiFileSource(dataPath));

                CompositeReader<IMultiStreamSource, ITransformer> loadedModel;
                using (var file = env.CreateTempFile())
                {
                    // Save model. 
                    using (var fs = file.CreateWriteStream())
                        model.Save(env, fs);

                    // Load model.
                    loadedModel = CompositeReader.LoadModel(env, file.OpenReadStream());

                }

                // Create prediction engine and test predictions.
                var engine = new MyPredictionEngine<SentimentData, SentimentPrediction>(env, model.Reader.GetOutputSchema(), model.Transformer);

                // Take a couple examples out of the test data and run predictions on top.
                var testData = model.Reader.Read(new MultiFileSource(GetDataPath(SentimentTestPath)))
                    .AsEnumerable<SentimentData>(env, false);
                foreach (var input in testData.Take(5))
                {
                    var prediction = engine.Predict(input);
                    // Verify that predictions match and scores are separated from zero.
                    Assert.Equal(input.Sentiment, prediction.Sentiment);
                    Assert.True(input.Sentiment && prediction.Score > 1 || !input.Sentiment && prediction.Score < -1);
                }

            }
        }
    }
}
