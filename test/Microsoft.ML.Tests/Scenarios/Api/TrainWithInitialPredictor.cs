using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Learners;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Train with initial predictor: Similar to the simple train scenario, but also accept a pre-trained initial model.
        /// The scenario might be one of the online linear learners that can take advantage of this, e.g., averaged perceptron.
        /// </summary>
        [Fact]
        public void TrainWithInitialPredictor()
        {
            var dataPath = GetDataPath(SentimentDataPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(), loader);
                var trainData = trans;

                var cachedTrain = new CacheDataView(env, trainData, prefetch: null);
                // Train the first predictor.
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                });
                var trainRoles = new RoleMappedData(cachedTrain, label: "Label", feature: "Features");
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                // Train the second predictor on the same data.
                var secondTrainer = new AveragedPerceptronTrainer(env, new AveragedPerceptronTrainer.Arguments());
                var finalPredictor = secondTrainer.Train(new TrainContext(trainRoles, initialPredictor: predictor));
            }
        }

        /// <summary>
        /// Train with initial predictor: Similar to the simple train scenario, but also accept a pre-trained initial model.
        /// The scenario might be one of the online linear learners that can take advantage of this, e.g., averaged perceptron.
        /// </summary>
        [Fact]
        public void New_TrainWithInitialPredictor()
        {
            var dataPath = GetDataPath(SentimentDataPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var pipeline = new MyTextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new MyTextTransform(env, MakeSentimentTextTransformArgs()));

                // Train the pipeline, prepare train set.
                var reader = pipeline.Fit(new MultiFileSource(dataPath));
                var trainData = reader.Read(new MultiFileSource(dataPath));


                // Train the first predictor.
                var trainer = new MySdca(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                }, "Features", "Label");
                var firstPredictor = trainer.Fit(trainData);

                // Train the second predictor on the same data.
                var secondTrainer = new MyAveragedPerceptron(env, new AveragedPerceptronTrainer.Arguments(), "Features", "Label");
                var finalPredictor = secondTrainer.Train(trainData, firstPredictor.Model);
            }
        }

    }
}
