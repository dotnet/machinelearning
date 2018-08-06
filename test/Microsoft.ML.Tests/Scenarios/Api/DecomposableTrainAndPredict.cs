using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{

    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Decomposable train and predict: Train on Iris multiclass problem, which will require
        /// a transform on labels. Be able to reconstitute the pipeline for a prediction only task,
        /// which will essentially "drop" the transform over labels, while retaining the property
        /// that the predicted label for this has a key-type, the probability outputs for the classes
        /// have the class labels as slot names, etc. This should be do-able without ugly compromises like,
        /// say, injecting a dummy label.
        /// </summary>
        [Fact]
        void DecomposableTrainAndPredict()
        {
            var dataPath = GetDataPath(IrisDataPath);
            using (var env = new TlcEnvironment())
            {
                var loader = new TextLoader(env, MakeIrisTextLoaderArgs(), new MultiFileSource(dataPath));
                var term = new TermTransform(env, loader, "Label");
                var concat = new ConcatTransform(env, term, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");
                var trainer = new SdcaMultiClassTrainer(env, new SdcaMultiClassTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 });

                IDataView trainData = trainer.Info.WantCaching ? (IDataView)new CacheDataView(env, concat, prefetch: null) : concat;
                var trainRoles = new RoleMappedData(trainData, label: "Label", feature: "Features");

                // Auto-normalization.
                NormalizeTransform.CreateIfNeeded(env, ref trainRoles, trainer);
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(concat, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);
                
                // Cut of term transform from pipeline.
                var new_scorer = ApplyTransformUtils.ApplyAllTransformsToData(env, scorer, loader, term);
                var keyToValue = new KeyToValueTransform(env, new_scorer, "PredictedLabel");
                var model = env.CreatePredictionEngine<IrisData, IrisPrediction>(keyToValue);

                var testLoader = new TextLoader(env, MakeIrisTextLoaderArgs(), new MultiFileSource(dataPath));
                var testData = testLoader.AsEnumerable<IrisData>(env, false);
                foreach (var input in testData.Take(20))
                {
                    var prediction = model.Predict(input);
                    Assert.True(prediction.PredictedLabel == input.Label);
                }
            }
        }
    }
}
