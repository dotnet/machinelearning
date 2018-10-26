using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using System;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Extensibility: We can't possibly write every conceivable transform and should not try.
        /// It should somehow be possible for a user to inject custom code to, say, transform data.
        /// This might have a much steeper learning curve than the other usages (which merely involve
        /// usage of already established components), but should still be possible.
        /// </summary>
        [Fact]
        void Extensibility()
        {
            using (var env = new LocalEnvironment()
                .AddStandardComponents()) // ScoreUtils.GetScorer requires scorers to be registered in the ComponentCatalog
            {
                var loader = TextLoader.ReadFile(env, MakeIrisTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.irisData.trainFilename)));
                Action<IrisData, IrisData> action = (i, j) =>
                {
                    j.Label = i.Label;
                    j.PetalLength = i.SepalLength > 3 ? i.PetalLength : i.SepalLength;
                    j.PetalWidth = i.PetalWidth;
                    j.SepalLength = i.SepalLength;
                    j.SepalWidth = i.SepalWidth;
                };
                var lambda = LambdaTransform.CreateMap(env, loader, action);
                var term = TermTransform.Create(env, lambda, "Label");
                var concat = new ConcatTransform(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Transform(term);

                var trainer = new SdcaMultiClassTrainer(env, "Features", "Label", advancedSettings: (s) => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; });

                IDataView trainData = trainer.Info.WantCaching ? (IDataView)new CacheDataView(env, concat, prefetch: null) : concat;
                var trainRoles = new RoleMappedData(trainData, label: "Label", feature: "Features");

                // Auto-normalization.
                NormalizeTransform.CreateIfNeeded(env, ref trainRoles, trainer);
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(concat, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);

                var keyToValue = new KeyToValueTransform(env, "PredictedLabel").Transform(scorer);
                var model = env.CreatePredictionEngine<IrisData, IrisPrediction>(keyToValue);

                var testLoader = TextLoader.ReadFile(env, MakeIrisTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.irisData.trainFilename)));
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
