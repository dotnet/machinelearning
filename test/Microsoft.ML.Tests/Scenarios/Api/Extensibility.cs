using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
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
            var dataPath = GetDataPath(IrisDataPath);
            using (var env = new TlcEnvironment())
            {
                var loader = TextLoader.ReadFile(env, MakeIrisTextLoaderArgs(), new MultiFileSource(dataPath));
                Action<IrisData, IrisData> action = (i, j) =>
                {
                    j.Label = i.Label;
                    j.PetalLength = i.SepalLength > 3 ? i.PetalLength : i.SepalLength;
                    j.PetalWidth = i.PetalWidth;
                    j.SepalLength = i.SepalLength;
                    j.SepalWidth = i.SepalWidth;
                };
                var lambda = LambdaTransform.CreateMap(env, loader, action);
                var term = new TermTransform(env, lambda, "Label").Transform(lambda);
                var concat = new ConcatTransform(env, term, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");

                var trainer = new SdcaMultiClassTrainer(env, new SdcaMultiClassTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 });

                IDataView trainData = trainer.Info.WantCaching ? (IDataView)new CacheDataView(env, concat, prefetch: null) : concat;
                var trainRoles = new RoleMappedData(trainData, label: "Label", feature: "Features");

                // Auto-normalization.
                NormalizeTransform.CreateIfNeeded(env, ref trainRoles, trainer);
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(concat, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);

                var keyToValue = new KeyToValueTransform(env, scorer, "PredictedLabel");
                var model = env.CreatePredictionEngine<IrisData, IrisPrediction>(keyToValue);

                var testLoader = TextLoader.ReadFile(env, MakeIrisTextLoaderArgs(), new MultiFileSource(dataPath));
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
