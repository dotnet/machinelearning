using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Learners;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Meta-components: Meta-components (e.g., components that themselves instantiate components) should not be booby-trapped.
        /// When specifying what trainer OVA should use, a user will be able to specify any binary classifier.
        /// If they specify a regression or multi-class classifier ideally that should be a compile error.
        /// </summary>
        [Fact]
        void Metacomponents()
        {
            var dataPath = GetDataPath(IrisDataPath);
            using (var env = new TlcEnvironment())
            {
                var loader = new TextLoader(env, MakeIrisTextLoaderArgs(), new MultiFileSource(dataPath));
                var term = new TermTransform(env, loader, "Label");
                var concat = new ConcatTransform(env, term, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");
                var trainer = new Ova(env, new Ova.Arguments
                {
                    PredictorType = ComponentFactoryUtils.CreateFromFunction(
                        (e) => new FastTreeBinaryClassificationTrainer(e, new FastTreeBinaryClassificationTrainer.Arguments()))
                });

                IDataView trainData = trainer.Info.WantCaching ? (IDataView)new CacheDataView(env, concat, prefetch: null) : concat;
                var trainRoles = new RoleMappedData(trainData, label: "Label", feature: "Features");

                // Auto-normalization.
                NormalizeTransform.CreateIfNeeded(env, ref trainRoles, trainer);
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(concat, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);

                var keyToValue = new KeyToValueTransform(env, scorer, "PredictedLabel");
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