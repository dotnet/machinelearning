using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model.Onnx;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Exporting models: Models when defined ought to be exportable, e.g., to ONNX, PFA, text, etc.
        /// </summary>
        [Fact]
        void SaveAsOnnx()
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
                DeleteOutputPath("model.zip");
                using (var file = env.CreateOutputFile("model.zip"))
                    TrainUtils.SaveModel(env, env.Start("saveChannel"), file, predictor, trainRoles);
                new SaveOnnxCommand(env, new SaveOnnxCommand.Arguments { InputModelFile = "model.zip",
                    Json = "model.json", Onnx = "model.onnx" , Domain="dunno" }).Run();
            }
        }
    }
}
