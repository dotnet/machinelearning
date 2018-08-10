using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Learners;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// File-based saving of data: Come up with transform pipeline. Transform training and
        /// test data, and save the featurized data to some file, using the .idv format.
        /// Train and evaluate multiple models over that pre-featurized data. (Useful for
        /// sweeping scenarios, where you are training many times on the same data,
        /// and don't necessarily want to transform it every single time.)
        /// </summary>
        [Fact]
        void FileBasedSavingOfData()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);
            
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));
                
                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(), loader);
                var saver = new BinarySaver(env, new BinarySaver.Arguments());
                using (var ch = env.Start("SaveData"))
                using (var file = env.CreateOutputFile("i.idv"))
                {
                    DataSaverUtils.SaveDataView(ch, saver, trans, file);
                }

                var binData = new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource("i.idv"));
                var trainRoles = new RoleMappedData(binData, label: "Label", feature: "Features");
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                });
                var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));
                
                DeleteOutputPath("i.idv");
            }
        }

        /// <summary>
        /// File-based saving of data: Come up with transform pipeline. Transform training and
        /// test data, and save the featurized data to some file, using the .idv format.
        /// Train and evaluate multiple models over that pre-featurized data. (Useful for
        /// sweeping scenarios, where you are training many times on the same data,
        /// and don't necessarily want to transform it every single time.)
        /// </summary>
        [Fact]
        void New_FileBasedSavingOfData()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var pipeline = new MyTextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new MyTextTransform(env, MakeSentimentTextTransformArgs()));

                var trainData = pipeline.Fit(new MultiFileSource(dataPath)).Read(new MultiFileSource(dataPath));

                using (var file = env.CreateOutputFile("i.idv"))
                    trainData.SaveAsBinary(env, file.CreateWriteStream());

                var trainer = new MySdca(env, new LinearClassificationTrainer.Arguments { NumThreads = 1 }, "Features", "Label");
                var loadedTrainData = new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource("i.idv"));

                // Train.
                var model = trainer.Train(loadedTrainData);
                DeleteOutputPath("i.idv");
            }
        }
    }
}
