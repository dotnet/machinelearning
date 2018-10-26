// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Trainers;
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
            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));
                
                var trans = TextFeaturizingEstimator.Create(env, MakeSentimentTextTransformArgs(), loader);
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
    }
}
