// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
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
        void New_FileBasedSavingOfData()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                var trainData = new TextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new TextTransform(env, "SentimentText", "Features"))
                    .FitAndRead(new MultiFileSource(dataPath));

                using (var file = env.CreateOutputFile("i.idv"))
                    trainData.SaveAsBinary(env, file.CreateWriteStream());

                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments { NumThreads = 1 }, "Features", "Label");
                var loadedTrainData = new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource("i.idv"));

                // Train.
                var model = trainer.Train(new RoleMappedData(loadedTrainData, DefaultColumnNames.Label, DefaultColumnNames.Features));
                DeleteOutputPath("i.idv");
            }
        }
    }
}
