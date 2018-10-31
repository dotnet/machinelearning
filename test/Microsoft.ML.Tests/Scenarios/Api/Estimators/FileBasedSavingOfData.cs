// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using System.IO;
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

            var ml = new MLContext(seed: 1, conc: 1);
            var src = new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename));
            var trainData = ml.Data.TextReader(MakeSentimentTextLoaderArgs())
                .Append(ml.Transforms.Text.FeaturizeText("SentimentText", "Features"))
                .Fit(src).Read(src);

            using (var file = File.Create(DeleteOutputPath("i.idv")))
            {
                var saver = new BinarySaver(ml, new BinarySaver.Arguments());
                using (var ch = ((IHostEnvironment)ml).Start("SaveData"))
                    DataSaverUtils.SaveDataView(ch, saver, trainData, file);
            }

            var trainer = ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: s => s.NumThreads = 1);
            var loadedTrainData = new BinaryLoader(ml, new BinaryLoader.Arguments(), new MultiFileSource("i.idv"));

            // Train.
            var model = trainer.Fit(loadedTrainData);
            DeleteOutputPath("i.idv");
        }
    }
}
