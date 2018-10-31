// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Evaluation: Similar to the simple train scenario, except instead of having some 
        /// predictive structure, be able to score another "test" data file, run the result 
        /// through an evaluator and get metrics like AUC, accuracy, PR curves, and whatnot. 
        /// Getting metrics out of this shoudl be as straightforward and unannoying as possible.
        /// </summary>
        [Fact]
        public void New_Evaluation()
        {
            var ml = new MLContext(seed: 1, conc: 1);

            // Pipeline.
            var pipeline = ml.Data.TextReader(MakeSentimentTextLoaderArgs())
                .Append(ml.Transforms.Text.FeaturizeText("SentimentText", "Features"))
                .Append(ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: s => s.NumThreads = 1));

            // Train.
            var readerModel = pipeline.Fit(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

            // Evaluate on the test set.
            var dataEval = readerModel.Read(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.testFilename)));
            var metrics = ml.BinaryClassification.Evaluate(dataEval);
        }
    }
}
