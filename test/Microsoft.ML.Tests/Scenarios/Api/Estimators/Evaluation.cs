// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
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
            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                var reader = new TextLoader(env, MakeSentimentTextLoaderArgs());

                // Pipeline.
                var pipeline = new TextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new TextTransform(env, "SentimentText", "Features"))
                    .Append(new LinearClassificationTrainer(env, "Features", "Label", advancedSettings: (s) => s.NumThreads = 1));

                // Train.
                var readerModel = pipeline.Fit(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

                // Evaluate on the test set.
                var dataEval = readerModel.Read(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.validFilename)));
                var evaluator = new MyBinaryClassifierEvaluator(env, new BinaryClassifierEvaluator.Arguments() { });
                var metrics = evaluator.Evaluate(dataEval);
            }
        }
    }
}
