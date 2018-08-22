// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
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
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var pipeline = new MyTextLoader(env, MakeSentimentTextLoaderArgs())
                    .Append(new MyTextTransform(env, MakeSentimentTextTransformArgs()))
                    .Append(new MySdca(env, new LinearClassificationTrainer.Arguments { NumThreads = 1 }, "Features", "Label"));

                // Train.
                var model = pipeline.Fit(new MultiFileSource(dataPath));

                // Evaluate on the test set.
                var dataEval = model.Read(new MultiFileSource(testDataPath));
                var evaluator = new MyBinaryClassifierEvaluator(env, new BinaryClassifierEvaluator.Arguments() { });
                var metrics = evaluator.Evaluate(dataEval);
            }
        }
    }
}
