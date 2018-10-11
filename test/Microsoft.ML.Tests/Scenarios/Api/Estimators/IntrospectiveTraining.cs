// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.TextAnalytics;
using System.Collections.Generic;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{

    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Introspective training: Models that produce outputs and are otherwise black boxes are of limited use;
        /// it is also necessary often to understand at least to some degree what was learnt. To outline critical
        /// scenarios that have come up multiple times:
        ///  *) When I train a linear model, I should be able to inspect coefficients.
        ///  *) The tree ensemble learners, I should be able to inspect the trees.
        ///  *) The LDA transform, I should be able to inspect the topics.
        ///  I view it as essential from a usability perspective that this be discoverable to someone without 
        ///  having to read documentation. For example, if I have var lda = new LdaTransform().Fit(data)(I don't insist on that
        ///  exact signature, just giving the idea), then if I were to type lda.
        ///  In Visual Studio, one of the auto-complete targets should be something like GetTopics.
        /// </summary>

        [Fact]
        public void New_IntrospectiveTraining()
        {
            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                var data = new TextLoader(env, MakeSentimentTextLoaderArgs())
                    .Read(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

                var pipeline = new TextTransform(env, "SentimentText", "Features")
                    .Append(new StochasticDualCoordinateAscent(env, new StochasticDualCoordinateAscent.Arguments { NumThreads = 1 }, "Features", "Label"));

                // Train.
                var model = pipeline.Fit(data);

                // Get feature weights.
                VBuffer<float> weights = default;
                model.LastTransformer.Model.GetFeatureWeights(ref weights);
            }
        }
    }
}
