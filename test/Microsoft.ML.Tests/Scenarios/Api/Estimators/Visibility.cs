// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Visibility: It should, possibly through the debugger, be not such a pain to actually
        /// see what is happening to your data when you apply this or that transform. For example, if I
        /// were to have the text "Help I'm a bug!" I should be able to see the steps where it is
        /// normalized to "help i'm a bug" then tokenized into ["help", "i'm", "a", "bug"] then
        /// mapped into term numbers [203, 25, 3, 511] then projected into the sparse
        /// float vector {3:1, 25:1, 203:1, 511:1}, etc. etc.
        /// </summary>
        [Fact]
        void Visibility()
        {
            var ml = new MLContext(seed: 1);
            var pipeline = ml.Data.CreateTextLoader(TestDatasets.Sentiment.GetLoaderColumns(), hasHeader: true)
                .Append(ml.Transforms.Text.FeaturizeText(
                    "Features", new Transforms.Text.TextFeaturizingEstimator.Options { OutputTokens = true }, "SentimentText"));

            var src = new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename));
            var data = pipeline.Fit(src).Load(src);

            var textColumn = data.GetColumn<string>(data.Schema["SentimentText"]).Take(20);
            var transformedTextColumn = data.GetColumn<string[]>(data.Schema["Features_TransformedText"]).Take(20);
            var features = data.GetColumn<float[]>(data.Schema["Features"]).Take(20);
        }
    }
}
