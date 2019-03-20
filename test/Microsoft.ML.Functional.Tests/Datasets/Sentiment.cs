// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class for reading in the Sentiment test dataset.
    /// </summary>
    internal sealed class TweetSentiment
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool Sentiment { get; set; }

        [LoadColumn(1)]
        public string SentimentText { get; set; }
    }
}
