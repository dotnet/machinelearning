// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp.NasBert;

namespace Microsoft.ML.TorchSharp
{
    /// <summary>
    /// Collection of extension methods for <see cref="T:Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers" /> to create instances of TorchSharp trainer components.
    /// </summary>
    /// <remarks>
    /// This requires additional nuget dependencies to link against TorchSharp native dlls. See <see cref="T:Microsoft.ML.Vision.ImageClassificationTrainer"/> for more information.
    /// </remarks>
    public static class TorchSharpCatalog
    {
        /// <summary>
        /// Fine tune a NAS-BERT model for NLP classification.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="labelColumnName">Name of the label column. Column should be of type Int64.</param>
        /// <param name="outputColumnName">Name of the output column. It will be of type Double. It is the predicted label.</param>
        /// <param name="sentence1ColumnName">Name of the column for the first sentence.</param>
        /// <param name="sentence2ColumnName">Name of the column for the second sentence. Only required if your NLP classification requires sentence pairs.</param>
        /// <param name="numberOfClasses">Number of classes to train on.</param>
        /// <param name="maxEpochs">Maximum number of times to loop through your training set.</param>
        /// <param name="maxUpdates">Maximum number of updated rows. Will stop training when this number is hit.</param>
        /// <returns></returns>
        public static NasBertEstimator NasBertSentenceClassification(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string outputColumnName = DefaultColumnNames.PredictedLabel,
            string sentence1ColumnName = "Sentence1",
            string sentence2ColumnName = default,
            int numberOfClasses = 2,
            int maxEpochs = 10,
            int maxUpdates = 2147483647) =>
            new NasBertEstimator(CatalogUtils.GetEnvironment(catalog), labelColumnName, outputColumnName, sentence1ColumnName, sentence2ColumnName, numberOfClasses, maxEpochs, maxUpdates);
    }
}
