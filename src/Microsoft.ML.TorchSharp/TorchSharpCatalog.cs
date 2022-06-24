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
        /// Fine tune a NAS-BERT model for NLP classification. The limit for any sentence is 512 tokens. Each word typically
        /// will map to a single token, and we automatically add 2 specical tokens (a start token and a separator token)
        /// so in general this limit will be 510 words for all sentences.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="labelColumnName">Name of the label column. Column should be a key type.</param>
        /// <param name="scoreColumnName">Name of the score column.</param>
        /// <param name="outputColumnName">Name of the output column. It will be a key type. It is the predicted label.</param>
        /// <param name="sentence1ColumnName">Name of the column for the first sentence.</param>
        /// <param name="sentence2ColumnName">Name of the column for the second sentence. Only required if your NLP classification requires sentence pairs.</param>
        /// <param name="batchSize">Number of rows in the batch.</param>
        /// <param name="maxEpochs">Maximum number of times to loop through your training set.</param>
        /// <param name="architecture">Architecture for the model. Defaults to Roberta.</param>
        /// <param name="validationSet">The validation set used while training to improve model quality.</param>
        /// <returns></returns>
        public static TextClassificationTrainer TextClassification(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string scoreColumnName = DefaultColumnNames.Score,
            string outputColumnName = DefaultColumnNames.PredictedLabel,
            string sentence1ColumnName = "Sentence1",
            string sentence2ColumnName = default,
            int batchSize = 32,
            int maxEpochs = 10,
            BertArchitecture architecture = BertArchitecture.Roberta,
            IDataView validationSet = null)
            => new TextClassificationTrainer(CatalogUtils.GetEnvironment(catalog), labelColumnName, outputColumnName, scoreColumnName, sentence1ColumnName, sentence2ColumnName, batchSize, maxEpochs, validationSet, architecture);
    }
}
