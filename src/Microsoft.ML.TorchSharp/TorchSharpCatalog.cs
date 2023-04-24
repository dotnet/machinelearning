// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp.AutoFormerV2;
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

        /// <summary>
        /// Fine tune a NAS-BERT model for NLP classification. The limit for any sentence is 512 tokens. Each word typically
        /// will map to a single token, and we automatically add 2 specical tokens (a start token and a separator token)
        /// so in general this limit will be 510 words for all sentences.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="options">Advanced Options.</param>
        /// <returns></returns>
        public static TextClassificationTrainer TextClassification(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            NasBertTrainer.NasBertOptions options)
            => new TextClassificationTrainer(CatalogUtils.GetEnvironment(catalog), options);

        /// <summary>
        /// Fine tune a NAS-BERT model for NLP sentence Similarity. The limit for any sentence is 512 tokens. Each word typically
        /// will map to a single token, and we automatically add 2 specical tokens (a start token and a separator token)
        /// so in general this limit will be 510 words for all sentences.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="labelColumnName">Name of the label column. Column should be a float type.</param>
        /// <param name="scoreColumnName">Name of the score column.</param>
        /// <param name="sentence1ColumnName">Name of the column for the first sentence.</param>
        /// <param name="sentence2ColumnName">Name of the column for the second sentence. Only required if your NLP classification requires sentence pairs.</param>
        /// <param name="batchSize">Number of rows in the batch.</param>
        /// <param name="maxEpochs">Maximum number of times to loop through your training set.</param>
        /// <param name="architecture">Architecture for the model. Defaults to Roberta.</param>
        /// <param name="validationSet">The validation set used while training to improve model quality.</param>
        /// <returns></returns>
        public static SentenceSimilarityTrainer SentenceSimilarity(
            this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string scoreColumnName = DefaultColumnNames.Score,
            string sentence1ColumnName = "Sentence1",
            string sentence2ColumnName = "Sentence2",
            int batchSize = 32,
            int maxEpochs = 10,
            BertArchitecture architecture = BertArchitecture.Roberta,
            IDataView validationSet = null)
            => new SentenceSimilarityTrainer(CatalogUtils.GetEnvironment(catalog), labelColumnName, scoreColumnName, sentence1ColumnName, sentence2ColumnName, batchSize, maxEpochs, validationSet, architecture);

        /// <summary>
        /// Fine tune a NAS-BERT model for NLP sentence Similarity. The limit for any sentence is 512 tokens. Each word typically
        /// will map to a single token, and we automatically add 2 specical tokens (a start token and a separator token)
        /// so in general this limit will be 510 words for all sentences.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="options">Advanced Options</param>
        /// <returns></returns>
        public static SentenceSimilarityTrainer SentenceSimilarity(
            this RegressionCatalog.RegressionTrainers catalog,
            NasBertTrainer.NasBertOptions options)
            => new SentenceSimilarityTrainer(CatalogUtils.GetEnvironment(catalog), options);


        /// <summary>
        /// Fine tune an object detection model.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="labelColumnName">The label column name. Should be a vector of keytype</param>
        /// <param name="predictedLabelColumnName">The output predicted label column name. Is a vector of keytype</param>
        /// <param name="scoreColumnName">The output score column name. Is a vector of float.</param>
        /// <param name="boundingBoxColumnName">The bounding box column name. Is a vector of float. Values should be in the order x0 y0 x1 y1.</param>
        /// <param name="predictedBoundingBoxColumnName">The output bounding box column name. Is a vector of float. Values should be in the order x0 y0 x1 y1.</param>
        /// <param name="imageColumnName">The column name holding the image Data. Is an MLImage</param>
        /// <param name="maxEpoch">How many epochs to run.</param>
        /// <returns></returns>
        public static ObjectDetectionTrainer ObjectDetection(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string predictedLabelColumnName = DefaultColumnNames.PredictedLabel,
            string scoreColumnName = DefaultColumnNames.Score,
            string boundingBoxColumnName = "BoundingBoxes",
            string predictedBoundingBoxColumnName = "PredictedBoundingBoxes",
            string imageColumnName = "Image",
            int maxEpoch = 10)
            => new ObjectDetectionTrainer(CatalogUtils.GetEnvironment(catalog), labelColumnName, predictedLabelColumnName, scoreColumnName, boundingBoxColumnName, predictedBoundingBoxColumnName, imageColumnName, maxEpoch);

        /// <summary>
        /// Fine tune an object detection model.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="options">The full set of advanced options.</param>
        /// <returns></returns>
        public static ObjectDetectionTrainer ObjectDetection(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            ObjectDetectionTrainer.Options options)
            => new ObjectDetectionTrainer(CatalogUtils.GetEnvironment(catalog), options);

        /// <summary>
        /// Evaluates scored object detection data.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="data">IDataView with the data</param>
        /// <param name="labelCol">Column that has the actual labels.</param>
        /// <param name="actualBoundingBoxColumn">Column that has the actual bounding boxes.</param>
        /// <param name="predictedLabelCol">Column that has the predicted labels.</param>
        /// <param name="predictedBoundingBoxColumn">Column that has the predicted bounding boxes.</param>
        /// <param name="scoreCol">Column that has the predicted score (confidence level).</param>
        /// <returns></returns>
        public static ObjectDetectionMetrics EvaluateObjectDetection(
            this MulticlassClassificationCatalog catalog,
            IDataView data,
            DataViewSchema.Column labelCol,
            DataViewSchema.Column actualBoundingBoxColumn,
            DataViewSchema.Column predictedLabelCol,
            DataViewSchema.Column predictedBoundingBoxColumn,
            DataViewSchema.Column scoreCol)
        {
            return ObjectDetectionMetrics.MeasureMetrics(data, labelCol, actualBoundingBoxColumn, predictedLabelCol, predictedBoundingBoxColumn, scoreCol);
        }
    }
}
