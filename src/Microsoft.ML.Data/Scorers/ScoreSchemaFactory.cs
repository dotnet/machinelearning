// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class contains method for creating commonly used <see cref="Schema"/>s.
    /// </summary>
    [BestFriend]
    internal static class ScoreSchemaFactory
    {
        /// <summary>
        /// Return a <see cref="Schema"/> which contains a single score column.
        /// </summary>
        /// <param name="scoreType">The type of the score column.</param>
        /// <param name="scoreColumnKindValue">The kind of the score column. It's the value of <see cref="MetadataUtils.Kinds.ScoreColumnKind"/> in the score column's metadata.</param>
        /// <param name="scoreColumnName">The score column's name in the generated <see cref="Schema"/>.</param>
        /// <returns><see cref="Schema"/> which contains only one column.</returns>
        public static Schema Create(ColumnType scoreType, string scoreColumnKindValue, string scoreColumnName = MetadataUtils.Const.ScoreValueKind.Score)
        {
            Contracts.CheckValue(scoreType, nameof(scoreType));
            Contracts.CheckNonEmpty(scoreColumnKindValue, nameof(scoreColumnKindValue));

            // Two metadata fields. One can set up by caller of this function while the other one is a constant.
            var metadataBuilder = new MetadataBuilder();
            metadataBuilder.Add(MetadataUtils.Kinds.ScoreColumnKind, TextType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = scoreColumnKindValue.AsMemory(); });
            metadataBuilder.Add(MetadataUtils.Kinds.ScoreValueKind, TextType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = MetadataUtils.Const.ScoreValueKind.Score.AsMemory(); });

            // Build a schema consisting of a single column.
            var schemaBuilder = new SchemaBuilder();
            schemaBuilder.AddColumn(scoreColumnName, scoreType, metadataBuilder.GetMetadata());

            return schemaBuilder.GetSchema();
        }

        /// <summary>
        /// Create a <see cref="Schema"/> with two columns for binary classifier. The first column, indexed by 0, is the score column.
        /// The second column is the probability column. For example, for linear support vector machine, score column stands for the inner product
        /// of linear coefficients and the input feature vector and we convert score column to probability column using a calibrator.
        /// </summary>
        /// <param name="scoreColumnName">Column name of score column</param>
        /// <param name="probabilityColumnName">Column name of probability column</param>
        /// <returns><see cref="Schema"/> of binary classifier's output.</returns>
        public static Schema CreateBinaryClassificationSchema(string scoreColumnName = MetadataUtils.Const.ScoreValueKind.Score,
            string probabilityColumnName = MetadataUtils.Const.ScoreValueKind.Probability)
        {
            // Schema of Score column. We are going to extend it by adding a Probability column.
            var partialSchema = Create(NumberType.Float, MetadataUtils.Const.ScoreColumnKind.BinaryClassification, scoreColumnName);

            var schemaBuilder = new SchemaBuilder();
            // Copy Score column from partialSchema.
            schemaBuilder.AddColumn(partialSchema[0].Name, partialSchema[0].Type, partialSchema[0].Metadata);

            // Create Probability column's metadata.
            var probabilityMetadataBuilder = new MetadataBuilder();
            probabilityMetadataBuilder.Add(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, (ref bool value) => { value = true; });
            probabilityMetadataBuilder.Add(MetadataUtils.Kinds.ScoreColumnKind, TextType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = MetadataUtils.Const.ScoreColumnKind.BinaryClassification.AsMemory(); });
            probabilityMetadataBuilder.Add(MetadataUtils.Kinds.ScoreValueKind, TextType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = MetadataUtils.Const.ScoreValueKind.Probability.AsMemory(); });

            // Add probability column.
            schemaBuilder.AddColumn(probabilityColumnName, NumberType.Float, probabilityMetadataBuilder.GetMetadata());

            return schemaBuilder.GetSchema();
        }

        /// <summary>
        /// This is very similar to <see cref="Create(ColumnType, string, string)"/> but adds one extra metadata field to the only score column.
        /// </summary>
        /// <param name="scoreType">Output element's type of quantile regressor. Note that a quantile regressor can produce an array of <see cref="PrimitiveType"/>.</param>
        /// <param name="quantiles">Quantiles used in quantile regressor.</param>
        /// <returns><see cref="Schema"/> of quantile regressor's output.</returns>
        public static Schema CreateQuantileRegressionSchema(ColumnType scoreType, double[] quantiles)
        {
            Contracts.CheckValue(scoreType, nameof(scoreType));
            Contracts.CheckValue(scoreType as PrimitiveType, nameof(scoreType));
            Contracts.AssertValue(quantiles);

            // Create a schema using standard function. The produced schema will be modified by adding one metadata column.
            var partialSchema = Create(new VectorType(scoreType as PrimitiveType, quantiles.Length), MetadataUtils.Const.ScoreColumnKind.QuantileRegression);

            var metadataBuilder = new MetadataBuilder();
            // Add the extra metadata.
            metadataBuilder.AddSlotNames(quantiles.Length, (ref VBuffer<ReadOnlyMemory<char>> value) =>
                {
                    var bufferEditor = VBufferEditor.Create(ref value, quantiles.Length);
                    for (int i = 0; i < quantiles.Length; ++i)
                        bufferEditor.Values[i] = string.Format("Quantile-{0}", quantiles[i]).AsMemory();
                    value = bufferEditor.Commit();
                });
            // Copy default metadata from the partial schema.
            metadataBuilder.Add(partialSchema[0].Metadata, (string kind) => true);

            // Build a schema consisting of a single column. Comparing with partial schema, the only difference is a metadata field.
            var schemaBuilder = new SchemaBuilder();
            schemaBuilder.AddColumn(partialSchema[0].Name, partialSchema[0].Type, metadataBuilder.GetMetadata());

            return schemaBuilder.GetSchema();
        }

        /// <summary>
        /// This function returns a schema for sequence predictor's output. Its output column is always called <see cref="MetadataUtils.Const.ScoreValueKind.PredictedLabel"/>.
        /// </summary>
        /// <param name="scoreType">Score column's type produced by sequence predictor.</param>
        /// <param name="scoreColumnKindValue">A metadata value of score column. It's the value associated with key
        /// <see cref="MetadataUtils.Kinds.ScoreColumnKind"/>.</param>
        /// <param name="keyNames">Sequence predictor usually generates integer outputs. This field tells the tags of all possible output values.
        /// For example, output integer 0 cound be mapped to "Sell" and 0 to "Buy" when predicting stock trend.</param>
        /// <returns><see cref="Schema"/> of sequence predictor's output.</returns>
        public static Schema CreateSequencePredictionSchema(ColumnType scoreType, string scoreColumnKindValue, VBuffer<ReadOnlyMemory<char>> keyNames=default)
        {
            Contracts.CheckValue(scoreType, nameof(scoreType));
            Contracts.CheckValue(scoreColumnKindValue, nameof(scoreColumnKindValue));

            var metadataBuilder = new MetadataBuilder();
            // Add metadata columns including their getters. We starts with key names of predicted keys if they exist.
            if (keyNames.Length > 0)
                metadataBuilder.AddKeyValues(keyNames.Length, TextType.Instance,
                    (ref VBuffer<ReadOnlyMemory<char>> value) => value = keyNames);
            metadataBuilder.Add(MetadataUtils.Kinds.ScoreColumnKind, TextType.Instance,
                (ref ReadOnlyMemory<char> value) => value = scoreColumnKindValue.AsMemory());
            metadataBuilder.Add(MetadataUtils.Kinds.ScoreValueKind, TextType.Instance,
                (ref ReadOnlyMemory<char> value) => value = MetadataUtils.Const.ScoreValueKind.PredictedLabel.AsMemory());

            // Build a schema consisting of a single column.
            var schemaBuilder = new SchemaBuilder();
            schemaBuilder.AddColumn(MetadataUtils.Const.ScoreValueKind.PredictedLabel, scoreType, metadataBuilder.GetMetadata());

            return schemaBuilder.GetSchema();
        }
    }
}
