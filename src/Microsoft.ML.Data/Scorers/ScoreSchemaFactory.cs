// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class contains method for creating commonly used <see cref="DataViewSchema"/>s.
    /// </summary>
    [BestFriend]
    internal static class ScoreSchemaFactory
    {
        /// <summary>
        /// Return a <see cref="DataViewSchema"/> which contains a single score column.
        /// </summary>
        /// <param name="scoreType">The type of the score column.</param>
        /// <param name="scoreColumnKindValue">The kind of the score column. It's the value of <see cref="AnnotationUtils.Kinds.ScoreColumnKind"/> in the score column's metadata.</param>
        /// <param name="scoreColumnName">The score column's name in the generated <see cref="DataViewSchema"/>.</param>
        /// <returns><see cref="DataViewSchema"/> which contains only one column.</returns>
        public static DataViewSchema Create(DataViewType scoreType, string scoreColumnKindValue, string scoreColumnName = AnnotationUtils.Const.ScoreValueKind.Score)
        {
            Contracts.CheckValue(scoreType, nameof(scoreType));
            Contracts.CheckNonEmpty(scoreColumnKindValue, nameof(scoreColumnKindValue));

            // Two metadata fields. One can set up by caller of this function while the other one is a constant.
            var metadataBuilder = new DataViewSchema.Annotations.Builder();
            metadataBuilder.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = scoreColumnKindValue.AsMemory(); });
            metadataBuilder.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });

            // Build a schema consisting of a single column.
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(scoreColumnName, scoreType, metadataBuilder.ToAnnotations());

            return schemaBuilder.ToSchema();
        }

        /// <summary>
        /// Create a <see cref="DataViewSchema"/> with two columns for binary classifier. The first column, indexed by 0, is the score column.
        /// The second column is the probability column. For example, for linear support vector machine, score column stands for the inner product
        /// of linear coefficients and the input feature vector and we convert score column to probability column using a calibrator.
        /// </summary>
        /// <param name="scoreColumnName">Column name of score column</param>
        /// <param name="probabilityColumnName">Column name of probability column</param>
        /// <returns><see cref="DataViewSchema"/> of binary classifier's output.</returns>
        public static DataViewSchema CreateBinaryClassificationSchema(string scoreColumnName = AnnotationUtils.Const.ScoreValueKind.Score,
            string probabilityColumnName = AnnotationUtils.Const.ScoreValueKind.Probability)
        {
            // Schema of Score column. We are going to extend it by adding a Probability column.
            var partialSchema = Create(NumberDataViewType.Single, AnnotationUtils.Const.ScoreColumnKind.BinaryClassification, scoreColumnName);

            var schemaBuilder = new DataViewSchema.Builder();
            // Copy Score column from partialSchema.
            schemaBuilder.AddColumn(partialSchema[0].Name, partialSchema[0].Type, partialSchema[0].Annotations);

            // Create Probability column's metadata.
            var probabilityMetadataBuilder = new DataViewSchema.Annotations.Builder();
            probabilityMetadataBuilder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, (ref bool value) => { value = true; });
            probabilityMetadataBuilder.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.BinaryClassification.AsMemory(); });
            probabilityMetadataBuilder.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance,
                (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Probability.AsMemory(); });

            // Add probability column.
            schemaBuilder.AddColumn(probabilityColumnName, NumberDataViewType.Single, probabilityMetadataBuilder.ToAnnotations());

            return schemaBuilder.ToSchema();
        }

        /// <summary>
        /// This is very similar to <see cref="Create(DataViewType, string, string)"/> but adds one extra metadata field to the only score column.
        /// </summary>
        /// <param name="scoreType">Output element's type of quantile regressor. Note that a quantile regressor can produce an array of <see cref="PrimitiveDataViewType"/>.</param>
        /// <param name="quantiles">Quantiles used in quantile regressor.</param>
        /// <returns><see cref="DataViewSchema"/> of quantile regressor's output.</returns>
        public static DataViewSchema CreateQuantileRegressionSchema(DataViewType scoreType, double[] quantiles)
        {
            Contracts.CheckValue(scoreType, nameof(scoreType));
            Contracts.CheckValue(scoreType as PrimitiveDataViewType, nameof(scoreType));
            Contracts.AssertValue(quantiles);

            // Create a schema using standard function. The produced schema will be modified by adding one metadata column.
            var partialSchema = Create(new VectorType(scoreType as PrimitiveDataViewType, quantiles.Length), AnnotationUtils.Const.ScoreColumnKind.QuantileRegression);

            var metadataBuilder = new DataViewSchema.Annotations.Builder();
            // Add the extra metadata.
            metadataBuilder.AddSlotNames(quantiles.Length, (ref VBuffer<ReadOnlyMemory<char>> value) =>
                {
                    var bufferEditor = VBufferEditor.Create(ref value, quantiles.Length);
                    for (int i = 0; i < quantiles.Length; ++i)
                        bufferEditor.Values[i] = string.Format("Quantile-{0}", quantiles[i]).AsMemory();
                    value = bufferEditor.Commit();
                });
            // Copy default metadata from the partial schema.
            metadataBuilder.Add(partialSchema[0].Annotations, (string kind) => true);

            // Build a schema consisting of a single column. Comparing with partial schema, the only difference is a metadata field.
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(partialSchema[0].Name, partialSchema[0].Type, metadataBuilder.ToAnnotations());

            return schemaBuilder.ToSchema();
        }

        /// <summary>
        /// This function returns a schema for sequence predictor's output. Its output column is always called <see cref="AnnotationUtils.Const.ScoreValueKind.PredictedLabel"/>.
        /// </summary>
        /// <param name="scoreType">Score column's type produced by sequence predictor.</param>
        /// <param name="scoreColumnKindValue">A metadata value of score column. It's the value associated with key
        /// <see cref="AnnotationUtils.Kinds.ScoreColumnKind"/>.</param>
        /// <param name="keyNames">Sequence predictor usually generates integer outputs. This field tells the tags of all possible output values.
        /// For example, output integer 0 cound be mapped to "Sell" and 0 to "Buy" when predicting stock trend.</param>
        /// <returns><see cref="DataViewSchema"/> of sequence predictor's output.</returns>
        public static DataViewSchema CreateSequencePredictionSchema(DataViewType scoreType, string scoreColumnKindValue, VBuffer<ReadOnlyMemory<char>> keyNames=default)
        {
            Contracts.CheckValue(scoreType, nameof(scoreType));
            Contracts.CheckValue(scoreColumnKindValue, nameof(scoreColumnKindValue));

            var metadataBuilder = new DataViewSchema.Annotations.Builder();
            // Add metadata columns including their getters. We starts with key names of predicted keys if they exist.
            if (keyNames.Length > 0)
                metadataBuilder.AddKeyValues(keyNames.Length, TextDataViewType.Instance,
                    (ref VBuffer<ReadOnlyMemory<char>> value) => value = keyNames);
            metadataBuilder.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance,
                (ref ReadOnlyMemory<char> value) => value = scoreColumnKindValue.AsMemory());
            metadataBuilder.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance,
                (ref ReadOnlyMemory<char> value) => value = AnnotationUtils.Const.ScoreValueKind.PredictedLabel.AsMemory());

            // Build a schema consisting of a single column.
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(AnnotationUtils.Const.ScoreValueKind.PredictedLabel, scoreType, metadataBuilder.ToAnnotations());

            return schemaBuilder.ToSchema();
        }
    }
}
