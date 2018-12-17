// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal static class ScoringUtils
    {
        /// <summary>
        /// Utility function that returns the SchemaShape of the generated output columns when scoring a model.
        /// Notice this assumes that the task specific scorer is used.
        /// </summary>
        /// <param name="predictionKind">The learning task that is being performed.</param>
        /// <param name="isNormalized">Whether the probability column (if generated) will be normalized or not.</param>
        /// <param name="labelColumn"></param>
        public static SchemaShape GetPredictorOutputColumns(PredictionKind predictionKind, bool isNormalized = true, SchemaShape.Column? labelColumn = null)
        {
            var columns = new List<SchemaShape.Column>();
            switch (predictionKind)
            {
                case PredictionKind.BinaryClassification:
                    columns.AddRange(new[]
                    {
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(isNormalized))),
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
                    });
                    break;

                case PredictionKind.MultiClassClassification:
                    var predLabelMeta = new SchemaShape(MetadataUtils.MetadataForMulticlassPredLabelColumn(labelColumn));
                    var scoreMeta = new SchemaShape(MetadataUtils.MetadataForMulticlassScoreColumn(labelColumn));

                    columns.AddRange(new[]
                    {
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, scoreMeta),
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, predLabelMeta)
                    });
                    break;

                case PredictionKind.Ranking:
                case PredictionKind.Recommendation:
                case PredictionKind.Regression:
                    columns.Add(new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())));
                    break;

                case PredictionKind.AnomalyDetection:
                    columns.AddRange(new[]
                    {
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
                    });
                    break;

                case PredictionKind.Clustering:
                    columns.AddRange(new[]
{
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                        new SchemaShape.Column(MetadataUtils.Const.ScoreValueKind.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
                    });
                    break;
                default:
                    throw Contracts.Except("Prediction.Kind not recognized and should define it's own output columns");
            }

            return new SchemaShape(columns);
        }
    }
}