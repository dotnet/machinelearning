// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Prediction;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML
{
    public static class RecommenderCatalog
    {
        /// <summary>
        /// Initializing a new instance of <see cref="MatrixFactorizationTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext.RegressionTrainers"/> instance.</param>
        /// <param name="matrixColumnIndexColumnName">The name of the column hosting the matrix's column IDs.</param>
        /// <param name="matrixRowIndexColumnName">The name of the column hosting the matrix's row IDs.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        /// <param name="context">The <see cref="TrainerEstimatorContext"/> for additional input data to training.</param>
        public static MatrixFactorizationTrainer MatrixFactorization(this RegressionContext.RegressionTrainers ctx,
            string matrixColumnIndexColumnName,
            string matrixRowIndexColumnName,
            string labelColumn = DefaultColumnNames.Label,
            TrainerEstimatorContext context = null,
            Action<MatrixFactorizationTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            return new MatrixFactorizationTrainer(CatalogUtils.GetEnvironment(ctx), matrixColumnIndexColumnName, matrixRowIndexColumnName, labelColumn, context, advancedSettings);
        }
    }
}
