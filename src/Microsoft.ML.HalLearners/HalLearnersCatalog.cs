// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.SymSgd;
using Microsoft.ML.Transforms.Projections;

namespace Microsoft.ML
{
    /// <summary>
    /// The trainer context extensions for the <see cref="OlsLinearRegressionTrainer"/> and <see cref="SymSgdClassificationTrainer"/>.
    /// </summary>
    public static class HalLearnersCatalog
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OlsLinearRegressionTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="labelColumn">The label column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="weights">The weights column.</param>
        public static OlsLinearRegressionTrainer OrdinaryLeastSquares(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            var options = new OlsLinearRegressionTrainer.Options
            {
                LabelColumn = labelColumn,
                FeatureColumn = featureColumn,
                WeightColumn = weights != null ? Optional<string>.Explicit(weights) : Optional<string>.Implicit(DefaultColumnNames.Weight)
            };

            return new OlsLinearRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OlsLinearRegressionTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="OlsLinearRegressionTrainer.Options"/>.</param>
        public static OlsLinearRegressionTrainer OrdinaryLeastSquares(
            this RegressionContext.RegressionTrainers ctx,
            OlsLinearRegressionTrainer.Options options)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(ctx);
            return new OlsLinearRegressionTrainer(env, options);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="SymSgdClassificationTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="labelColumn">The label column.</param>
        /// <param name="featureColumn">The features column.</param>
        public static SymSgdClassificationTrainer SymbolicStochasticGradientDescent(
            this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);

            var options = new SymSgdClassificationTrainer.Options
            {
                LabelColumn = labelColumn,
                FeatureColumn = featureColumn,
            };

            return new SymSgdClassificationTrainer(env, options);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="SymSgdClassificationTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="SymSgdClassificationTrainer.Options"/>.</param>
        public static SymSgdClassificationTrainer SymbolicStochasticGradientDescent(
            this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            SymSgdClassificationTrainer.Options options)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(options, nameof(options));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SymSgdClassificationTrainer(env, options);
        }

        /// <summary>
        /// Takes column filled with a vector of random variables with a known covariance matrix into a set of new variables whose covariance is the identity matrix,
        /// meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>.
        /// Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="eps">Whitening constant, prevents division by zero.</param>
        /// <param name="maxRows">Maximum number of rows used to train the transform.</param>
        /// <param name="pcaNum">In case of PCA whitening, indicates the number of components to retain.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[VectorWhiten](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog,
            string inputColumn, string outputColumn = null,
            WhiteningKind kind = VectorWhiteningTransformer.Defaults.Kind,
            float eps = VectorWhiteningTransformer.Defaults.Eps,
            int maxRows = VectorWhiteningTransformer.Defaults.MaxRows,
            int pcaNum = VectorWhiteningTransformer.Defaults.PcaNum)
                => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, kind, eps, maxRows, pcaNum);

        /// <summary>
        /// Takes columns filled with a vector of random variables with a known covariance matrix into a set of new variables whose
        /// covariance is the identity matrix, meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of the whitening process for each column pair.</param>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, params VectorWhiteningTransformer.ColumnInfo[] columns)
            => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }
}
