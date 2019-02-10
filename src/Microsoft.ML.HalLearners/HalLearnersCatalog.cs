// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Transforms.Projections;

namespace Microsoft.ML
{
    /// <summary>
    /// The trainer catalog extensions for the <see cref="OlsLinearRegressionTrainer"/> and <see cref="SymSgdClassificationTrainer"/>.
    /// </summary>
    public static class HalLearnersCatalog
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OlsLinearRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="weights">The weights column.</param>
        public static OlsLinearRegressionTrainer OrdinaryLeastSquares(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
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
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="OlsLinearRegressionTrainer.Options"/>.</param>
        public static OlsLinearRegressionTrainer OrdinaryLeastSquares(
            this RegressionCatalog.RegressionTrainers catalog,
            OlsLinearRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new OlsLinearRegressionTrainer(env, options);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="SymSgdClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        public static SymSgdClassificationTrainer SymbolicStochasticGradientDescent(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);

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
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="SymSgdClassificationTrainer.Options"/>.</param>
        public static SymSgdClassificationTrainer SymbolicStochasticGradientDescent(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            SymSgdClassificationTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SymSgdClassificationTrainer(env, options);
        }

        /// <summary>
        /// Takes column filled with a vector of random variables with a known covariance matrix into a set of new variables whose covariance is the identity matrix,
        /// meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
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
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, string outputColumnName, string inputColumnName = null,
            WhiteningKind kind = VectorWhiteningEstimator.Defaults.Kind,
            float eps = VectorWhiteningEstimator.Defaults.Eps,
            int maxRows = VectorWhiteningEstimator.Defaults.MaxRows,
            int pcaNum = VectorWhiteningEstimator.Defaults.PcaNum)
                => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, kind, eps, maxRows, pcaNum);

        /// <summary>
        /// Takes columns filled with a vector of random variables with a known covariance matrix into a set of new variables whose
        /// covariance is the identity matrix, meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of the whitening process for each column pair.</param>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, params VectorWhiteningEstimator.ColumnInfo[] columns)
            => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }
}
