// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// The trainer catalog extensions for the <see cref="OrdinaryLeastSquaresRegressionTrainer"/> and <see cref="SymbolicStochasticGradientDescentClassificationTrainer"/>.
    /// </summary>
    public static class MklComponentsCatalog
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OrdinaryLeastSquaresRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[OrdinaryLeastSquares](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/OrdinaryLeastSquares.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OrdinaryLeastSquaresRegressionTrainer OrdinaryLeastSquares(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            var options = new OrdinaryLeastSquaresRegressionTrainer.Options
            {
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
                ExampleWeightColumnName = exampleWeightColumnName
            };

            return new OrdinaryLeastSquaresRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OrdinaryLeastSquaresRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="OrdinaryLeastSquaresRegressionTrainer.Options"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[OrdinaryLeastSquares](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/OrdinaryLeastSquaresWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OrdinaryLeastSquaresRegressionTrainer OrdinaryLeastSquares(
            this RegressionCatalog.RegressionTrainers catalog,
            OrdinaryLeastSquaresRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new OrdinaryLeastSquaresRegressionTrainer(env, options);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="SymbolicStochasticGradientDescentClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="numberOfIterations">Number of training iterations.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SymbolicStochasticGradientDescent](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SymbolicStochasticGradientDescent.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SymbolicStochasticGradientDescentClassificationTrainer SymbolicStochasticGradientDescent(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            int numberOfIterations = SymbolicStochasticGradientDescentClassificationTrainer.Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);

            var options = new SymbolicStochasticGradientDescentClassificationTrainer.Options
            {
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
            };

            return new SymbolicStochasticGradientDescentClassificationTrainer(env, options);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="SymbolicStochasticGradientDescentClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="SymbolicStochasticGradientDescentClassificationTrainer.Options"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SymbolicStochasticGradientDescent](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SymbolicStochasticGradientDescentWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SymbolicStochasticGradientDescentClassificationTrainer SymbolicStochasticGradientDescent(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            SymbolicStochasticGradientDescentClassificationTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SymbolicStochasticGradientDescentClassificationTrainer(env, options);
        }

        /// <summary>
        /// Takes column filled with a vector of random variables with a known covariance matrix into a set of new variables whose covariance is the identity matrix,
        /// meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="epsilon">Whitening constant, prevents division by zero.</param>
        /// <param name="maximumNumberOfRows">Maximum number of rows used to train the transform.</param>
        /// <param name="rank">In case of PCA whitening, indicates the number of components to retain.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[VectorWhiten](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Projection/VectorWhiten.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null,
            WhiteningKind kind = VectorWhiteningEstimator.Defaults.Kind,
            float epsilon = VectorWhiteningEstimator.Defaults.Epsilon,
            int maximumNumberOfRows = VectorWhiteningEstimator.Defaults.MaximumNumberOfRows,
            int rank = VectorWhiteningEstimator.Defaults.Rank)
                => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, kind, epsilon, maximumNumberOfRows, rank);

        /// <summary>
        /// Takes columns filled with a vector of random variables with a known covariance matrix into a set of new variables whose
        /// covariance is the identity matrix, meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of the whitening process for each column pair.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[VectorWhiten](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Projection/VectorWhitenWithColumnOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog catalog, params VectorWhiteningEstimator.ColumnOptions[] columns)
            => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }
}
