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
    /// Collection of extension methods for <see cref="RegressionCatalog.RegressionTrainers"/>,
    /// <see cref="BinaryClassificationCatalog.BinaryClassificationTrainers"/>, and <see cref="TransformsCatalog"/>
    /// to create MKL (Math Kernel Library) trainer and transform components.
    /// </summary>
    public static class MklComponentsCatalog
    {
        /// <summary>
        /// Create <see cref="OlsTrainer"/>, which predicts a target using a linear regression model.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Single"/>.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[OrdinaryLeastSquares](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/OrdinaryLeastSquares.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OlsTrainer Ols(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            var options = new OlsTrainer.Options
            {
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
                ExampleWeightColumnName = exampleWeightColumnName
            };

            return new OlsTrainer(env, options);
        }

        /// <summary>
        /// Create <see cref="OlsTrainer"/> with advanced options, which predicts a target using a linear regression model.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="OlsTrainer.Options"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[OrdinaryLeastSquares](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/OrdinaryLeastSquaresWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OlsTrainer Ols(
            this RegressionCatalog.RegressionTrainers catalog,
            OlsTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new OlsTrainer(env, options);
        }

        /// <summary>
        /// Create <see cref="SymbolicSgdLogisticRegressionBinaryTrainer"/>, which predicts a target using a linear binary classification model trained over boolean label data.
        /// Stochastic gradient descent (SGD) is an iterative algorithm that optimizes a differentiable objective function.
        /// The <see cref="SymbolicSgdLogisticRegressionBinaryTrainer"/> parallelizes SGD using <a href="https://www.microsoft.com/en-us/research/project/project-parade/#!symbolic-execution">symbolic execution</a>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Boolean"/>.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="numberOfIterations">Number of training iterations.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SymbolicSgdLogisticRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SymbolicSgdLogisticRegression.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SymbolicSgdLogisticRegressionBinaryTrainer SymbolicSgdLogisticRegression(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            int numberOfIterations = SymbolicSgdLogisticRegressionBinaryTrainer.Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);

            var options = new SymbolicSgdLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
            };

            return new SymbolicSgdLogisticRegressionBinaryTrainer(env, options);
        }

        /// <summary>
        /// Create <see cref= "SymbolicSgdLogisticRegressionBinaryTrainer" /> with advanced options, which predicts a target using a linear binary classification model trained over boolean label data.
        /// Stochastic gradient descent (SGD) is an iterative algorithm that optimizes a differentiable objective function.
        /// The <see cref="SymbolicSgdLogisticRegressionBinaryTrainer"/> parallelizes SGD using <a href="https://www.microsoft.com/en-us/research/project/project-parade/#!symbolic-execution">symbolic execution</a>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Algorithm advanced options. See <see cref="SymbolicSgdLogisticRegressionBinaryTrainer.Options"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SymbolicSgdLogisticRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SymbolicSgdLogisticRegressionWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SymbolicSgdLogisticRegressionBinaryTrainer SymbolicSgdLogisticRegression(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            SymbolicSgdLogisticRegressionBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SymbolicSgdLogisticRegressionBinaryTrainer(env, options);
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
        /// [!code-csharp[VectorWhiten](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Projection/VectorWhitenWithOptions.cs)]
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
        /// [!code-csharp[VectorWhiten](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Projection/VectorWhitenWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        [BestFriend]
        internal static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog catalog, params VectorWhiteningEstimator.ColumnOptions[] columns)
            => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
