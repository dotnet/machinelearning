// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for normalizer operations.
    /// </summary>
    public static class NormalizationCatalog
    {
        /// <summary>
        /// Normalize (rescale) several columns according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mode">The <see cref="NormalizingEstimator.NormalizationMode"/> used to map the old values to the new ones. </param>
        /// <param name="columns">The pairs of input and output columns.</param>
        [BestFriend]
        internal static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            NormalizingEstimator.NormalizationMode mode,
            params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new NormalizingEstimator(env, mode, InputOutputColumnPair.ConvertToValueTuples(columns));
        }

        /// <summary>
        /// It normalizes the data based on the observed minimum and maximum values of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeMinMax](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeMinMax.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator NormalizeMinMax(this TransformsCatalog catalog,
           string outputColumnName, string inputColumnName = null,
           long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
           bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched)
        {
            var columnOptions = new NormalizingEstimator.MinMaxColumnOptions(outputColumnName, inputColumnName, maximumExampleCount, fixZero);
            return new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);
        }

        /// <summary>
        /// It normalizes the data based on the observed minimum and maximum values of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">List of Output and Input column pairs.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        public static NormalizingEstimator NormalizeMinMax(this TransformsCatalog catalog, InputOutputColumnPair[] columns,
           long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
           bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched) =>
            new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog),
                columns.Select(column =>
                    new NormalizingEstimator.MinMaxColumnOptions(column.OutputColumnName, column.InputColumnName, maximumExampleCount, fixZero)).ToArray());

        /// <summary>
        /// It normalizes the data based on the computed mean and variance of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="useCdf">Whether to use CDF as the output.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeMeanVariance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeMeanVariance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator NormalizeMeanVariance(this TransformsCatalog catalog,
            string outputColumnName, string inputColumnName = null,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched,
            bool useCdf = NormalizingEstimator.Defaults.MeanVarCdf)
        {
            var columnOptions = new NormalizingEstimator.MeanVarianceColumnOptions(outputColumnName, inputColumnName, maximumExampleCount, fixZero, useCdf);
            return new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);
        }

        /// <summary>
        /// It normalizes the data based on the computed mean and variance of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">List of Output and Input column pairs.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="useCdf">Whether to use CDF as the output.</param>
        public static NormalizingEstimator NormalizeMeanVariance(this TransformsCatalog catalog, InputOutputColumnPair[] columns,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched,
            bool useCdf = NormalizingEstimator.Defaults.MeanVarCdf) =>
                new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog),
                    columns.Select(column =>
                        new NormalizingEstimator.MeanVarianceColumnOptions(column.OutputColumnName, column.InputColumnName, maximumExampleCount, fixZero, useCdf)).ToArray());

        /// <summary>
        /// It normalizes the data based on the computed mean and variance of the logarithm of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="useCdf">Whether to use CDF as the output.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeLogMeanVariance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeLogMeanVariance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator NormalizeLogMeanVariance(this TransformsCatalog catalog,
            string outputColumnName, string inputColumnName = null,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool useCdf = NormalizingEstimator.Defaults.LogMeanVarCdf)
        {
            var columnOptions = new NormalizingEstimator.LogMeanVarianceColumnOptions(outputColumnName, inputColumnName, maximumExampleCount, useCdf);
            return new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);
        }

        /// <summary>
        /// It normalizes the data based on the computed mean and variance of the logarithm of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">List of Output and Input column pairs.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="useCdf">Whether to use CDF as the output.</param>
        public static NormalizingEstimator NormalizeLogMeanVariance(this TransformsCatalog catalog, InputOutputColumnPair[] columns,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool useCdf = NormalizingEstimator.Defaults.LogMeanVarCdf) =>
            new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog),
                columns.Select(column =>
                    new NormalizingEstimator.LogMeanVarianceColumnOptions(column.OutputColumnName, column.InputColumnName, maximumExampleCount, useCdf)).ToArray());

        /// <summary>
        /// The values are assigned into bins with equal density.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="maximumBinCount">Maximum number of bins (power of 2 recommended).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeBinning](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeBinning.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator NormalizeBinning(this TransformsCatalog catalog,
            string outputColumnName, string inputColumnName = null,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched,
            int maximumBinCount = NormalizingEstimator.Defaults.MaximumBinCount)
        {
            var columnOptions = new NormalizingEstimator.BinningColumnOptions(outputColumnName, inputColumnName, maximumExampleCount, fixZero, maximumBinCount);
            return new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);
        }

        /// <summary>
        /// The values are assigned into bins with equal density.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">List of Output and Input column pairs.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="maximumBinCount">Maximum number of bins (power of 2 recommended).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeBinning](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeBinningMulticolumn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator NormalizeBinning(this TransformsCatalog catalog, InputOutputColumnPair[] columns,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched,
            int maximumBinCount = NormalizingEstimator.Defaults.MaximumBinCount) =>
            new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog),
                columns.Select(column =>
                    new NormalizingEstimator.BinningColumnOptions(column.OutputColumnName, column.InputColumnName, maximumExampleCount, fixZero, maximumBinCount)).ToArray());

        /// <summary>
        /// The values are assigned into bins based on correlation with the <paramref name="labelColumnName"/> column.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="labelColumnName">Name of the label column for supervised binning.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="maximumBinCount">Maximum number of bins (power of 2 recommended).</param>
        /// <param name="mininimumExamplesPerBin">Minimum number of examples per bin.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeBinning](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeSupervisedBinning.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator NormalizeSupervisedBinning(this TransformsCatalog catalog,
            string outputColumnName, string inputColumnName = null,
            string labelColumnName = DefaultColumnNames.Label,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched,
            int maximumBinCount = NormalizingEstimator.Defaults.MaximumBinCount,
            int mininimumExamplesPerBin = NormalizingEstimator.Defaults.MininimumBinSize)
        {
            var columnOptions = new NormalizingEstimator.SupervisedBinningColumOptions(outputColumnName, inputColumnName, labelColumnName, maximumExampleCount, fixZero, maximumBinCount, mininimumExamplesPerBin);
            return new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);
        }

        /// <summary>
        /// The values are assigned into bins based on correlation with the <paramref name="labelColumnName"/> column.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">List of Output and Input column pairs.</param>
        /// <param name="labelColumnName">Name of the label column for supervised binning.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="maximumBinCount">Maximum number of bins (power of 2 recommended).</param>
        /// <param name="mininimumExamplesPerBin">Minimum number of examples per bin.</param>
        public static NormalizingEstimator NormalizeSupervisedBinning(this TransformsCatalog catalog, InputOutputColumnPair[] columns,
            string labelColumnName = DefaultColumnNames.Label,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched,
            int maximumBinCount = NormalizingEstimator.Defaults.MaximumBinCount,
            int mininimumExamplesPerBin = NormalizingEstimator.Defaults.MininimumBinSize) =>
                new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog),
                    columns.Select(column =>
                        new NormalizingEstimator.SupervisedBinningColumOptions(
                            column.OutputColumnName, column.InputColumnName, labelColumnName, maximumExampleCount, fixZero, maximumBinCount, mininimumExamplesPerBin)).ToArray());

        /// <summary>
        /// Normalize (rescale) columns according to specified custom parameters.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The normalization settings for all the columns</param>
        [BestFriend]
        internal static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            params NormalizingEstimator.ColumnOptionsBase[] columns)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Takes column filled with a vector of floats and normalize its <paramref name="norm"/> to one. By setting <paramref name="ensureZeroMean"/> to <see langword="true"/>,
        /// a pre-processing step would be applied to make the specified column's mean be a zero vector.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="norm">Type of norm to use to normalize each sample. The indicated norm of the resulted vector will be normalized to one.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <remarks>
        /// This transform performs the following operation on a each row X:  Y = (X - M(X)) / D(X)
        /// where M(X) is scalar value of mean for all elements in the current row if <paramref name="ensureZeroMean"/>set to <see langword="true"/> or <value>0</value> othewise
        /// and D(X) is scalar value of selected <paramref name="norm"/>.
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeLpNorm](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeLpNorm.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LpNormNormalizingEstimator NormalizeLpNorm(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null,
            LpNormNormalizingEstimatorBase.NormFunction norm = LpNormNormalizingEstimatorBase.Defaults.Norm, bool ensureZeroMean = LpNormNormalizingEstimatorBase.Defaults.LpEnsureZeroMean)
            => new LpNormNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, norm, ensureZeroMean);

        /// <summary>
        /// Takes column filled with a vector of floats and normalize its norm to one. Note that the allowed norm functions are defined in <see cref="LpNormNormalizingEstimatorBase.NormFunction"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the lp-normalization process for each column pair.</param>
        [BestFriend]
        internal static LpNormNormalizingEstimator NormalizeLpNorm(this TransformsCatalog catalog, params LpNormNormalizingEstimator.ColumnOptions[] columns)
            => new LpNormNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Takes column filled with a vector of floats and computes global contrast normalization of it. By setting <paramref name="ensureZeroMean"/> to <see langword="true"/>,
        /// a pre-processing step would be applied to make the specified column's mean be a zero vector.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <param name="ensureUnitStandardDeviation">If <see langword="true"/>, resulted vector's standard deviation would be one. Otherwise, resulted vector's L2-norm would be one.</param>
        /// <param name="scale">Scale features by this value.</param>
        /// <remarks>
        /// This transform performs the following operation on a row X: Y = scale * (X - M(X)) / D(X)
        /// where M(X) is scalar value of mean for all elements in the current row if <paramref name="ensureZeroMean"/>set to <see langword="true"/> or <value>0</value> othewise
        /// D(X) is scalar value of standard deviation for row if <paramref name="ensureUnitStandardDeviation"/> set to <see langword="true"/> or
        /// L2 norm of this row vector if <paramref name="ensureUnitStandardDeviation"/> set to <see langword="false"/> and scale is <paramref name="scale"/>.
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeGlobalContrast](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/NormalizeGlobalContrast.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static GlobalContrastNormalizingEstimator NormalizeGlobalContrast(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null,
             bool ensureZeroMean = LpNormNormalizingEstimatorBase.Defaults.GcnEnsureZeroMean,
             bool ensureUnitStandardDeviation = LpNormNormalizingEstimatorBase.Defaults.EnsureUnitStdDev,
             float scale = LpNormNormalizingEstimatorBase.Defaults.Scale)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, ensureZeroMean, ensureUnitStandardDeviation, scale);

        /// <summary>
        /// Takes columns filled with a vector of floats and computes global contrast normalization of it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the gcn-normaliztion process for each column pair.</param>
        [BestFriend]
        internal static GlobalContrastNormalizingEstimator NormalizeGlobalContrast(this TransformsCatalog catalog, params GlobalContrastNormalizingEstimator.ColumnOptions[] columns)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
