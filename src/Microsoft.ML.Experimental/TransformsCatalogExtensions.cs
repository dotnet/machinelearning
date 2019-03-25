// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Experimental
{
    public static class TransformsCatalogExtensions
    {
        /// <summary>
        /// Normalize (rescale) the column according to the <see cref="NormalizingEstimator.NormalizationMode.MinMax"/> mode.
        /// It normalizes the data based on the observed minimum and maximum values of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        public static NormalizingEstimator NormalizeMinMax(this TransformsCatalog catalog,
           string outputColumnName, string inputColumnName = null,
           long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
           bool fixZero = NormalizingEstimator.Defaults.EnsureZeroUntouched)
        {
            var columnOptions = new NormalizingEstimator.MinMaxColumnOptions(outputColumnName, inputColumnName, maximumExampleCount, fixZero);
            return new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);
        }

        /// <summary>
        /// Normalize (rescale) the column according to the <see cref="NormalizingEstimator.NormalizationMode.MeanVariance"/> mode.
        /// It normalizes the data based on the computed mean and variance of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="useCdf">Whether to use CDF as the output.</param>
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
        /// Normalize (rescale) the column according to the <see cref="NormalizingEstimator.NormalizationMode.LogMeanVariance"/> mode.
        /// It normalizes the data based on the computed mean and variance of the logarithm of the data.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="useCdf">Whether to use CDF as the output.</param>
        public static NormalizingEstimator NormalizeLogMeanVariance(this TransformsCatalog catalog,
            string outputColumnName, string inputColumnName = null,
            long maximumExampleCount = NormalizingEstimator.Defaults.MaximumExampleCount,
            bool useCdf = NormalizingEstimator.Defaults.LogMeanVarCdf)
        {
            var columnOptions = new NormalizingEstimator.LogMeanVarianceColumnOptions(outputColumnName, inputColumnName, maximumExampleCount, useCdf);
            return new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);
        }

        /// <summary>
        /// Normalize (rescale) the column according to the <see cref="NormalizingEstimator.NormalizationMode.Binning"/> mode.
        /// The values are assigned into bins with equal density.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumExampleCount">Maximum number of examples used to train the normalizer.</param>
        /// <param name="fixZero">Whether to map zero to zero, preserving sparsity.</param>
        /// <param name="maximumBinCount">Maximum number of bins (power of 2 recommended).</param>
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
        /// Normalize (rescale) the column according to the <see cref="NormalizingEstimator.NormalizationMode.SupervisedBinning"/> mode.
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
    }
}
