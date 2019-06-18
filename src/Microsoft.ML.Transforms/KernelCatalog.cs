// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog"/> to create instances of kernel method
    /// feature engineering transformer components.
    /// </summary>
    public static class KernelExpansionCatalog
    {
        /// <summary>
        /// Create an <see cref="ApproximatedKernelMappingEstimator"/> that maps input vectors to a low dimensional
        /// feature space where inner products approximate a shift-invariant kernel function.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        ///  The data type on this column will be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>,
        /// the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates on known-sized vector of <see cref="System.Single"/> data type.</param>
        /// <param name="rank">The dimension of the feature space to map the input to.</param>
        /// <param name="useCosAndSinBases">If <see langword="true"/>, use both of cos and sin basis functions to create
        /// two features for every random Fourier frequency. Otherwise, only cos bases would be used. Note that if set
        /// to <see langword="true"/>, the dimension of the output feature space will be 2*<paramref name="rank"/>.</param>
        /// <param name="generator">The argument that indicates which kernel to use. The two available implementations
        /// are <see cref="GaussianKernel"/> and <see cref="LaplacianKernel"/>.</param>
        /// <param name="seed">The seed of the random number generator for generating the new features (if unspecified, the global random is used).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ApproximatedKernelMap](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ApproximatedKernelMap.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ApproximatedKernelMappingEstimator ApproximatedKernelMap(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null,
            int rank = ApproximatedKernelMappingEstimator.Defaults.Rank,
            bool useCosAndSinBases = ApproximatedKernelMappingEstimator.Defaults.UseCosAndSinBases,
            KernelBase generator = null,
            int? seed = null)
            => new ApproximatedKernelMappingEstimator(CatalogUtils.GetEnvironment(catalog),
                new[] { new ApproximatedKernelMappingEstimator.ColumnOptions(outputColumnName, rank, useCosAndSinBases, inputColumnName, generator, seed) });

        /// <summary>
        /// Create an <see cref="ApproximatedKernelMappingEstimator"/> that maps input vectors to a low dimensional
        /// feature space where inner products approximate a shift-invariant kernel function.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The input columns to use for the transformation.</param>
        [BestFriend]
        internal static ApproximatedKernelMappingEstimator ApproximatedKernelMap(this TransformsCatalog catalog, params ApproximatedKernelMappingEstimator.ColumnOptions[] columns)
            => new ApproximatedKernelMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
