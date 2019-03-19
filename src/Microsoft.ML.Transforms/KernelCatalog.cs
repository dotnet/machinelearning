// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// The catalog of numerical feature engineering using kernel methods.
    /// </summary>
    public static class KernelExpansionCatalog
    {
        /// <summary>
        /// Takes column filled with a vector of floats and maps its to a random low-dimensional feature space.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="rank">The number of random Fourier features to create.</param>
        /// <param name="useCosAndSinBases">If <see langword="true"/>, use both of cos and sin basis functions to create two features for every random Fourier frequency.
        /// Otherwise, only cos bases would be used.</param>
        /// <param name="generator">Which fourier generator to use.</param>
        /// <param name="seed">The seed of the random number generator for generating the new features (if unspecified, the global random is used).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CreateRandomFourierFeatures](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
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
        /// Takes columns filled with a vector of floats and maps its to a random low-dimensional feature space.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The input columns to use for the transformation.</param>
        [BestFriend]
        internal static ApproximatedKernelMappingEstimator ApproximatedKernelMap(this TransformsCatalog catalog, params ApproximatedKernelMappingEstimator.ColumnOptions[] columns)
            => new ApproximatedKernelMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
