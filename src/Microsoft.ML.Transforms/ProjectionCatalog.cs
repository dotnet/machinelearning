// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Projections;

namespace Microsoft.ML
{
    public static class ProjectionCatalog
    {
        /// <summary>
        /// Initializes a new instance of <see cref="RandomFourierFeaturizingEstimator"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the column to be transformed.</param>
        /// <param name="outputColumn">Name of the output column. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="newDim">The number of random Fourier features to create.</param>
        /// <param name="useSin">Create two features for every random Fourier frequency? (one for cos and one for sin).</param>
        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int newDim = RandomFourierFeaturizingEstimator.Defaults.NewDim,
            bool useSin = RandomFourierFeaturizingEstimator.Defaults.UseSin)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, newDim, useSin);

        /// <summary>
        /// Initializes a new instance of <see cref="RandomFourierFeaturizingEstimator"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The input columns to use for the transformation.</param>
        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog, params RffTransform.ColumnInfo[] columns)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
