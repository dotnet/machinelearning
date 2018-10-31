﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML
{
    using HashDefaults = HashingEstimator.Defaults;
    using ConvertDefaults = ConvertingEstimator.Defaults;

    /// <summary>
    /// Extensions for the HashEstimator.
    /// </summary>
    public static class HashingEstimatorCatalog
    {
        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public static HashingEstimator Hash(this TransformsCatalog.Conversions catalog, string inputColumn, string outputColumn = null,
            int hashBits = HashDefaults.HashBits, int invertHash = HashDefaults.InvertHash)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, hashBits, invertHash);

        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public static HashingEstimator Hash(this TransformsCatalog.Conversions catalog, params HashTransformer.ColumnInfo[] columns)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="outputKind">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        public static ConvertingEstimator ConvertTo(this TransformsCatalog.Conversions catalog, string inputColumn, string outputColumn = null,
            DataKind outputKind = ConvertDefaults.DefaultOutputKind)
            => new ConvertingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, outputKind);

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public static ConvertingEstimator ConvertTo(this TransformsCatalog.Conversions catalog, params ConvertingTransform.ColumnInfo[] columns)
            => new ConvertingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
