// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Categorical;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for the ValueToKeyMappingEstimator
    /// </summary>
   public static class ValueToKeyCatalog
    {
        /// <summary>
        /// Initializes a new instance of <see cref="ValueToKeyMappingEstimator"/>.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">Name of the column to be transformed.</param>
        /// <param name="outputColumn">Name of the output column. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="maxNumTerms">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="sort">How items should be ordered when vectorized. By default, they will be in the order encountered.
        /// If by value items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.CategoricalTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int maxNumTerms = ValueToKeyMappingEstimator.Defaults.MaxNumTerms,
            TermTransformer.SortOrder sort = ValueToKeyMappingEstimator.Defaults.Sort)
           => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, maxNumTerms, sort);

        /// <summary>
        /// Initializes a new instance of <see cref="ValueToKeyMappingEstimator"/> loading the terms to use from <paramref name="file"/>.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The data columns to map to keys.</param>
        /// <param name="file">The path of the file containing the terms.</param>
        /// <param name="termsColumn"></param>
        /// <param name="loaderFactory"></param>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.CategoricalTransforms catalog,
            TermTransformer.ColumnInfo[] columns,
            string file = null,
            string termsColumn = null,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory = null)
            => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns, file, termsColumn, loaderFactory);
    }
}
