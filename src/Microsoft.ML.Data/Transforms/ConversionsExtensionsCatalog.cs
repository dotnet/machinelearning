// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML
{
    using ConvertDefaults = TypeConvertingEstimator.Defaults;
    using HashDefaults = HashingEstimator.Defaults;

    /// <summary>
    /// Extensions for the HashEstimator.
    /// </summary>
    public static class ConversionsExtensionsCatalog
    {
        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public static HashingEstimator Hash(this TransformsCatalog.ConversionTransforms catalog, string inputColumn, string outputColumn = null,
            int hashBits = HashDefaults.HashBits, int invertHash = HashDefaults.InvertHash)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, hashBits, invertHash);

        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public static HashingEstimator Hash(this TransformsCatalog.ConversionTransforms catalog, params HashingTransformer.ColumnInfo[] columns)
            => new HashingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="outputKind">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
        public static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog, string inputColumn, string outputColumn = null,
            DataKind outputKind = ConvertDefaults.DefaultOutputKind)
            => new TypeConvertingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, outputKind);

        /// <summary>
        /// Changes column type of the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public static TypeConvertingEstimator ConvertType(this TransformsCatalog.ConversionTransforms catalog, params TypeConvertingTransformer.ColumnInfo[] columns)
            => new TypeConvertingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original values.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, string inputColumn)
            => new KeyToValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn);

        /// <summary>
        ///  Convert the key types (name of the column specified in the first item of the tuple) back to their original values
        ///  (named as specified in the second item of the tuple).
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static KeyToValueMappingEstimator MapKeyToValue(this TransformsCatalog.ConversionTransforms catalog, params (string input, string output)[] columns)
             => new KeyToValueMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original vectors.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The input column to map back to vectors.</param>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            params KeyToVectorMappingTransformer.ColumnInfo[] columns)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original vectors.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">The name of the input column.</param>
        /// <param name="outputColumn">The name of the output column.</param>
        /// <param name="bag">Whether bagging is used for the conversion. </param>
        public static KeyToVectorMappingEstimator MapKeyToVector(this TransformsCatalog.ConversionTransforms catalog,
            string inputColumn, string outputColumn = null, bool bag = KeyToVectorMappingEstimator.Defaults.Bag)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, bag);

        /// <summary>
        /// Converts value types into <see cref="KeyType"/>.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">Name of the column to be transformed.</param>
        /// <param name="outputColumn">Name of the output column. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="maxNumTerms">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="sort">How items should be ordered when vectorized. By default, they will be in the order encountered.
        /// If by value items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int maxNumTerms = ValueToKeyMappingEstimator.Defaults.MaxNumTerms,
            ValueToKeyMappingTransformer.SortOrder sort = ValueToKeyMappingEstimator.Defaults.Sort)
           => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, maxNumTerms, sort);

        /// <summary>
        /// Converts value types into <see cref="KeyType"/> loading the keys to use from <paramref name="file"/>.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The data columns to map to keys.</param>
        /// <param name="file">The path of the file containing the terms.</param>
        /// <param name="termsColumn"></param>
        /// <param name="loaderFactory"></param>
        public static ValueToKeyMappingEstimator MapValueToKey(this TransformsCatalog.ConversionTransforms catalog,
            ValueToKeyMappingTransformer.ColumnInfo[] columns,
            string file = null,
            string termsColumn = null,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory = null)
            => new ValueToKeyMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns, file, termsColumn, loaderFactory);
    }
}
