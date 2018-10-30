using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    using HashDefaults = Transforms.HashEstimator.Defaults;

    /// <summary>
    /// Extensions for Column Copying Estiamtor.
    /// </summary>
    public static class ColumnCopyingCatalog
    {
        /// <summary>
        /// Copies the input column to another column named as specified in <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">The source column.</param>
        /// <param name="outputColumn">The new column, resulting from copying.</param>
        public static CopyColumnsEstimator CopyColumns(this TransformsCatalog catalog, string inputColumn, string outputColumn)
            => new CopyColumnsEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn);

        /// <summary>
        /// Copies the input column, name specified in the first item of the tuple,
        /// to another column, named as specified in the second item of the tuple.
        /// </summary>
        /// <param name="catalog">The transform's catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static CopyColumnsEstimator CopyColumns(this TransformsCatalog catalog, params (string source, string name)[] columns)
            => new CopyColumnsEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }

    /// <summary>
    /// Extension ColumnConcatenatingEstimator
    /// </summary>
    public static class ColumnConcatenatingEstimator
    {
        /// <summary>
        /// Concatenates two columns together.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumns">The names of the columns to concatenate together.</param>
        /// <param name="outputColumn">The name of the output column.</param>
        public static ConcatEstimator Concatenate(this TransformsCatalog catalog, string outputColumn, params string[] inputColumns)
            => new ConcatEstimator(CatalogUtils.GetEnvironment(catalog), outputColumn, inputColumns);

    }

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
        public static HashEstimator Hash(this TransformsCatalog catalog, string inputColumn, string outputColumn = null,
            int hashBits = HashDefaults.HashBits, int invertHash = HashDefaults.InvertHash)
            => new HashEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, hashBits, invertHash);

        /// <summary>
        /// Hashes the values in the input column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Description of dataset columns and how to process them.</param>
        public static HashEstimator Hash(this TransformsCatalog catalog, params HashTransformer.ColumnInfo[] columns)
            => new HashEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }

    public static class ToValueCatalog
    {
        /// <summary>
        /// Convert the key types back to their original values.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        public static KeyToValueEstimator ToValue(this TransformsCatalog.CategoricalTransforms catalog, string inputColumn)
            => new KeyToValueEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn);

        /// <summary>
        ///  Convert the key types (name of the column specified in the first item of the tuple) back to their original values
        ///  (named as specified in the second item of the tuple).
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static KeyToValueEstimator ToValue(this TransformsCatalog.CategoricalTransforms catalog, params (string input, string output)[] columns)
             => new KeyToValueEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
