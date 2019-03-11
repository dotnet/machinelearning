using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for normalizer operations.
    /// </summary>
    public static class NormalizationCatalog
    {
        /// <summary>
        /// Normalize (rescale) the column according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="mode">The <see cref="NormalizingEstimator.NormalizationMode"/> used to map the old values in the new scale. </param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Normalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Normalizer.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
           string outputColumnName, string inputColumnName = null,
            NormalizingEstimator.NormalizationMode mode = NormalizingEstimator.NormalizationMode.MinMax)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName ?? outputColumnName, mode);

        /// <summary>
        /// Normalize (rescale) several columns according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mode">The <see cref="NormalizingEstimator.NormalizationMode"/> used to map the old values to the new ones. </param>
        /// <param name="columns">The pairs of input and output columns.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Normalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Normalizer.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            NormalizingEstimator.NormalizationMode mode,
            params ColumnOptions[] columns)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), mode, ColumnOptions.ConvertToValueTuples(columns));

        /// <summary>
        /// Normalize (rescale) columns according to specified custom parameters.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The normalization settings for all the columns</param>
        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            params NormalizingEstimator.ColumnOptionsBase[] columns)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Takes column filled with a vector of floats and normazlize its <paramref name="normKind"/> to one. By setting <paramref name="ensureZeroMean"/> to <see langword="true"/>,
        /// a pre-processing step would be applied to make the specified column's mean be a zero vector.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="normKind">Type of norm to use to normalize each sample. The indicated norm of the resulted vector will be normalized to one.</param>
        /// <param name="ensureZeroMean">If <see langword="true"/>, subtract mean from each value before normalizing and use the raw input otherwise.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LpNormalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null,
            LpNormalizingEstimatorBase.NormFunction normKind = LpNormalizingEstimatorBase.Defaults.Norm, bool ensureZeroMean = LpNormalizingEstimatorBase.Defaults.LpEnsureZeroMean)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, normKind, ensureZeroMean);

        /// <summary>
        /// Takes column filled with a vector of floats and normazlize its norm to one. Note that the allowed norm functions are defined in <see cref="LpNormalizingEstimatorBase.NormFunction"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the lp-normalization process for each column pair.</param>
        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog catalog, params LpNormalizingEstimator.LpNormColumnOptions[] columns)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

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
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[GlobalContrastNormalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=1-6,12-112)]
        /// ]]>
        /// </format>
        /// </example>
        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null,
             bool ensureZeroMean = LpNormalizingEstimatorBase.Defaults.GcnEnsureZeroMean,
             bool ensureUnitStandardDeviation = LpNormalizingEstimatorBase.Defaults.EnsureUnitStdDev,
             float scale = LpNormalizingEstimatorBase.Defaults.Scale)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, ensureZeroMean, ensureUnitStandardDeviation, scale);

        /// <summary>
        /// Takes columns filled with a vector of floats and computes global contrast normalization of it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns"> Describes the parameters of the gcn-normaliztion process for each column pair.</param>
        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog catalog, params GlobalContrastNormalizingEstimator.GcnColumnOptions[] columns)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
