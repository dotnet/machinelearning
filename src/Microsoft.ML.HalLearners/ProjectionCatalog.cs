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
        /// Takes column filled with a vector of random variables with a known covariance matrix into a set of new variables whose covariance is the identity matrix,
        /// meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="eps">Whitening constant, prevents division by zero.</param>
        /// <param name="maxRows">Maximum number of rows used to train the transform.</param>
        /// <param name="pcaNum">In case of PCA whitening, indicates the number of components to retain.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[VectorWhiten](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ProjectionTransforms.cs?range=5-11,16-115)]
        /// ]]>
        /// </format>
        /// </example>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, string inputColumn, string outputColumn = null,
        WhiteningKind kind = VectorWhiteningTransformer.Defaults.Kind,
        float eps = VectorWhiteningTransformer.Defaults.Eps,
        int maxRows = VectorWhiteningTransformer.Defaults.MaxRows,
        int pcaNum = VectorWhiteningTransformer.Defaults.PcaNum)
        => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, kind, eps, maxRows, pcaNum);

        /// <summary>
        /// Takes columns filled with a vector of random variables with a known covariance matrix into a set of new variables whose covariance is the identity matrix,
        /// meaning that they are uncorrelated and each have variance 1.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of the whitening process for each column pair.</param>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, params VectorWhiteningTransformer.ColumnInfo[] columns)
            => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
