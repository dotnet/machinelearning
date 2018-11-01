// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Projections;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for the VectorWhiteningEstimator.
    /// </summary>
    public static class VectorWhiteningEstimatorCatalog
    {
        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="eps">Whitening constant, prevents division by zero.</param>
        /// <param name="maxRows">Maximum number of rows used to train the transform.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        /// <param name="pcaNum">In case of PCA whitening, indicates the number of components to retain.</param>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, IHostEnvironment env, string inputColumn, string outputColumn,
            WhiteningKind kind = VectorWhiteningTransform.Defaults.Kind,
            float eps = VectorWhiteningTransform.Defaults.Eps,
            int maxRows = VectorWhiteningTransform.Defaults.MaxRows,
            bool saveInverse = VectorWhiteningTransform.Defaults.SaveInverse,
            int pcaNum = VectorWhiteningTransform.Defaults.PcaNum)
            => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, kind, eps, maxRows, saveInverse, pcaNum);

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="env">The environment.</param>
        /// <param name="columns"> Describes the settings of the transformation.</param>
        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, IHostEnvironment env, params VectorWhiteningTransform.ColInfo[] columns)
            => new VectorWhiteningEstimator(env, columns);
    }
}
