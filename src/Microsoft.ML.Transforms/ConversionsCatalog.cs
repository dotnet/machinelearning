// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for the <see cref="TransformsCatalog.ConversionTransforms"/>.
    /// </summary>
    public static class ConversionsCatalog
    {

        /// <summary>
        ///  Convert the key types back to binary vector.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">Specifies the output and input columns on which the transformation should be applied.</param>
        [BestFriend]
        internal static KeyToBinaryVectorMappingEstimator MapKeyToBinaryVector(this TransformsCatalog.ConversionTransforms catalog,
            params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new KeyToBinaryVectorMappingEstimator(env, InputOutputColumnPair.ConvertToValueTuples(columns));
        }

        /// <summary>
        ///  Convert the key types back to binary vector.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapKeyToVector](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversions/MapKeyToBinaryVector.cs)]
        /// ]]></format>
        /// </example>
        public static KeyToBinaryVectorMappingEstimator MapKeyToBinaryVector(this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null)
            => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);
    }
}
