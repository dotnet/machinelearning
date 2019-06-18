// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog"/> to create instances of
    /// key to binary vector mapping transformer components
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
        /// Create a <see cref="KeyToBinaryVectorMappingEstimator"/>, which converts key types to their corresponding binary representation of the original value.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The data type is a known-size vector of <see cref="System.Single"/> representing the input value.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The data type is a key or a known-size vector of keys.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MapKeyToVector](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Conversion/MapKeyToBinaryVector.cs)]
        /// ]]></format>
        /// </example>
        public static KeyToBinaryVectorMappingEstimator MapKeyToBinaryVector(this TransformsCatalog.ConversionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null)
            => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);
    }
}
