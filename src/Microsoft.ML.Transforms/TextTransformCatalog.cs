// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML
{
    public static class TextTransformCatalog
    {
        /// <summary>
        /// Transform a text column into featurized float array that represents counts of ngrams and char-grams.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputColumn">The input column</param>
        /// <param name="outputColumn">The output column</param>
        /// <param name="advancedSettings">Advanced transform settings</param>
        public static TextTransform FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            string inputColumn, string outputColumn = null,
            Action<TextTransform.Settings> advancedSettings = null)
            => new TextTransform(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn, advancedSettings);

        /// <summary>
        /// Transform several text columns into featurized float array that represents counts of ngrams and char-grams.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputColumns">The input columns</param>
        /// <param name="outputColumn">The output column</param>
        /// <param name="advancedSettings">Advanced transform settings</param>
        public static TextTransform FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            IEnumerable<string> inputColumns, string outputColumn,
            Action<TextTransform.Settings> advancedSettings = null)
            => new TextTransform(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumns, outputColumn, advancedSettings);
    }
}
