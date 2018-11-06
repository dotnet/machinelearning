// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;

namespace Microsoft.ML
{
    using CharTokenizingDefaults = CharacterTokenizingEstimator.Defaults;
    using TextNormalizeDefaults = TextNormalizingEstimator.Defaults;

    public static class TextCatalog
    {
        /// <summary>
        /// Transform a text column into featurized float array that represents counts of ngrams and char-grams.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The input column</param>
        /// <param name="outputColumn">The output column</param>
        /// <param name="advancedSettings">Advanced transform settings</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TextTransform.cs?line=37 )]
        /// ]]>
        /// </format>
        /// </example>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TextTransform.cs?range=6-12,17-86)]
        /// ]]>
        /// </format>
        /// </example>
        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            Action<TextFeaturizingEstimator.Settings> advancedSettings = null)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, advancedSettings);

        /// <summary>
        /// Transform several text columns into featurized float array that represents counts of ngrams and char-grams.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumns">The input columns</param>
        /// <param name="outputColumn">The output column</param>
        /// <param name="advancedSettings">Advanced transform settings</param>
        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            IEnumerable<string> inputColumns,
            string outputColumn,
            Action<TextFeaturizingEstimator.Settings> advancedSettings = null)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumns, outputColumn, advancedSettings);

        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumn"/> and output the tokens as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        public static CharacterTokenizingEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters)
            => new CharacterTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                useMarkerCharacters, new[] { (inputColumn, outputColumn) });

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>

        public static CharacterTokenizingEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters,
            params (string inputColumn, string outputColumn)[] columns)
            => new CharacterTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), useMarkerCharacters, columns);

        /// <summary>
        /// Normalizes incoming text in <paramref name="inputColumn"/> by changing case, removing diacritical marks, punctuation marks and/or numbers
        /// and outputs new text as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The column containing text to normalize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="textCase">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        public static TextNormalizingEstimator NormalizeText(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            TextNormalizingEstimator.CaseNormalizationMode textCase = TextNormalizeDefaults.TextCase,
            bool keepDiacritics = TextNormalizeDefaults.KeepDiacritics,
            bool keepPunctuations = TextNormalizeDefaults.KeepPunctuations,
            bool keepNumbers = TextNormalizeDefaults.KeepNumbers)
            => new TextNormalizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, textCase, keepDiacritics, keepPunctuations, keepNumbers);

        /// <summary>
        /// Extracts word embeddings.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The input column.</param>
        /// <param name="outputColumn">The optional output column. If it is <value>null</value> the input column will be substituted with its value.</param>
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingsTransform.PretrainedModelKind"/> to use. </param>
        public static WordEmbeddingsExtractorEstimator ExtractWordEmbeedings(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            WordEmbeddingsTransform.PretrainedModelKind modelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe)
            => new WordEmbeddingsExtractorEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn, modelKind);

        /// <summary>
        /// Extracts word embeddings.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The input column.</param>
        /// <param name="outputColumn">The optional output column. If it is <value>null</value> the input column will be substituted with its value.</param>
        /// <param name="customModelFile">The path of the pre-trained embeedings model to use. </param>
        public static WordEmbeddingsExtractorEstimator ExtractWordEmbeedings(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string customModelFile,
            string outputColumn = null)
            => new WordEmbeddingsExtractorEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, customModelFile);

        /// <summary>
        /// Extracts word embeddings.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingsTransform.PretrainedModelKind"/> to use. </param>
        /// <param name="columns">The array columns, and per-column configurations to extract embeedings from.</param>
        public static WordEmbeddingsExtractorEstimator ExtractWordEmbeedings(this TransformsCatalog.TextTransforms catalog,
           WordEmbeddingsTransform.PretrainedModelKind modelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe,
           params WordEmbeddingsTransform.ColumnInfo[] columns)
            => new WordEmbeddingsExtractorEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), modelKind, columns);

        /// <summary>
        /// Tokenizes incoming text in <paramref name="inputColumn"/>, using <paramref name="separators"/> as separators,
        /// and outputs the tokens as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            char[] separators = null)
            => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn, separators);

        /// <summary>
        /// Tokenizes incoming text in input columns and outputs the tokens using <paramref name="separators"/> as separators.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            (string inputColumn, string outputColumn)[] columns,
            char[] separators = null)
            => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, separators);

        /// <summary>
        ///  Tokenizes incoming text in input columns, using per-column configurations, and outputs the tokens.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            params WordTokenizeTransform.ColumnInfo[] columns)
          => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

    }
}
