﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;

namespace Microsoft.ML
{
    using CharTokenizingDefaults = TokenizingByCharactersEstimator.Defaults;
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
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TextTransform.cs)]
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
        public static TokenizingByCharactersEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters)
            => new TokenizingByCharactersEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                useMarkerCharacters, new[] { (inputColumn, outputColumn) });

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>

        public static TokenizingByCharactersEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters,
            params (string inputColumn, string outputColumn)[] columns)
            => new TokenizingByCharactersEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), useMarkerCharacters, columns);

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
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingsExtractingTransformer.PretrainedModelKind"/> to use. </param>
        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            WordEmbeddingsExtractingTransformer.PretrainedModelKind modelKind = WordEmbeddingsExtractingTransformer.PretrainedModelKind.Sswe)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn, modelKind);

        /// <summary>
        /// Extracts word embeddings.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The input column.</param>
        /// <param name="outputColumn">The optional output column. If it is <value>null</value> the input column will be substituted with its value.</param>
        /// <param name="customModelFile">The path of the pre-trained embeedings model to use. </param>
        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string customModelFile,
            string outputColumn = null)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, customModelFile);

        /// <summary>
        /// Extracts word embeddings.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingsExtractingTransformer.PretrainedModelKind"/> to use. </param>
        /// <param name="columns">The array columns, and per-column configurations to extract embeedings from.</param>
        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
           WordEmbeddingsExtractingTransformer.PretrainedModelKind modelKind = WordEmbeddingsExtractingTransformer.PretrainedModelKind.Sswe,
           params WordEmbeddingsExtractingTransformer.ColumnInfo[] columns)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), modelKind, columns);

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
            params WordTokenizingTransformer.ColumnInfo[] columns)
          => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumn"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LpNormalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/NgramExtraction.cs?range=1-5,11-74)]
        /// ]]>
        /// </format>
        /// </example>
        public static NgramCountingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int ngramLength = NgramCountingEstimator.Defaults.NgramLength,
            int skipLength = NgramCountingEstimator.Defaults.SkipLength,
            bool allLengths = NgramCountingEstimator.Defaults.AllLength,
            int maxNumTerms = NgramCountingEstimator.Defaults.MaxNumTerms,
            NgramCountingEstimator.WeightingCriteria weighting = NgramCountingEstimator.Defaults.Weighting) =>
            new NgramCountingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn,
                ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static NgramCountingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
            (string input, string output)[] columns,
            int ngramLength = NgramCountingEstimator.Defaults.NgramLength,
            int skipLength = NgramCountingEstimator.Defaults.SkipLength,
            bool allLengths = NgramCountingEstimator.Defaults.AllLength,
            int maxNumTerms = NgramCountingEstimator.Defaults.MaxNumTerms,
            NgramCountingEstimator.WeightingCriteria weighting = NgramCountingEstimator.Defaults.Weighting)
            => new NgramCountingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns,
                ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the ngram process on.</param>
        public static NgramCountingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
             params NgramCountingTransformer.ColumnInfo[] columns)
          => new NgramCountingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

    }
}
