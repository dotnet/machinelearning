// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

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
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName);

        /// <summary>
        /// Transform several text columns into featurized float array that represents counts of ngrams and char-grams.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.</param>
        /// <param name="inputColumnNames">Name of the columns to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TextTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            IEnumerable<string> inputColumnNames,
            TextFeaturizingEstimator.Options options)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnNames, options);

        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumnName"/> and output the tokens as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        public static TokenizingByCharactersEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters)
            => new TokenizingByCharactersEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                useMarkerCharacters, new[] { (outputColumnName, inputColumnName) });

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>

        public static TokenizingByCharactersEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters,
            params (string outputColumnName, string inputColumnName)[] columns)
            => new TokenizingByCharactersEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), useMarkerCharacters, columns);

        /// <summary>
        /// Normalizes incoming text in <paramref name="inputColumnName"/> by changing case, removing diacritical marks, punctuation marks and/or numbers
        /// and outputs new text as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="textCase">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        public static TextNormalizingEstimator NormalizeText(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            TextNormalizingEstimator.CaseNormalizationMode textCase = TextNormalizeDefaults.TextCase,
            bool keepDiacritics = TextNormalizeDefaults.KeepDiacritics,
            bool keepPunctuations = TextNormalizeDefaults.KeepPunctuations,
            bool keepNumbers = TextNormalizeDefaults.KeepNumbers)
            => new TextNormalizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName, textCase, keepDiacritics, keepPunctuations, keepNumbers);

        /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingsExtractingEstimator.PretrainedModelKind"/> to use. </param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/WordEmbeddingTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            WordEmbeddingsExtractingEstimator.PretrainedModelKind modelKind = WordEmbeddingsExtractingEstimator.PretrainedModelKind.Sswe)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, modelKind);

        /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="customModelFile">The path of the pre-trained embeedings model to use. </param>
        /// <param name="inputColumnName">Name of the column to transform.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/WordEmbeddingTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string customModelFile,
            string inputColumnName = null)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, customModelFile, inputColumnName ?? outputColumnName);

        /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingsExtractingEstimator.PretrainedModelKind"/> to use. </param>
        /// <param name="columns">The array columns, and per-column configurations to extract embeedings from.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/WordEmbeddingTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
           WordEmbeddingsExtractingEstimator.PretrainedModelKind modelKind = WordEmbeddingsExtractingEstimator.PretrainedModelKind.Sswe,
           params WordEmbeddingsExtractingEstimator.ColumnInfo[] columns)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), modelKind, columns);

        /// <summary>
        /// Tokenizes incoming text in <paramref name="inputColumnName"/>, using <paramref name="separators"/> as separators,
        /// and outputs the tokens as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            char[] separators = null)
            => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, separators);

        /// <summary>
        /// Tokenizes incoming text in input columns and outputs the tokens using <paramref name="separators"/> as separators.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            (string outputColumnName, string inputColumnName)[] columns,
            char[] separators = null)
            => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, separators);

        /// <summary>
        ///  Tokenizes incoming text in input columns, using per-column configurations, and outputs the tokens.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            params WordTokenizingEstimator.ColumnInfo[] columns)
          => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumnName"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
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
        public static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.Defaults.Weighting) =>
            new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName,
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
        public static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
            (string outputColumnName, string inputColumnName)[] columns,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.Defaults.Weighting)
            => new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns,
                ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the ngram process on.</param>
        public static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
             params NgramExtractingEstimator.ColumnInfo[] columns)
          => new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumnName"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">The column containing output text. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">The column containing text to remove stop words on.</param>
        /// <param name="language">Langauge of the input text column <paramref name="inputColumnName"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/StopWordRemoverTransform.cs)]
        /// ]]></format>
        /// </example>
        public static StopWordsRemovingEstimator RemoveDefaultStopWords(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            StopWordsRemovingEstimator.Language language = StopWordsRemovingEstimator.Language.English)
            => new StopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, language);

        /// <summary>
        /// Removes stop words from incoming token streams in input columns
        /// and outputs the token streams without stop words as output columns.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to remove stop words on.</param>
        /// <param name="language">Langauge of the input text columns <paramref name="columns"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/StopWordRemoverTransform.cs)]
        /// ]]></format>
        /// </example>
        public static StopWordsRemovingEstimator RemoveDefaultStopWords(this TransformsCatalog.TextTransforms catalog,
            (string outputColumnName, string inputColumnName)[] columns,
             StopWordsRemovingEstimator.Language language = StopWordsRemovingEstimator.Language.English)
            => new StopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, language);

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumnName"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">The column containing output text. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">The column containing text to remove stop words on.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/StopWordRemoverTransform.cs)]
        /// ]]></format>
        /// </example>
        public static CustomStopWordsRemovingEstimator RemoveStopWords(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            params string[] stopwords)
            => new CustomStopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, stopwords);

        /// <summary>
        /// Removes stop words from incoming token streams in input columns
        /// and outputs the token streams without stop words as output columns.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to remove stop words on.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/StopWordRemoverTransform.cs)]
        /// ]]></format>
        /// </example>
        public static CustomStopWordsRemovingEstimator RemoveStopWords(this TransformsCatalog.TextTransforms catalog,
            (string outputColumnName, string inputColumnName)[] columns,
             params string[] stopwords)
            => new CustomStopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, stopwords);

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumnName"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName, ngramLength, skipLength, allLengths, maxNumTerms);

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumnNames"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.</param>
        /// <param name="inputColumnNames">Name of the columns to transform.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string[] inputColumnNames,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnNames, ngramLength, skipLength, allLengths, maxNumTerms, weighting);

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
        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            (string outputColumnName, string[] inputColumnNames)[] columns,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnName"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int hashBits = NgramHashExtractingTransformer.DefaultArguments.HashBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool allLengths = NgramHashExtractingTransformer.DefaultArguments.AllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool ordered = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int invertHash = NgramHashExtractingTransformer.DefaultArguments.InvertHash)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnNames"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.</param>
        /// <param name="inputColumnNames">Name of the columns to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string[] inputColumnNames,
            int hashBits = NgramHashExtractingTransformer.DefaultArguments.HashBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool allLengths = NgramHashExtractingTransformer.DefaultArguments.AllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool ordered = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int invertHash = NgramHashExtractingTransformer.DefaultArguments.InvertHash)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnNames, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            (string outputColumnName, string[] inputColumnNames)[] columns,
            int hashBits = NgramHashExtractingTransformer.DefaultArguments.HashBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool allLengths = NgramHashExtractingTransformer.DefaultArguments.AllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool ordered = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int invertHash = NgramHashExtractingTransformer.DefaultArguments.InvertHash)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
               columns, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnName"/>
        /// and outputs ngram vector as <paramref name="outputColumnName"/>
        ///
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int hashBits = NgramHashingEstimator.Defaults.HashBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool allLengths = NgramHashingEstimator.Defaults.AllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool ordered = NgramHashingEstimator.Defaults.Ordered,
            int invertHash = NgramHashingEstimator.Defaults.InvertHash)
            => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnNames"/>
        /// and outputs ngram vector as <paramref name="outputColumnName"/>
        ///
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.</param>
        /// <param name="inputColumnNames">Name of the columns to transform.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string[] inputColumnNames,
            int hashBits = NgramHashingEstimator.Defaults.HashBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool allLengths = NgramHashingEstimator.Defaults.AllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool ordered = NgramHashingEstimator.Defaults.Ordered,
            int invertHash = NgramHashingEstimator.Defaults.InvertHash)
             => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                 outputColumnName, inputColumnNames, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="columns.inputs"/>
        /// and outputs ngram vector for each output in <paramref name="columns.output"/>
        ///
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            (string outputColumnName, string[] inputColumnNames)[] columns,
            int hashBits = NgramHashingEstimator.Defaults.HashBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool allLengths = NgramHashingEstimator.Defaults.AllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool ordered = NgramHashingEstimator.Defaults.Ordered,
            int invertHash = NgramHashingEstimator.Defaults.InvertHash)
             => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                 columns, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <summary>
        /// Uses <a href="https://arxiv.org/abs/1412.1576">LightLDA</a> to transform a document (represented as a vector of floats)
        /// into a vector of floats over a set of topics.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="numTopic">The number of topics.</param>
        /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
        /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
        /// <param name="mhstep">Number of Metropolis Hasting step.</param>
        /// <param name="numIterations">Number of iterations.</param>
        /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
        /// <param name="numThreads">The number of training threads. Default value depends on number of logical processors.</param>
        /// <param name="numMaxDocToken">The threshold of maximum count of tokens per doc.</param>
        /// <param name="numSummaryTermPerTopic">The number of words to summarize the topic.</param>
        /// <param name="numBurninIterations">The number of burn-in iterations.</param>
        /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LatentDirichletAllocation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/LdaTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LatentDirichletAllocationEstimator LatentDirichletAllocation(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int numTopic = LatentDirichletAllocationEstimator.Defaults.NumTopic,
            float alphaSum = LatentDirichletAllocationEstimator.Defaults.AlphaSum,
            float beta = LatentDirichletAllocationEstimator.Defaults.Beta,
            int mhstep = LatentDirichletAllocationEstimator.Defaults.Mhstep,
            int numIterations = LatentDirichletAllocationEstimator.Defaults.NumIterations,
            int likelihoodInterval = LatentDirichletAllocationEstimator.Defaults.LikelihoodInterval,
            int numThreads = LatentDirichletAllocationEstimator.Defaults.NumThreads,
            int numMaxDocToken = LatentDirichletAllocationEstimator.Defaults.NumMaxDocToken,
            int numSummaryTermPerTopic = LatentDirichletAllocationEstimator.Defaults.NumSummaryTermPerTopic,
            int numBurninIterations = LatentDirichletAllocationEstimator.Defaults.NumBurninIterations,
            bool resetRandomGenerator = LatentDirichletAllocationEstimator.Defaults.ResetRandomGenerator)
            => new LatentDirichletAllocationEstimator(CatalogUtils.GetEnvironment(catalog),
                outputColumnName, inputColumnName, numTopic, alphaSum, beta, mhstep, numIterations, likelihoodInterval, numThreads,
                numMaxDocToken, numSummaryTermPerTopic, numBurninIterations, resetRandomGenerator);

        /// <summary>
        /// Uses <a href="https://arxiv.org/abs/1412.1576">LightLDA</a> to transform a document (represented as a vector of floats)
        /// into a vector of floats over a set of topics.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of LDA for each column pair.</param>
        public static LatentDirichletAllocationEstimator LatentDirichletAllocation(
            this TransformsCatalog.TextTransforms catalog,
            params LatentDirichletAllocationEstimator.ColumnInfo[] columns)
            => new LatentDirichletAllocationEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
