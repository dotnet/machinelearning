// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML
{
    using CharTokenizingDefaults = TokenizingByCharactersEstimator.Defaults;
    using TextNormalizeDefaults = TextNormalizingEstimator.Defaults;

    /// <summary>
    /// Collection of extension methods for the <see cref="TransformsCatalog"/>.
    /// </summary>
    public static class TextCatalog
    {
        /// <summary>
        /// Create a <see cref="TextFeaturizingEstimator"/>, which transforms a text column into a featurized vector of <see cref="System.Single"/> that represents normalized counts of n-grams and char-grams.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of <see cref="System.Single"/>. </param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over text data.
        /// </param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/FeaturizeText.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName);

        /// <summary>
        ///  Create a <see cref="TextFeaturizingEstimator"/>, which transforms a text column into featurized vector of <see cref="System.Single"/> that represents normalized counts of n-grams and char-grams.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.
        /// This column's data type will be a vector of <see cref="System.Single"/>.
        /// </param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <param name="inputColumnNames">Name of the columns to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over text data, and it can transform several columns at once, yielding one vector of <see cref="System.Single"/>
        /// as the resulting features for all columns.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/FeaturizeTextWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            TextFeaturizingEstimator.Options options,
            params string[] inputColumnNames)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, Utils.Size(inputColumnNames) == 0 ? new[] { outputColumnName } : inputColumnNames,
                options);

        /// <summary>
        /// Create a <see cref="TokenizingByCharactersEstimator"/>, which tokenizes by splitting text into sequences of characters
        /// using a sliding window.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a variable-sized vector of keys.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the
        /// <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over text data type.</param>
        /// <param name="useMarkerCharacters">To be able to distinguish the tokens, for example for debugging purposes,
        /// you can choose to prepend a marker character, <see langword="0x02"/>, to the beginning,
        /// and append another marker character, <see langword="0x03"/>, to the end of the output vector of characters.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[TokenizeIntoCharacters](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/TokenizeIntoCharactersAsKeys.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TokenizingByCharactersEstimator TokenizeIntoCharactersAsKeys(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters)
            => new TokenizingByCharactersEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                useMarkerCharacters, new[] { (outputColumnName, inputColumnName) });

        /// <summary>
        /// Create a <see cref="TokenizingByCharactersEstimator"/>, which tokenizes incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="useMarkerCharacters">Whether to prepend a marker character, <see langword="0x02"/>, to the beginning,
        /// and append another marker character, <see langword="0x03"/>, to the end of the output vector of characters.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        [BestFriend]
        internal static TokenizingByCharactersEstimator TokenizeIntoCharactersAsKeys(this TransformsCatalog.TextTransforms catalog,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters,
            params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new TokenizingByCharactersEstimator(env, useMarkerCharacters, InputOutputColumnPair.ConvertToValueTuples(columns));
        }

        /// <summary>
        /// Creates a <see cref="TextNormalizingEstimator"/>, which normalizes incoming text in <paramref name="inputColumnName"/> by optionally
        /// changing case, removing diacritical marks, punctuation marks, numbers, and outputs new text as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type is a scalar or a vector of text depending on the input column data type.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>,
        /// the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates on text or vector of text data types.</param>
        /// <param name="caseMode">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[NormalizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/NormalizeText.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TextNormalizingEstimator NormalizeText(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            TextNormalizingEstimator.CaseMode caseMode = TextNormalizeDefaults.Mode,
            bool keepDiacritics = TextNormalizeDefaults.KeepDiacritics,
            bool keepPunctuations = TextNormalizeDefaults.KeepPunctuations,
            bool keepNumbers = TextNormalizeDefaults.KeepNumbers)
            => new TextNormalizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName, caseMode, keepDiacritics, keepPunctuations, keepNumbers);

        /// <summary>
        /// Create an <see cref="WordEmbeddingEstimator"/>, which is a text featurizer that converts a vector
        /// of text into a numerical vector using pre-trained embeddings models.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>,
        /// the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over known-size vector of text data type.</param>
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingEstimator.PretrainedModelKind"/> to use. </param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ApplyWordEmbedding](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/ApplyWordEmbedding.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static WordEmbeddingEstimator ApplyWordEmbedding(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            WordEmbeddingEstimator.PretrainedModelKind modelKind = WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding)
            => new WordEmbeddingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, modelKind);

        /// <summary>
        /// Create an <see cref="WordEmbeddingEstimator"/>, which is a text featurizer that converts vectors
        /// of text into numerical vectors using pre-trained embeddings models.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="customModelFile">The path of the pre-trained embeddings model to use.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>,
        /// the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over known-size vector of text data type.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ApplyWordEmbedding](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/ApplyCustomWordEmbedding.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static WordEmbeddingEstimator ApplyWordEmbedding(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string customModelFile,
            string inputColumnName = null)
            => new WordEmbeddingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, customModelFile, inputColumnName ?? outputColumnName);

        /// <summary>
        /// Create an <see cref="WordEmbeddingEstimator"/>, which is a text featurizer that converts vectors
        /// of text into numerical vectors using pre-trained embeddings models.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="modelKind">The embeddings <see cref="WordEmbeddingEstimator.PretrainedModelKind"/> to use. </param>
        /// <param name="columns">The array columns, and per-column configurations to extract embeddings from.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FeaturizeText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/WordEmbeddingTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        [BestFriend]
        internal static WordEmbeddingEstimator ApplyWordEmbedding(this TransformsCatalog.TextTransforms catalog,
           WordEmbeddingEstimator.PretrainedModelKind modelKind = WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding,
           params WordEmbeddingEstimator.ColumnOptions[] columns)
            => new WordEmbeddingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), modelKind, columns);

        /// <summary>
        /// Create a <see cref="WordTokenizingEstimator"/>, which tokenizes input text using <paramref name="separators"/> as separators.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a variable-size vector of text.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates on scalar of text and vector of text data type.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[TokenizeIntoWords](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/TokenizeIntoWords.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static WordTokenizingEstimator TokenizeIntoWords(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            char[] separators = null)
            => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, separators);

        /// <summary>
        ///  Tokenizes incoming text in input columns, using per-column configurations, and outputs the tokens.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        [BestFriend]
        internal static WordTokenizingEstimator TokenizeIntoWords(this TransformsCatalog.TextTransforms catalog,
            params WordTokenizingEstimator.ColumnOptions[] columns)
          => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        /// <summary>
        /// Creates a <see cref="NgramExtractingEstimator"/> which produces a vector of counts of n-grams (sequences of consecutive words)
        /// encountered in the input text.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over vectors of keys data type.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Number of tokens to skip between each n-gram. By default no token is skipped.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maximumNgramsCount">Maximum number of n-grams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word or n-gram is to a document in a corpus.
        /// When <paramref name="maximumNgramsCount"/> is smaller than the total number of encountered n-grams this measure is used
        /// to determine which n-grams to keep.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ProduceNgrams](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/ProduceNgrams.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool useAllLengths = NgramExtractingEstimator.Defaults.UseAllLengths,
            int maximumNgramsCount = NgramExtractingEstimator.Defaults.MaximumNgramsCount,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.Defaults.Weighting) =>
            new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName,
                ngramLength, skipLength, useAllLengths, maximumNgramsCount, weighting);

        /// <summary>
        /// Produces a bag of counts of n-grams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to run the n-gram process on.</param>
        [BestFriend]
        internal static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
             params NgramExtractingEstimator.ColumnOptions[] columns)
          => new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        /// <summary>
        /// Create a <see cref="CustomStopWordsRemovingEstimator"/>, which copies the data from the column specified in <paramref name="inputColumnName"/>
        /// to a new column: <paramref name="outputColumnName"/> and removes predifined set of text specific for <paramref name="language"/> from it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be variable-size vector of text.</param>
        /// <param name="inputColumnName">Name of the column to copy the data from.
        /// This estimator operates over vector of text.</param>
        /// <param name="language">Langauge of the input text column <paramref name="inputColumnName"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[RemoveDefaultStopWords](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/RemoveDefaultStopWords.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static StopWordsRemovingEstimator RemoveDefaultStopWords(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            StopWordsRemovingEstimator.Language language = StopWordsRemovingEstimator.Language.English)
            => new StopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, language);

        /// <summary>
        /// Create a <see cref="CustomStopWordsRemovingEstimator"/>, which copies the data from the column specified in <paramref name="inputColumnName"/>
        /// to a new column: <paramref name="outputColumnName"/> and removes text specified in <paramref name="stopwords"/> from it.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be variable-size vector of text.</param>
        /// <param name="inputColumnName">Name of the column to copy the data from.
        /// This estimator operates over a vector of text.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[RemoveStopWords](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/RemoveStopWords.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static CustomStopWordsRemovingEstimator RemoveStopWords(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            params string[] stopwords)
            => new CustomStopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), outputColumnName, inputColumnName, stopwords);

        /// <summary>
        /// Create a <see cref="WordBagEstimator"/>, which maps the column specified in <paramref name="inputColumnName"/>
        /// to a vector of n-gram counts in a new column named <paramref name="outputColumnName"/>.
        /// </summary>
        /// <remarks>
        /// <see cref="WordBagEstimator"/> is different from <see cref="NgramExtractingEstimator"/> in that the former
        /// tokenizes text internally and the latter takes tokenized text as input.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be known-size vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of the column to take the data from.
        /// This estimator operates over vector of text.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maximumNgramsCount">Maximum number of n-grams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool useAllLengths = NgramExtractingEstimator.Defaults.UseAllLengths,
            int maximumNgramsCount = NgramExtractingEstimator.Defaults.MaximumNgramsCount,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName, ngramLength, skipLength, useAllLengths, maximumNgramsCount, weighting);

        /// <summary>
        /// Create a <see cref="WordBagEstimator"/>, which maps the multiple columns specified in <paramref name="inputColumnNames"/>
        /// to a vector of n-gram counts in a new column named <paramref name="outputColumnName"/>.
        /// </summary>
        /// <remarks>
        /// <see cref="WordBagEstimator"/> is different from <see cref="NgramExtractingEstimator"/> in that the former
        /// tokenizes text internally and the latter takes tokenized text as input.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.
        /// This column's data type will be known-size vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnNames">Names of the multiple columns to take the data from.
        /// This estimator operates over vector of text.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maximumNgramsCount">Maximum number of n-grams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string[] inputColumnNames,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool useAllLengths = NgramExtractingEstimator.Defaults.UseAllLengths,
            int maximumNgramsCount = NgramExtractingEstimator.Defaults.MaximumNgramsCount,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnNames, ngramLength, skipLength, useAllLengths, maximumNgramsCount, weighting);

        /// <summary>
        /// Create a <see cref="WordHashBagEstimator"/>, which maps the column specified in <paramref name="inputColumnName"/>
        /// to a vector of counts of hashed n-grams in a new column named <paramref name="outputColumnName"/>.
        /// </summary>
        /// <remarks>
        /// <see cref="WordHashBagEstimator"/> is different from <see cref="NgramHashingEstimator"/> in that the former
        /// tokenizes text internally and the latter takes tokenized text as input.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be known-size vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of the column to take the data from.
        /// This estimator operates over vector of text.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we construct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  annotations for the new column. Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int numberOfBits = NgramHashExtractingTransformer.DefaultArguments.NumberOfBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool useAllLengths = NgramHashExtractingTransformer.DefaultArguments.UseAllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool useOrderedHashing = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int maximumNumberOfInverts = NgramHashExtractingTransformer.DefaultArguments.MaximumNumberOfInverts)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnName, numberOfBits: numberOfBits, ngramLength: ngramLength,
                skipLength: skipLength, useAllLengths: useAllLengths, seed: seed, useOrderedHashing: useOrderedHashing,
                maximumNumberOfInverts: maximumNumberOfInverts);

        /// <summary>
        /// Create a <see cref="WordHashBagEstimator"/>, which maps the multiple columns specified in <paramref name="inputColumnNames"/>
        /// to a vector of counts of hashed n-grams in a new column named <paramref name="outputColumnName"/>.
        /// </summary>
        /// <remarks>
        /// <see cref="WordHashBagEstimator"/> is different from <see cref="NgramHashingEstimator"/> in that the former
        /// tokenizes text internally and the latter takes tokenized text as input.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.
        /// This column's data type will be known-size vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnNames">Names of the multiple columns to take the data from.
        /// This estimator operates over vector of text.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we construct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  annotations for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string[] inputColumnNames,
            int numberOfBits = NgramHashExtractingTransformer.DefaultArguments.NumberOfBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool useAllLengths = NgramHashExtractingTransformer.DefaultArguments.UseAllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool useOrderedHashing = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int maximumNumberOfInverts = NgramHashExtractingTransformer.DefaultArguments.MaximumNumberOfInverts)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                outputColumnName, inputColumnNames, numberOfBits: numberOfBits, ngramLength: ngramLength,
                skipLength: skipLength, useAllLengths: useAllLengths, seed: seed, useOrderedHashing: useOrderedHashing,
                maximumNumberOfInverts: maximumNumberOfInverts);

        /// <summary>
        /// Create a <see cref="NgramHashingEstimator"/>, which copies the data from the column specified in <paramref name="inputColumnName"/>
        /// to a new column: <paramref name="outputColumnName"/> and produces a vector of counts of hashed n-grams.
        /// </summary>
        /// <remarks>
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of the column to copy the data from.
        /// This estimator operates over vector of key type.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we construct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  annotations for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <param name="rehashUnigrams">Whether to rehash unigrams.</param>
        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int numberOfBits = NgramHashingEstimator.Defaults.NumberOfBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool useAllLengths = NgramHashingEstimator.Defaults.UseAllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool useOrderedHashing = NgramHashingEstimator.Defaults.UseOrderedHashing,
            int maximumNumberOfInverts = NgramHashingEstimator.Defaults.MaximumNumberOfInverts,
            bool rehashUnigrams = NgramHashingEstimator.Defaults.RehashUnigrams)
            => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                new[] {new NgramHashingEstimator.ColumnOptions(outputColumnName, new[] { inputColumnName }, ngramLength: ngramLength, skipLength: skipLength,
                useAllLengths: useAllLengths, numberOfBits: numberOfBits, seed: seed, useOrderedHashing: useOrderedHashing, maximumNumberOfInverts: maximumNumberOfInverts, rehashUnigrams) });

        /// <summary>
        /// Create a <see cref="NgramHashingEstimator"/>, which takes the data from the multiple columns specified in <paramref name="inputColumnNames"/>
        /// to a new column: <paramref name="outputColumnName"/> and produces a vector of counts of hashed n-grams.
        /// </summary>
        /// <remarks>
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.
        /// This column's data type will be vector of known size of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnNames">Name of the multiple columns to take the data from.
        /// This estimator operates over vector of key type.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an n-gram.</param>
        /// <param name="useAllLengths">Whether to include all n-gram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we construct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  annotations for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <param name="rehashUnigrams">Whether to rehash unigrams.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ProduceHashedNgrams](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/ProduceHashedNgrams.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string[] inputColumnNames = null,
            int numberOfBits = NgramHashingEstimator.Defaults.NumberOfBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool useAllLengths = NgramHashingEstimator.Defaults.UseAllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool useOrderedHashing = NgramHashingEstimator.Defaults.UseOrderedHashing,
            int maximumNumberOfInverts = NgramHashingEstimator.Defaults.MaximumNumberOfInverts,
            bool rehashUnigrams = NgramHashingEstimator.Defaults.RehashUnigrams)
            => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                new[] {new NgramHashingEstimator.ColumnOptions(outputColumnName, inputColumnNames, ngramLength: ngramLength, skipLength: skipLength,
                useAllLengths: useAllLengths, numberOfBits: numberOfBits, seed: seed, useOrderedHashing: useOrderedHashing, maximumNumberOfInverts: maximumNumberOfInverts, rehashUnigrams) });

        /// <summary>
        /// Produces a bag of counts of hashed n-grams for each <paramref name="columns"/>. For each column,
        /// <see cref="NgramHashingEstimator.ColumnOptions.InputColumnNames"/> are the input columns of the output column named as <see cref="NgramHashingEstimator.ColumnOptions.Name"/>.
        ///
        /// <see cref="NgramHashingEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashingEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="catalog">The text-related transform's catalog.</param>
        /// <param name="columns">Pairs of columns to compute n-grams. Note that gram indices are generated by hashing.</param>
        [BestFriend]
        internal static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            NgramHashingEstimator.ColumnOptions[] columns)
             => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        /// <summary>
        /// Create a <see cref="LatentDirichletAllocationEstimator"/>, which uses <a href="https://arxiv.org/abs/1412.1576">LightLDA</a> to transform text (represented as a vector of floats)
        /// into a vector of <see cref="System.Single"/> indicating the similarity of the text with each topic identified.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This estimator outputs a vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over a vector of <see cref="System.Single"/>.
        /// </param>
        /// <param name="numberOfTopics">The number of topics.</param>
        /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
        /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
        /// <param name="samplingStepCount">Number of Metropolis Hasting step.</param>
        /// <param name="maximumNumberOfIterations">Number of iterations.</param>
        /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
        /// <param name="numberOfThreads">The number of training threads. Default value depends on number of logical processors.</param>
        /// <param name="maximumTokenCountPerDocument">The threshold of maximum count of tokens per doc.</param>
        /// <param name="numberOfSummaryTermsPerTopic">The number of words to summarize the topic.</param>
        /// <param name="numberOfBurninIterations">The number of burn-in iterations.</param>
        /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LatentDirichletAllocation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Text/LatentDirichletAllocation.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LatentDirichletAllocationEstimator LatentDirichletAllocation(this TransformsCatalog.TextTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            int numberOfTopics = LatentDirichletAllocationEstimator.Defaults.NumberOfTopics,
            float alphaSum = LatentDirichletAllocationEstimator.Defaults.AlphaSum,
            float beta = LatentDirichletAllocationEstimator.Defaults.Beta,
            int samplingStepCount = LatentDirichletAllocationEstimator.Defaults.SamplingStepCount,
            int maximumNumberOfIterations = LatentDirichletAllocationEstimator.Defaults.MaximumNumberOfIterations,
            int likelihoodInterval = LatentDirichletAllocationEstimator.Defaults.LikelihoodInterval,
            int numberOfThreads = LatentDirichletAllocationEstimator.Defaults.NumberOfThreads,
            int maximumTokenCountPerDocument = LatentDirichletAllocationEstimator.Defaults.MaximumTokenCountPerDocument,
            int numberOfSummaryTermsPerTopic = LatentDirichletAllocationEstimator.Defaults.NumberOfSummaryTermsPerTopic,
            int numberOfBurninIterations = LatentDirichletAllocationEstimator.Defaults.NumberOfBurninIterations,
            bool resetRandomGenerator = LatentDirichletAllocationEstimator.Defaults.ResetRandomGenerator)
            => new LatentDirichletAllocationEstimator(CatalogUtils.GetEnvironment(catalog),
                outputColumnName, inputColumnName, numberOfTopics, alphaSum, beta, samplingStepCount,
                maximumNumberOfIterations, numberOfThreads, maximumTokenCountPerDocument, numberOfSummaryTermsPerTopic,
                likelihoodInterval, numberOfBurninIterations, resetRandomGenerator);

        /// <summary>
        /// Uses <a href="https://arxiv.org/abs/1412.1576">LightLDA</a> to transform a document (represented as a vector of floats)
        /// into a vector of floats over a set of topics.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of LDA for each column pair.</param>
        [BestFriend]
        internal static LatentDirichletAllocationEstimator LatentDirichletAllocation(
            this TransformsCatalog.TextTransforms catalog,
            params LatentDirichletAllocationEstimator.ColumnOptions[] columns)
            => new LatentDirichletAllocationEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
