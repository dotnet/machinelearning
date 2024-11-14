// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// </summary>
    public sealed class EnglishRobertaTokenizer : Tokenizer
    {
        private readonly HighestOccurrenceMapping _vocabIdToHighestOccurrence;
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;
        private Dictionary<string, int>? _vocabOriginal;
        private readonly SortedDictionary<int, StringSpanOrdinalKey> _vocabReverse;
        private readonly Cache<(string, string), int> _mergeRanks;
        private readonly StringSpanOrdinalKeyCache<List<EncodedToken>> _cache;
        private readonly PreTokenizer? _preTokenizer;
        private readonly Normalizer? _normalizer;

        /// <summary>
        /// Indicate if want to filter the unsupported characters during the decoding.
        /// </summary>
        public bool FilterUnsupportedChars { get; }

        /// <summary>
        /// Create tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingPath">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary file is sourced from a trusted provider.
        /// </remarks>
        public static EnglishRobertaTokenizer Create(
                string vocabularyPath,
                string mergePath,
                string highestOccurrenceMappingPath)
            => new EnglishRobertaTokenizer(vocabularyPath, mergePath, highestOccurrenceMappingPath);

        /// <summary>
        /// Create tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingPath">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary file is sourced from a trusted provider.
        /// </remarks>
        public static EnglishRobertaTokenizer Create(
                string vocabularyPath,
                string mergePath,
                string highestOccurrenceMappingPath,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                bool filterUnsupportedChars = true)
            => new EnglishRobertaTokenizer(vocabularyPath, mergePath, highestOccurrenceMappingPath, preTokenizer, normalizer, filterUnsupportedChars);

        /// <summary>
        /// Create tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingStream">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static EnglishRobertaTokenizer Create(
                Stream vocabularyStream,
                Stream mergeStream,
                Stream highestOccurrenceMappingStream)
            => new EnglishRobertaTokenizer(vocabularyStream, mergeStream, highestOccurrenceMappingStream);


        /// <summary>
        /// Create tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingStream">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static EnglishRobertaTokenizer Create(
                Stream vocabularyStream,
                Stream mergeStream,
                Stream highestOccurrenceMappingStream,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                bool filterUnsupportedChars = true)
            => new EnglishRobertaTokenizer(vocabularyStream, mergeStream, highestOccurrenceMappingStream, preTokenizer, normalizer, filterUnsupportedChars);

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingPath">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        internal EnglishRobertaTokenizer(string vocabularyPath, string mergePath, string highestOccurrenceMappingPath, PreTokenizer? preTokenizer = null, Normalizer? normalizer = null, bool filterUnsupportedChars = true) :
            this(vocabularyPath is null ? throw new ArgumentNullException(nameof(vocabularyPath)) : File.OpenRead(vocabularyPath),
                mergePath is null ? throw new ArgumentNullException(nameof(mergePath)) : File.OpenRead(mergePath),
                highestOccurrenceMappingPath is null ? throw new ArgumentNullException(nameof(highestOccurrenceMappingPath)) : File.OpenRead(highestOccurrenceMappingPath),
                preTokenizer, normalizer, filterUnsupportedChars, disposeStream: true)
        {
        }

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingStream">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        internal EnglishRobertaTokenizer(Stream vocabularyStream, Stream mergeStream, Stream highestOccurrenceMappingStream, PreTokenizer? preTokenizer = null, Normalizer? normalizer = null, bool filterUnsupportedChars = true) :
            this(vocabularyStream, mergeStream, highestOccurrenceMappingStream, preTokenizer, normalizer, filterUnsupportedChars, disposeStream: false)
        {
        }

        private EnglishRobertaTokenizer(Stream vocabularyStream, Stream mergeStream, Stream highestOccurrenceMappingStream, PreTokenizer? preTokenizer, Normalizer? normalizer, bool filterUnsupportedChars, bool disposeStream)
        {
            if (vocabularyStream is null)
            {
                throw new ArgumentNullException(nameof(vocabularyStream));
            }

            if (mergeStream is null)
            {
                throw new ArgumentNullException(nameof(mergeStream));
            }

            if (highestOccurrenceMappingStream is null)
            {
                throw new ArgumentNullException(nameof(highestOccurrenceMappingStream));
            }

            FilterUnsupportedChars = filterUnsupportedChars;
            _preTokenizer = preTokenizer;
            _normalizer = normalizer;

            // vocabularyPath like "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
            // merge file like "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"
            // highestOccurrenceMappingPath like "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt"

            _vocabIdToHighestOccurrence = GetHighestOccurrenceMapping(highestOccurrenceMappingStream);
            _vocab = GetVocabulary(vocabularyStream);
            _vocabReverse = _vocab.ReverseSorted();
            _mergeRanks = GetMergeRanks(mergeStream);
            _cache = new StringSpanOrdinalKeyCache<List<EncodedToken>>();

            if (disposeStream)
            {
                vocabularyStream.Dispose();
                mergeStream.Dispose();
                highestOccurrenceMappingStream.Dispose();
            }
        }

        private static Dictionary<StringSpanOrdinalKey, int> GetVocabulary(Stream vocabularyStream)
        {
            Dictionary<StringSpanOrdinalKey, int>? vocab;
            try
            {
                vocab = JsonSerializer.Deserialize(vocabularyStream, ModelSourceGenerationContext.Default.DictionaryStringSpanOrdinalKeyInt32);
            }
            catch (Exception e)
            {
                throw new ArgumentException($"Problems met when parsing JSON vocabulary object.{Environment.NewLine}Error message: {e.Message}");
            }

            if (vocab is null)
            {
                throw new ArgumentException($"Failed to read the vocabulary file.");
            }

            return vocab;
        }

        private static Cache<(string, string), int> GetMergeRanks(Stream mergeStream)
        {
            var mergeRanks = new Cache<(string, string), int>(60_000);
            try
            {
                using StreamReader reader = new StreamReader(mergeStream);

                // We ignore the first and last line in the file
                if (reader.Peek() >= 0)
                {
                    string ignored = reader.ReadLine()!;
                }

                int rank = 1;
                while (reader.Peek() >= 0)
                {
                    string line = reader.ReadLine()!;
                    int index = line.IndexOf(' ');
                    if (index < 1 || index == line.Length - 1 || line.IndexOf(' ', index + 1) != -1)
                    {
                        throw new FormatException($"Invalid format of merge file at line: \"{line}\"");
                    }

                    mergeRanks.Set((line.Substring(0, index), line.Substring(index + 1)), rank++);
                }
            }
            catch (Exception e)
            {
                // Report any issues encountered while consuming a data file as IOExceptions.
                throw new IOException($"Cannot read the file Merge file.{Environment.NewLine}Error message: {e.Message}", e);
            }

            return mergeRanks;
        }

        private Dictionary<string, int> GetVocab()
        {
            Dictionary<string, int>? publicVocab = Volatile.Read(ref _vocabOriginal);
            if (publicVocab is null)
            {
                var vocab = new Dictionary<string, int>();
                foreach (var item in _vocab)
                {
                    vocab.Add(item.Key.ToString(), item.Value);
                }

                Interlocked.CompareExchange(ref _vocabOriginal, vocab, null);
                publicVocab = _vocabOriginal;
            }

            return publicVocab;
        }

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => _preTokenizer;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _normalizer;

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocabulary => GetVocab();

        //
        // Public Model interfaces implementation
        //

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the string.</param>
        /// <returns>The mapped token of the Id.</returns>
        private string? MapIdToToken(int id)
        {
            if (_vocabReverse.TryGetValue(id, out var value))
            {
                string v = value.Data!;
                if (FilterUnsupportedChars)
                {
                    char[] buffer = ArrayPool<char>.Shared.Rent(v.Length);
                    int i = 0;

                    IReadOnlyDictionary<char, char> unicodeToByte = ByteToUnicodeEncoding.Instance.UnicodeToByte;
                    for (int j = 0; j < v.Length; j++)
                    {
                        if (unicodeToByte.TryGetValue(v[j], out var c))
                        {
                            buffer[i++] = c;
                        }
                    }

                    string result = new string(buffer, 0, i);
                    ArrayPool<char>.Shared.Return(buffer);
                    return result;
                }
                else
                {
                    return v;
                }
            }

            return null;
        }

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                return new EncodeResults<EncodedToken> { Tokens = [], NormalizedText = null, CharsConsumed = 0 };
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                settings.ConsiderPreTokenization,
                                                                settings.ConsiderNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out string? normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out int charsConsumed);

            if (splits is not null)
            {
                List<EncodedToken> tokens = new();
                foreach ((int Offset, int Length) split in splits)
                {
                    foreach (EncodedToken t in EncodeInternal(textSpanToEncode.Slice(split.Offset, split.Length)))
                    {
                        tokens.Add(new EncodedToken(t.Id, t.Value, new Range(split.Offset + t.Offset.Start.Value, split.Offset + t.Offset.End.Value)));
                    }
                }

                return new EncodeResults<EncodedToken> { Tokens = tokens, NormalizedText = normalizedText, CharsConsumed = charsConsumed };
            }
            else
            {
                return new EncodeResults<EncodedToken> { Tokens = EncodeInternal(textSpanToEncode), NormalizedText = normalizedText, CharsConsumed = charsConsumed };
            }
        }

        /// <summary>
        /// Encode a text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        private IReadOnlyList<EncodedToken> EncodeInternal(ReadOnlySpan<char> text)
        {
            if (text.IsEmpty)
            {
                return [];
            }

            char[] token = ArrayPool<char>.Shared.Rent(text.Length);
            int[] indexMapping = ArrayPool<int>.Shared.Rent(text.Length);

            int newTokenIndex = 0;
            IReadOnlyDictionary<char, char> byteToUnicode = ByteToUnicodeEncoding.Instance.ByteToUnicode;

            for (int i = 0; i < text.Length; i++)
            {
                if (byteToUnicode.TryGetValue(text[i], out var value))
                {
                    token[newTokenIndex] = value;
                    indexMapping[newTokenIndex] = i;
                    newTokenIndex++;
                }
            }

            if (newTokenIndex == 0)
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                return [];
            }

            if (_cache.TryGetValue(text, out List<EncodedToken>? hit))
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                return ModifyTokenListOffsets(hit, indexMapping);
            }

            List<EncodedToken> result = EncodeToTokens(token.AsSpan().Slice(0, newTokenIndex), indexMapping);
            _cache.Set(text.ToString(), result);
            ArrayPool<char>.Shared.Return(token);
            ArrayPool<int>.Shared.Return(indexMapping);
            return result;
        }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The encoded results containing the list of encoded Ids.</returns>
        protected override EncodeResults<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => EncodeToIds(text, textSpan, settings.ConsiderPreTokenization, settings.ConsiderNormalization, settings.MaxTokenCount);

        private EncodeResults<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, bool considerPreTokenization, bool considerNormalization, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                return new EncodeResults<int> { Tokens = [], NormalizedText = null, CharsConsumed = 0 };
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                considerPreTokenization,
                                                                considerNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out string? normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            List<int> ids = new();
            int textLength = 0;

            if (splits is not null)
            {
                textLength = 0;
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeToIdsInternal(textSpanToEncode.Slice(split.Offset, split.Length), ids, out int length, maxTokenCount - ids.Count);
                    textLength = split.Offset + length;

                    if (length < split.Length || ids.Count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                EncodeToIdsInternal(textSpanToEncode, ids, out textLength, maxTokenCount);
            }

            return new EncodeResults<int> { Tokens = ids, NormalizedText = normalizedText, CharsConsumed = textLength };
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        protected override int CountTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => CountTokens(text, textSpan, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out _, out _, settings.MaxTokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <param name="fromEnd">Indicate whether to find the index from the end of the text.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="settings" /> has <see cref="EncodeSettings.ConsiderNormalization"/> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// If <paramRef name="fromEnd" /> is <see langword="false"/>, it represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the input text or the <paramref name="normalizedText"/> if the normalization is enabled.
        /// If <paramRef name="fromEnd" /> is <see langword="true"/>, it represents the index of the first character to be included. In cases where no tokens fit, the result will be the text length; conversely,
        /// if all tokens fit, the result will be zero.
        /// </returns>
        protected override int GetIndexByTokenCount(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings, bool fromEnd, out string? normalizedText, out int tokenCount)
        {
            if (fromEnd)
            {
                return LastIndexOf(text, textSpan, settings.MaxTokenCount, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out normalizedText, out tokenCount);
            }

            tokenCount = CountTokens(text, textSpan, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out normalizedText, out int charsConsumed, settings.MaxTokenCount);
            return charsConsumed;
        }

        private int CountTokens(string? text, ReadOnlySpan<char> textSpan, bool considerPreTokenization, bool considerNormalization, out string? normalizedText, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            charsConsumed = 0;
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedText = null;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                considerPreTokenization,
                                                                considerNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            int count = 0;
            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    count += EncodeToIdsInternal(textSpanToEncode.Slice(split.Offset, split.Length), null, out int length, maxTokenCount - count);
                    charsConsumed = split.Offset + length;

                    if (length < split.Length || count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                count += EncodeToIdsInternal(textSpanToEncode, null, out charsConsumed, maxTokenCount);
            }

            return count;
        }

        private int LastIndexOf(string? text, ReadOnlySpan<char> textSpan, int maxTokenCount, bool considerPreTokenization, bool considerNormalization, out string? normalizedText, out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedText = null;
                tokenCount = 0;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                considerPreTokenization,
                                                                considerNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            if (splits is not null)
            {
                tokenCount = 0;
                foreach ((int Offset, int Length) split in splits.Reverse())
                {
                    tokenCount += EncodeToIdsFromEndInternal(textSpanToEncode.Slice(split.Offset, split.Length), null, out int textIndex, maxTokenCount - tokenCount);
                    if (textIndex > 0 || tokenCount >= maxTokenCount)
                    {
                        return split.Offset + textIndex;
                    }
                }
            }
            else
            {
                tokenCount = EncodeToIdsFromEndInternal(textSpanToEncode, null, out int charsConsumed, maxTokenCount);
                return charsConsumed;
            }

            return 0;
        }

        private int EncodeToIdsResult(List<EncodedToken> tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int charsConsumed)
        {
            charsConsumed = 0;

            if (tokens.Count <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                charsConsumed = fullTextLength;
                return tokens.Count;
            }

            if (accumulatedIds is not null)
            {
                for (int i = 0; i < maxTokens; i++)
                {
                    accumulatedIds.Add(tokens[i].Id);
                    charsConsumed += tokens[i].Offset.End.Value - tokens[i].Offset.Start.Value;
                }
            }
            else
            {
                for (int i = 0; i < maxTokens; i++)
                {
                    charsConsumed += tokens[i].Offset.End.Value - tokens[i].Offset.Start.Value;
                }
            }

            return maxTokens;
        }

        private int EncodeToIdsFromEndResult(List<EncodedToken> tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int textIndex)
        {
            textIndex = fullTextLength;

            if (tokens.Count <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                textIndex = 0;
                return tokens.Count;
            }

            if (accumulatedIds is not null)
            {
                for (int i = tokens.Count - maxTokens; i < tokens.Count; i++)
                {
                    accumulatedIds.Add(tokens[i].Id);
                    textIndex -= tokens[i].Offset.End.Value - tokens[i].Offset.Start.Value;
                }
            }
            else
            {
                for (int i = tokens.Count - maxTokens; i < tokens.Count; i++)
                {
                    textIndex -= tokens[i].Offset.End.Value - tokens[i].Offset.Start.Value;
                }
            }

            return maxTokens;
        }

        private int EncodeToIdsInternal(ReadOnlySpan<char> text, IList<int>? accumulatedIds, out int charsConsumed, int maxTokens)
        {
            if (text.IsEmpty)
            {
                charsConsumed = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out List<EncodedToken>? hit))
            {
                return EncodeToIdsResult(hit, accumulatedIds, maxTokens, text.Length, out charsConsumed);
            }

            char[] token = ArrayPool<char>.Shared.Rent(text.Length);
            int[] indexMapping = ArrayPool<int>.Shared.Rent(text.Length);

            int newTokenIndex = 0;
            IReadOnlyDictionary<char, char> byteToUnicode = ByteToUnicodeEncoding.Instance.ByteToUnicode;

            for (int i = 0; i < text.Length; i++)
            {
                if (byteToUnicode.TryGetValue(text[i], out var value))
                {
                    token[newTokenIndex] = value;
                    indexMapping[newTokenIndex] = i;
                    newTokenIndex++;
                }
            }

            if (newTokenIndex == 0)
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                charsConsumed = text.Length;
                return 0;
            }

            List<EncodedToken> result = EncodeToTokens(token.AsSpan().Slice(0, newTokenIndex), indexMapping);
            _cache.Set(text.ToString(), result);
            ArrayPool<char>.Shared.Return(token);
            ArrayPool<int>.Shared.Return(indexMapping);

            return EncodeToIdsResult(result, accumulatedIds, maxTokens, text.Length, out charsConsumed);
        }

        private int EncodeToIdsFromEndInternal(ReadOnlySpan<char> text, IList<int>? accumulatedIds, out int textIndex, int maxTokens)
        {
            if (text.IsEmpty)
            {
                textIndex = text.Length;
                return 0;
            }

            if (_cache.TryGetValue(text, out List<EncodedToken>? hit))
            {
                return EncodeToIdsFromEndResult(hit, accumulatedIds, maxTokens, text.Length, out textIndex);
            }

            char[] token = ArrayPool<char>.Shared.Rent(text.Length);
            int[] indexMapping = ArrayPool<int>.Shared.Rent(text.Length);

            int newTokenIndex = 0;
            IReadOnlyDictionary<char, char> byteToUnicode = ByteToUnicodeEncoding.Instance.ByteToUnicode;

            for (int i = 0; i < text.Length; i++)
            {
                if (byteToUnicode.TryGetValue(text[i], out var value))
                {
                    token[newTokenIndex] = value;
                    indexMapping[newTokenIndex] = i;
                    newTokenIndex++;
                }
            }

            if (newTokenIndex == 0)
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                textIndex = 0;
                return 0;
            }

            List<EncodedToken> result = EncodeToTokens(token.AsSpan().Slice(0, newTokenIndex), indexMapping);
            _cache.Set(text.ToString(), result);
            ArrayPool<char>.Shared.Return(token);
            ArrayPool<int>.Shared.Return(indexMapping);

            return EncodeToIdsFromEndResult(result, accumulatedIds, maxTokens, text.Length, out textIndex);
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        private int? MapTokenToId(ReadOnlySpan<char> token) => _vocab.TryGetValue(token, out int value) ? value : null;

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string Decode(IEnumerable<int> ids)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            ValueStringBuilder sb = new ValueStringBuilder();

            foreach (int id in ids)
            {
                if (MapIdToToken(id) is string s)
                {
                    sb.Append(s);
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            Span<char> buffer = destination;

            idsConsumed = 0;
            charsWritten = 0;

            foreach (int id in ids)
            {
                if (MapIdToToken(id) is string s)
                {
                    if (s.Length > buffer.Length)
                    {
                        return OperationStatus.DestinationTooSmall;
                    }

                    s.AsSpan().CopyTo(buffer);
                    buffer = buffer.Slice(s.Length);
                    charsWritten += s.Length;
                }

                idsConsumed++;
            }

            return OperationStatus.Done;
        }

        /// <summary>
        /// Convert a list of token Ids to highest occurrence rankings.
        /// </summary>
        /// <param name="ids">The Ids list to map to the high occurrence rank.</param>
        /// <returns>The list of ranks mapped from the list of Ids.</returns>
        public IReadOnlyList<int> ConvertIdsToOccurrenceRanks(IReadOnlyList<int> ids)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            List<int> list = new List<int>(ids.Count);

            foreach (int id in ids)
            {
                list.Add(id <= 0 ? -id : _vocabIdToHighestOccurrence.IdToOccurrenceRank(id));
            }

            return list;
        }

        /// <summary>
        /// Convert a list of token Ids to highest occurrence values.
        /// </summary>
        /// <param name="ids">The Ids list to map to the high occurrence values.</param>
        /// <returns>The list of occurrence values mapped from the list of Ids.</returns>
        public IReadOnlyList<int> ConvertIdsToOccurrenceValues(IReadOnlyList<int> ids)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            List<int> list = new List<int>(ids.Count);

            foreach (int id in ids)
            {
                list.Add(id <= 0 ? 0 : _vocabIdToHighestOccurrence.IdToOccurrenceValue(id));
            }

            return list;
        }

        /// <summary>
        /// Convert a list of highest occurrence rankings to token Ids list .
        /// </summary>
        /// <param name="ranks">The high occurrence ranks list to map to the Ids list.</param>
        /// <returns>The list of Ids mapped from the list of ranks.</returns>
        public IReadOnlyList<int> ConvertOccurrenceRanksToIds(IReadOnlyList<int> ranks)
        {
            if (ranks is null)
            {
                throw new ArgumentNullException(nameof(ranks));
            }

            List<int> list = new List<int>(ranks.Count);

            foreach (int rank in ranks)
            {
                list.Add(_vocabIdToHighestOccurrence.ConvertOccurrenceRankToId(rank));
            }

            return list;
        }

        /// <summary>
        /// Gets the index of the pad symbol inside the symbols list.
        /// </summary>
        public int PadIndex => _vocabIdToHighestOccurrence.PadIndex;

        /// <summary>
        /// Gets the symbols list length.
        /// </summary>
        public int SymbolsCount => _vocabIdToHighestOccurrence.Count;

        /// <summary>
        /// Add the mask symbol to the symbols list.
        /// </summary>
        /// <param name="mask">The mask symbol.</param>
        /// <returns>The index of the mask symbol in the symbols list.</returns>
        public int AddMaskSymbol(string mask = "<mask>") => _vocabIdToHighestOccurrence.AddMaskSymbol(mask);

        //
        // Private & Internal methods
        //

        private IReadOnlyList<EncodedToken> ModifyTokenListOffsets(IReadOnlyList<EncodedToken> tokens, Span<int> indexMapping)
        {
            int index = 0;

            for (int i = 0; i < tokens.Count; i++)
            {
                Debug.Assert(index + tokens[i].Value.Length <= indexMapping.Length);

                if (tokens[i].Offset.Start.Value != indexMapping[index] || tokens[i].Offset.End.Value != indexMapping[index] + tokens[i].Value.Length)
                {
                    List<EncodedToken> list = new List<EncodedToken>(tokens.Count);
                    for (int j = 0; j < i; j++)
                    {
                        list.Add(tokens[j]);
                    }

                    for (int j = i; j < tokens.Count; j++)
                    {
                        list.Add(new EncodedToken(tokens[j].Id, tokens[j].Value, new Range(indexMapping[index], indexMapping[index] + tokens[j].Value.Length)));
                        index += tokens[j].Value.Length;
                    }

                    return list;
                }

                index += tokens[i].Value.Length;
            }

            return tokens;
        }

        private static HighestOccurrenceMapping GetHighestOccurrenceMapping(Stream highestOccurrenceMappingStream) =>
            HighestOccurrenceMapping.Load(highestOccurrenceMappingStream);

        /// <summary>
        /// Encode a token into BPE-ed sub-tokens. E.g., "playing" into ["play", "ing"].
        /// </summary>
        private List<EncodedToken> EncodeToTokens(Span<char> token, Span<int> indexMapping)
        {
            if (token.Length == 0)
            {
                return [];
            }

            string[] charToString = ByteToUnicodeEncoding.Instance.CharToString;

            if (token.Length == 1)
            {
                Debug.Assert(token[0] < charToString.Length);
                string tokenValue = charToString[token[0]];
                return new List<EncodedToken> { new EncodedToken(_vocab[new StringSpanOrdinalKey(tokenValue)], tokenValue, new Range(indexMapping[0], indexMapping[0] + 1)) };
            }

            List<string> word = new(token.Length);
            foreach (char c in token)
            {
                Debug.Assert(c < charToString.Length);
                word.Add(charToString[c]);
            }

            HashSet<(string, string)> pairs = new();

            WordToPairs(word, pairs);

            var newWord = new List<string>();

            Debug.Assert(pairs.Count != 0, "Pairs should not be empty.");

            while (true)
            {
                /* while conditions */
                // if only one element left, merge is finished (with the whole word merged)
                if (word.Count == 1)
                {
                    break;
                }

                // get the most frequent bi-gram pair
                var (first, second) = pairs.ArgMin(pair => _mergeRanks.GetOrAdd(pair, int.MaxValue));
                if (!_mergeRanks.TryGetValue((first, second), out int _))
                {
                    break;
                }
                /* end while conditions */

                // search and merge all (first, second) pairs in {word}
                var i = 0;
                while (i < word.Count)
                {
                    // find the next occurrence of {first} and add the elements before into {newWord}
                    var j = word.IndexOf(first, i);
                    if (j == -1)
                    {
                        // Equivalent to newWord.AddRange(word.Skip(i)) without allocations
                        for (int k = i; k < word.Count; k++)
                        {
                            newWord.Add(word[k]);
                        }

                        break;
                    }
                    else
                    {
                        // Equivalent to newWord.AddRange(word.Skip(i).Take(j - i)) without allocations
                        for (int k = i; k < j; k++)
                        {
                            newWord.Add(word[k]);
                        }

                        i = j;
                    }

                    // check the next element is {second} or not
                    if (i < word.Count - 1 && word[i + 1] == second)
                    {
                        newWord.Add(first + second);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i += 1;
                    }
                }

                List<string> temp = word;
                word = newWord;
                newWord = temp;
                newWord.Clear();

                // otherwise, continue merging
                WordToPairs(word, pairs);
            }

            var tokens = new List<EncodedToken>(word.Count);
            int index = 0;

            foreach (string w in word)
            {
                tokens.Add(new EncodedToken(_vocab[new StringSpanOrdinalKey(w)], w, new Range(indexMapping[index], indexMapping[index] + w.Length)));
                index += w.Length;
            }

            return tokens;
        }

        /// <summary>
        /// Extract element pairs in an aggregating word. E.g. [p, l, ay] into [(p,l), (l,ay)].
        /// If word contains 0 or 1 element, an empty HashSet will be returned.
        /// </summary>
        private static void WordToPairs(IReadOnlyList<string> word, HashSet<(string, string)> pairs)
        {
            pairs.Clear();

            if (word.Count <= 1)
            {
                return;
            }

            var prevElem = word[0];
            foreach (var elem in word.Skip(1))
            {
                pairs.Add((prevElem, elem));
                prevElem = elem;
            }
        }

        /// <summary>
        /// Check if the character is supported by the tokenizer's model.
        /// </summary>
        /// <param name="ch">The character to check.</param>
        /// <returns>True if the character is supported, otherwise false.</returns>
        public bool IsSupportedChar(char ch) => ByteToUnicodeEncoding.Instance.ByteToUnicode.ContainsKey(ch);
    }

    /// <summary>
    /// HighestOccurrenceMapping maps the GPT-2 vocabulary Id to highest occurrence value came from dict.txt file
    /// </summary>
    internal sealed class HighestOccurrenceMapping
    {
        public const int NumSpecialSymbols = 4;

        public string? PadWord { get; }
        public string? EosWord { get; }
        public string? UnkWord { get; }
        public string? BosWord { get; }

        public int PadIndex { get; }
        public int EosIndex { get; }
        public int UnkIndex { get; }
        public int BosIndex { get; }

        public string? MaskWord { get; private set; }
        public int MaskIndex { get; private set; }

        private readonly List<(int Id, int OccurrenceScore)> _symbols;
        private readonly Dictionary<int, int> _idToIndex;
        private readonly Dictionary<string, int> _stringSymbolToIndexMapping;

        /// <exception cref="ArgumentNullException">Any of `pad`, `eos`, `unk` and `bos` is `null`.</exception>
        public HighestOccurrenceMapping(string pad = "<pad>", string eos = "</s>", string unk = "<unk>", string bos = "<s>", string[]? extraSpecialSymbols = null)
        {
            _idToIndex = new Dictionary<int, int>();
            _symbols = new List<(int, int)>();
            _stringSymbolToIndexMapping = new Dictionary<string, int>();

            BosWord = bos;
            PadWord = pad;
            EosWord = eos;
            UnkWord = unk;
            BosIndex = ReserveStringSymbolSlot(bos);
            PadIndex = ReserveStringSymbolSlot(pad);
            EosIndex = ReserveStringSymbolSlot(eos);
            UnkIndex = ReserveStringSymbolSlot(unk);

            if (extraSpecialSymbols is not null)
            {
                foreach (var symbol in extraSpecialSymbols)
                {
                    ReserveStringSymbolSlot(symbol);
                }
            }
        }

        public int IdToOccurrenceRank(int id)
        {
            if ((uint)id <= NumSpecialSymbols)
                return id;

            return _idToIndex.TryGetValue(id, out int rank) ? rank : UnkIndex;
        }

        public int IdToOccurrenceValue(int id)
        {
            if ((uint)id <= NumSpecialSymbols)
                return 0;

            if (_idToIndex.TryGetValue(id, out int rank))
            {
                Debug.Assert(rank < _symbols.Count);
                return _symbols[rank].OccurrenceScore;
            }

            return 0;
        }

        public int ConvertOccurrenceRankToId(int rank)
        {
            if ((uint)rank >= _symbols.Count)
            {
                return UnkIndex;
            }

            return _symbols[rank].Id;
        }

        private int ReserveStringSymbolSlot(string symbol, int defaultOccurrence = -1)
        {
            if (symbol is null)
            {
                throw new ArgumentNullException(nameof(symbol), $"argument {nameof(symbol)} should not be null.");
            }

            if (!_stringSymbolToIndexMapping.TryGetValue(symbol, out int idx))
            {
                idx = _symbols.Count;
                _symbols.Add((-1, defaultOccurrence));
                _stringSymbolToIndexMapping[symbol] = idx;
            }

            return idx;
        }

        public int AddSymbol(int id, int highOccurrenceScore)
        {
            if (!_idToIndex.TryGetValue(id, out int idx))
            {
                idx = _symbols.Count;
                _symbols.Add((id, highOccurrenceScore));
                _idToIndex[id] = idx;
            }

            return idx;
        }

        public int AddMaskSymbol(string mask = "<mask>")
        {
            MaskWord = mask;
            MaskIndex = ReserveStringSymbolSlot(mask, 1);
            return MaskIndex;
        }

        /// <exception cref="ArgumentOutOfRangeException">`idx` is negative.</exception>
        public int this[int idx]
        {
            get
            {
                if (idx < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(idx), $"Index should be non-negative, got {idx}.");
                }

                return idx < _symbols.Count ? _symbols[idx].Id : UnkIndex;
            }
        }

        public int Count => _symbols.Count;

        public bool Equals(HighestOccurrenceMapping other) => _idToIndex.SequenceEqual(other._idToIndex);

        public bool Contains(string symbol) => symbol != null && _stringSymbolToIndexMapping.ContainsKey(symbol);

        public bool Contains(int id) => _idToIndex.ContainsKey(id);

        /// <exception cref="ArgumentNullException">`symbol` is `null`.</exception>
        public int IndexOf(int id) => _idToIndex.ContainsKey(id) ? _idToIndex[id] : UnkIndex;

        /// <summary>
        /// Loads the mapping from a text file with the format:
        ///     13 850314647
        ///     262 800385005
        ///     11 800251374
        ///     284 432911125
        ///     ...
        /// </summary>
        public static HighestOccurrenceMapping Load(Stream stream)
        {
            var mapping = new HighestOccurrenceMapping();
            mapping.AddFromStream(stream);
            return mapping;
        }

        /// <summary>
        /// Loads a pre-existing vocabulary from a text stream and adds its symbols to this instance.
        /// </summary>
        public void AddFromStream(Stream stream)
        {
            Debug.Assert(stream is not null);
            using StreamReader reader = new StreamReader(stream);

            while (reader.Peek() >= 0)
            {
                string? line = reader.ReadLine();
                if (line is null)
                {
                    continue;
                }

                var splitLine = line.Trim().Split(' ');
                if (splitLine.Length != 2)
                {
                    throw new ArgumentException("Incorrect vocabulary format, expected \"<token> <cnt>\"");
                }

                if (!int.TryParse(splitLine[1], out int occurrenceScore))
                {
                    throw new ArgumentException($"Cannot parse the line: '{line}'.");
                }

                if (!int.TryParse(splitLine[0], out var id))
                {
                    ReserveStringSymbolSlot(splitLine[0], occurrenceScore);
                }
                else
                {
                    AddSymbol(id, occurrenceScore);
                }
            }
        }
    }
}
