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
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Tokenizer for Bert model.
    /// </summary>
    /// <remarks>
    /// The BertTokenizer is a based on the WordPieceTokenizer and is used to tokenize text for Bert models.
    /// The implementation of the BertTokenizer is based on the original Bert implementation in the Hugging Face Transformers library.
    /// https://huggingface.co/transformers/v3.0.2/model_doc/bert.html?highlight=berttokenizerfast#berttokenizer
    /// </remarks>
    public sealed partial class BertTokenizer : WordPieceTokenizer
    {
        internal BertTokenizer(
                    Dictionary<StringSpanOrdinalKey, int> vocab,
                    Dictionary<int, string> vocabReverse,
                    BertOptions options) : base(vocab, vocabReverse, options)
        {
            Debug.Assert(options is not null);

            LowerCaseBeforeTokenization = options!.LowerCaseBeforeTokenization;
            ApplyBasicTokenization = options.ApplyBasicTokenization;
            SplitOnSpecialTokens = options.SplitOnSpecialTokens;

            SeparatorToken = options.SeparatorToken;
            SeparatorTokenId = vocab[new StringSpanOrdinalKey(options.SeparatorToken)];

            PaddingToken = options.PaddingToken;
            PaddingTokenId = vocab[new StringSpanOrdinalKey(options.PaddingToken)];

            ClassificationToken = options.ClassificationToken;
            ClassificationTokenId = vocab[new StringSpanOrdinalKey(options.ClassificationToken)];

            MaskingToken = options.MaskingToken;
            MaskingTokenId = vocab[new StringSpanOrdinalKey(options.MaskingToken)];

            IndividuallyTokenizeCjk = options.IndividuallyTokenizeCjk;
            RemoveNonSpacingMarks = options.RemoveNonSpacingMarks;
        }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should lowercase the input text.
        /// </summary>
        public bool LowerCaseBeforeTokenization { get; }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should do basic tokenization. Like clean text, normalize it, lowercasing, etc.
        /// </summary>
        public bool ApplyBasicTokenization { get; }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should split on the special tokens or treat special tokens as normal text.
        /// </summary>
        public bool SplitOnSpecialTokens { get; }

        /// <summary>
        /// Gets the separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering.
        /// It is also used as the last token of a sequence built with special tokens.
        /// </summary>
        public string SeparatorToken { get; }

        /// <summary>
        /// Gets the separator token Id
        /// </summary>
        public int SeparatorTokenId { get; }

        /// <summary>
        /// Gets the token used for padding, for example when batching sequences of different lengths
        /// </summary>
        public string PaddingToken { get; }

        /// <summary>
        /// Gets padding token Id
        /// </summary>
        public int PaddingTokenId { get; }

        /// <summary>
        /// Gets the classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification).
        /// It is the first token of the sequence when built with special tokens.
        /// </summary>
        public string ClassificationToken { get; }

        /// <summary>
        /// Gets the classifier token Id
        /// </summary>
        public int ClassificationTokenId { get; }

        /// <summary>
        /// Gets the mask token used for masking values. This is the token used when training this model with masked language modeling.
        /// This is the token which the model will try to predict.
        /// </summary>
        public string MaskingToken { get; }

        /// <summary>
        /// Gets the mask token Id
        /// </summary>
        public int MaskingTokenId { get; }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should split the CJK characters into tokens.
        /// </summary>
        /// <remarks>
        /// This is useful when you want to tokenize CJK characters individually.
        /// The following Unicode ranges are considered CJK characters for this purpose:
        /// - U+3400 - U+4DBF   CJK Unified Ideographs Extension A.
        /// - U+4E00 - U+9FFF   basic set of CJK characters.
        /// - U+F900 - U+FAFF   CJK Compatibility Ideographs.
        /// - U+20000 - U+2A6DF CJK Unified Ideographs Extension B.
        /// - U+2A700 - U+2B73F CJK Unified Ideographs Extension C.
        /// - U+2B740 - U+2B81F CJK Unified Ideographs Extension D.
        /// - U+2B820 - U+2CEAF CJK Unified Ideographs Extension E.
        /// - U+2F800 - U+2FA1F CJK Compatibility Ideographs Supplement.
        /// </remarks>
        public bool IndividuallyTokenizeCjk { get; }

        /// <summary>
        /// Gets a value indicating whether to remove non-spacing marks.
        /// </summary>
        public bool RemoveNonSpacingMarks { get; }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public new IReadOnlyList<int> EncodeToIds(string text, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(text, ReadOnlySpan<char>.Empty, addSpecialTokens: true, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public new IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(null, text, addSpecialTokens: true, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addSpecialTokens">Indicate whether to add special tokens to the encoded Ids.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool addSpecialTokens, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(text, ReadOnlySpan<char>.Empty, addSpecialTokens, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addSpecialTokens">Indicate whether to add special tokens to the encoded Ids.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addSpecialTokens, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(null, text, addSpecialTokens, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to return.</param>
        /// <param name="normalizedText">The normalized text.</param>
        /// <param name="charsConsumed">The number of characters consumed from the input text.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public new IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(text, ReadOnlySpan<char>.Empty, maxTokenCount, addSpecialTokens: true, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to return.</param>
        /// <param name="normalizedText">The normalized text.</param>
        /// <param name="charsConsumed">The number of characters consumed from the input text.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public new IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(null, text, maxTokenCount, addSpecialTokens: true, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to return.</param>
        /// <param name="addSpecialTokens">Indicate whether to add special tokens to the encoded Ids.</param>
        /// <param name="normalizedText">The normalized text.</param>
        /// <param name="charsConsumed">The number of characters consumed from the input text.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, bool addSpecialTokens, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(text, ReadOnlySpan<char>.Empty, maxTokenCount, addSpecialTokens, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to return.</param>
        /// <param name="addSpecialTokens">Indicate whether to add special tokens to the encoded Ids.</param>
        /// <param name="normalizedText">The normalized text.</param>
        /// <param name="charsConsumed">The number of characters consumed from the input text.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, bool addSpecialTokens, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true) =>
            EncodeToIds(null, text, maxTokenCount, addSpecialTokens, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization);

        private IReadOnlyList<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, int maxTokenCount, bool addSpecialTokens, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            if (addSpecialTokens)
            {
                if (maxTokenCount < 2)
                {
                    charsConsumed = 0;
                    normalizedText = null;
                    return Array.Empty<int>();
                }

                IReadOnlyList<int> ids = text is null ?
                                            base.EncodeToIds(textSpan, maxTokenCount - 2, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization) :
                                            base.EncodeToIds(text, maxTokenCount - 2, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization);

                if (ids is not List<int> list)
                {
                    list = new List<int>(ids);
                }

                list.Insert(0, ClassificationTokenId);
                list.Add(SeparatorTokenId);

                return list;
            }

            return text is null ?
                    base.EncodeToIds(textSpan, maxTokenCount, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization) :
                    base.EncodeToIds(text, maxTokenCount, out normalizedText, out charsConsumed, considerPreTokenization, considerNormalization);
        }

        private IReadOnlyList<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, bool addSpecialTokens, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            IReadOnlyList<int> ids = text is null ? base.EncodeToIds(textSpan, considerPreTokenization, considerNormalization) : base.EncodeToIds(text, considerPreTokenization, considerNormalization);

            if (addSpecialTokens)
            {
                if (ids is not List<int> list)
                {
                    list = new List<int>(ids);
                }

                list.Insert(0, ClassificationTokenId);
                list.Add(SeparatorTokenId);

                return list;
            }

            return ids;
        }

        /// <summary>
        /// Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:
        ///     - single sequence: `[CLS] tokenIds [SEP]`
        ///     - pair of sequences: `[CLS] tokenIds [SEP] additionalTokenIds [SEP]`
        /// </summary>
        /// <param name="tokenIds">List of IDs to which the special tokens will be added.</param>
        /// <param name="additionalTokenIds">Optional second list of IDs for sequence pairs.</param>
        /// <returns>The list of IDs with special tokens added.</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="tokenIds"/> is null.</exception>
        public IReadOnlyList<int> BuildInputsWithSpecialTokens(IEnumerable<int> tokenIds, IEnumerable<int>? additionalTokenIds = null)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            List<int> ids;

            if (tokenIds is ICollection<int> c1)
            {
                int capacity = c1.Count + 2;    // Add 2 for [CLS] and two [SEP] tokens.

                if (additionalTokenIds is not null)
                {
                    capacity += additionalTokenIds is ICollection<int> c2 ? c2.Count + 1 : c1.Count + 1;
                }

                ids = new(capacity) { ClassificationTokenId };
            }
            else
            {
                // slow path
                ids = new List<int>(10) { ClassificationTokenId };
            }

            ids.AddRange(tokenIds);
            ids.Add(SeparatorTokenId);

            if (additionalTokenIds is not null)
            {
                ids.AddRange(additionalTokenIds);
                ids.Add(SeparatorTokenId);
            }

            return ids;
        }

        /// <summary>
        /// Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:
        ///     - single sequence: `[CLS] tokenIds [SEP]`
        ///     - pair of sequences: `[CLS] tokenIds [SEP] additionalTokenIds [SEP]`
        /// </summary>
        /// <param name="tokenIds">List of IDs to which the special tokens will be added.</param>
        /// <param name="destination">The destination buffer to write the token IDs with special tokens added.</param>
        /// <param name="valuesWritten">The number of elements written to the destination buffer.</param>
        /// <param name="additionalTokenIds">Optional second list of IDs for sequence pairs.</param>
        /// <returns>The status of the operation.</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="tokenIds"/> is null.</exception>
        public OperationStatus BuildInputsWithSpecialTokens(IEnumerable<int> tokenIds, Span<int> destination, out int valuesWritten, IEnumerable<int>? additionalTokenIds = null)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            valuesWritten = 0;
            if (destination.Length < 1)
            {
                return OperationStatus.DestinationTooSmall;
            }

            destination[valuesWritten++] = ClassificationTokenId;
            foreach (int id in tokenIds)
            {
                if (destination.Length <= valuesWritten)
                {
                    valuesWritten = 0;
                    return OperationStatus.DestinationTooSmall;
                }

                destination[valuesWritten++] = id;
            }

            if (destination.Length <= valuesWritten)
            {
                valuesWritten = 0;
                return OperationStatus.DestinationTooSmall;
            }
            destination[valuesWritten++] = SeparatorTokenId;

            if (additionalTokenIds is not null)
            {
                foreach (int id in additionalTokenIds)
                {
                    if (destination.Length <= valuesWritten)
                    {
                        valuesWritten = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    destination[valuesWritten++] = id;
                }

                if (destination.Length <= valuesWritten)
                {
                    valuesWritten = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                destination[valuesWritten++] = SeparatorTokenId;
            }

            return OperationStatus.Done;
        }

        /// <summary>
        /// Retrieve sequence tokens mask from a IDs list.
        /// </summary>
        /// <param name="tokenIds">List of IDs.</param>
        /// <param name="additionalTokenIds">Optional second list of IDs for sequence pairs.</param>
        /// <param name="alreadyHasSpecialTokens">Indicate whether or not the token list is already formatted with special tokens for the model.</param>
        /// <returns>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</returns>
        /// <exception cref="ArgumentNullException"></exception>
        public IReadOnlyList<int> GetSpecialTokensMask(IEnumerable<int> tokenIds, IEnumerable<int>? additionalTokenIds = null, bool alreadyHasSpecialTokens = false)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            List<int> mask;
            if (tokenIds is ICollection<int> c1)
            {
                int capacity = c1.Count + 2;

                if (additionalTokenIds is not null)
                {
                    capacity += additionalTokenIds is ICollection<int> c2 ? c2.Count + 1 : c1.Count + 1;
                }

                mask = new List<int>(capacity);
            }
            else
            {
                mask = new List<int>(10);
            }

            if (!alreadyHasSpecialTokens)
            {
                mask.Add(1); // CLS
                mask.AddRange(Enumerable.Repeat(0, tokenIds.Count()));
                mask.Add(1); // SEP
                if (additionalTokenIds is not null)
                {
                    mask.AddRange(Enumerable.Repeat(0, additionalTokenIds.Count()));
                    mask.Add(1); // SEP
                }

                return mask;
            }

            foreach (int id in tokenIds)
            {
                mask.Add(id == ClassificationTokenId || id == SeparatorTokenId || id == PaddingTokenId || id == MaskingTokenId || id == UnknownTokenId ? 1 : 0);
            }

            if (additionalTokenIds is not null)
            {
                foreach (int id in additionalTokenIds)
                {
                    mask.Add(id == ClassificationTokenId || id == SeparatorTokenId || id == PaddingTokenId || id == MaskingTokenId || id == UnknownTokenId ? 1 : 0);
                }
            }

            return mask;
        }

        /// <summary>
        /// Retrieve sequence tokens mask from a IDs list.
        /// </summary>
        /// <param name="tokenIds">List of IDs.</param>
        /// <param name="destination">The destination buffer to write the mask. The integers written values are in the range [0, 1]: 1 for a special token, 0 for a sequence token.</param>
        /// <param name="valuesWritten">The number of elements written to the destination buffer.</param>
        /// <param name="additionalTokenIds">Optional second list of IDs for sequence pairs.</param>
        /// <param name="alreadyHasSpecialTokens">Indicate whether or not the token list is already formatted with special tokens for the model.</param>
        /// <returns>The status of the operation.</returns>
        /// <exception cref="ArgumentNullException"></exception>
        public OperationStatus GetSpecialTokensMask(IEnumerable<int> tokenIds, Span<int> destination, out int valuesWritten, IEnumerable<int>? additionalTokenIds = null, bool alreadyHasSpecialTokens = false)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            valuesWritten = 0;
            if (!alreadyHasSpecialTokens)
            {
                if (destination.Length < 1)
                {
                    return OperationStatus.DestinationTooSmall;
                }
                destination[valuesWritten++] = 1; // CLS

                foreach (int id in tokenIds)
                {
                    if (destination.Length <= valuesWritten)
                    {
                        valuesWritten = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    destination[valuesWritten++] = 0;
                }

                if (destination.Length <= valuesWritten)
                {
                    valuesWritten = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                destination[valuesWritten++] = 1; // SEP

                if (additionalTokenIds is not null)
                {
                    foreach (int id in additionalTokenIds)
                    {
                        if (destination.Length <= valuesWritten)
                        {
                            valuesWritten = 0;
                            return OperationStatus.DestinationTooSmall;
                        }
                        destination[valuesWritten++] = 0;
                    }

                    if (destination.Length <= valuesWritten)
                    {
                        valuesWritten = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    destination[valuesWritten++] = 1; // SEP
                }

                return OperationStatus.Done;
            }

            foreach (int id in tokenIds)
            {
                if (destination.Length <= valuesWritten)
                {
                    valuesWritten = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                destination[valuesWritten++] = id == ClassificationTokenId || id == SeparatorTokenId || id == PaddingTokenId || id == MaskingTokenId || id == UnknownTokenId ? 1 : 0;
            }

            if (additionalTokenIds is not null)
            {
                foreach (int id in additionalTokenIds)
                {
                    if (destination.Length <= valuesWritten)
                    {
                        valuesWritten = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    destination[valuesWritten++] = id == ClassificationTokenId || id == SeparatorTokenId || id == PaddingTokenId || id == MaskingTokenId || id == UnknownTokenId ? 1 : 0;
                }
            }

            return OperationStatus.Done;
        }

        /// <summary>
        /// Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence pair mask has the following format:
        ///         0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        ///         | first sequence    | second sequence |
        /// If <paramref name="additionalTokenIds"/> is null, this method only returns the first portion of the type ids (0s).
        /// </summary>
        /// <param name="tokenIds">List of token IDs for the first sequence.</param>
        /// <param name="additionalTokenIds">Optional list of token IDs for the second sequence.</param>
        /// <returns>List of token type IDs according to the given sequence(s).</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="tokenIds"/> is null.</exception>
        public IReadOnlyList<int> CreateTokenTypeIdsFromSequences(IEnumerable<int> tokenIds, IEnumerable<int>? additionalTokenIds = null)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            List<int> typeIds;
            if (tokenIds is ICollection<int> c1)
            {
                int capacity = c1.Count + 2;    // Add 2 for [CLS] and [SEP] tokens.

                if (additionalTokenIds is not null)
                {
                    capacity += additionalTokenIds is ICollection<int> c2 ? c2.Count + 1 : c1.Count + 1;
                }

                typeIds = new List<int>(capacity);
            }
            else
            {
                typeIds = new List<int>(10);
            }

            foreach (var id in tokenIds)
            {
                typeIds.Add(0);
            }
            typeIds.Add(0); // [CLS]
            typeIds.Add(0); // [SEP]

            if (additionalTokenIds is not null)
            {
                foreach (int id in additionalTokenIds)
                {
                    typeIds.Add(1);
                }

                typeIds.Add(1); // [SEP]
            }

            return typeIds;
        }

        public OperationStatus CreateTokenTypeIdsFromSequences(IEnumerable<int> tokenIds, Span<int> destination, out int valuesWritten, IEnumerable<int>? additionalTokenIds = null)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            valuesWritten = 0;

            // Add 2 for [CLS] and [SEP] tokens. Add 1 for [SEP] token if additionalTokenIds is not null.
            int capacity = tokenIds.Count() + 2 + (additionalTokenIds is null ? 0 : additionalTokenIds.Count() + 1);
            if (destination.Length < 2)
            {
                return OperationStatus.DestinationTooSmall;
            }
            destination[valuesWritten++] = 0; // [CLS]
            destination[valuesWritten++] = 0; // [SEP]

            foreach (int id in tokenIds)
            {
                if (destination.Length <= valuesWritten)
                {
                    valuesWritten = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                destination[valuesWritten++] = 0;
            }

            if (additionalTokenIds is not null)
            {
                foreach (int id in additionalTokenIds)
                {
                    if (destination.Length <= valuesWritten)
                    {
                        valuesWritten = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    destination[valuesWritten++] = 1;
                }

                if (destination.Length < valuesWritten)
                {
                    return OperationStatus.DestinationTooSmall;
                }
                destination[valuesWritten++] = 1; // [SEP]
            }

            return OperationStatus.Done;
        }

        /// <summary>
        /// Create a new instance of the <see cref="BertTokenizer"/> class.
        /// </summary>
        /// <param name="vocabFilePath">The path to the vocabulary file.</param>
        /// <param name="options">The options to use for the Bert tokenizer.</param>
        /// <returns>A new instance of the <see cref="BertTokenizer"/> class.</returns>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary file is sourced from a trusted provider.
        /// </remarks>
        public static BertTokenizer Create(
                    string vocabFilePath,
                    BertOptions? options = null) =>
            Create(
                string.IsNullOrEmpty(vocabFilePath) ? throw new ArgumentNullException(nameof(vocabFilePath)) : File.OpenRead(vocabFilePath),
                options, disposeStream: true);

        /// <summary>
        /// Create a new instance of the <see cref="BertTokenizer"/> class.
        /// </summary>
        /// <param name="vocabStream">The stream containing the vocabulary file.</param>
        /// <param name="options">The options to use for the Bert tokenizer.</param>
        /// <returns>A new instance of the <see cref="BertTokenizer"/> class.</returns>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static BertTokenizer Create(
                    Stream vocabStream,
                    BertOptions? options = null) =>
            Create(vocabStream, options, disposeStream: false);

        /// <summary>
        /// Create a new instance of the <see cref="BertTokenizer"/> class asynchronously.
        /// </summary>
        /// <param name="vocabStream">The stream containing the vocabulary file.</param>
        /// <param name="options">The options to use for the Bert tokenizer.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A task that represents the asynchronous creation of the BertTokenizer.</returns>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static async Task<BertTokenizer> CreateAsync(
                    Stream vocabStream,
                    BertOptions? options = null,
                    CancellationToken cancellationToken = default)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            (Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, string> vocabReverse) = await LoadVocabAsync(vocabStream, useAsync: true, cancellationToken).ConfigureAwait(false);

            return Create(vocab, vocabReverse, options);
        }

        /// <summary>
        /// Create a new instance of the <see cref="BertTokenizer"/> class asynchronously.
        /// </summary>
        /// <param name="vocabFilePath">The path to the vocabulary file.</param>
        /// <param name="options">The options to use for the Bert tokenizer.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A task that represents the asynchronous creation of the BertTokenizer.</returns>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary file is sourced from a trusted provider.
        /// </remarks>
        public static async Task<BertTokenizer> CreateAsync(
                    string vocabFilePath,
                    BertOptions? options = null,
                    CancellationToken cancellationToken = default)
        {
            Stream stream = string.IsNullOrEmpty(vocabFilePath) ? throw new ArgumentNullException(nameof(vocabFilePath)) : File.OpenRead(vocabFilePath);

            try
            {
                return await CreateAsync(stream, options, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                stream.Dispose();
            }
        }

        private static BertTokenizer Create(Stream vocabStream, BertOptions? options, bool disposeStream)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            try
            {
                (Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, string> vocabReverse) = LoadVocabAsync(vocabStream, useAsync: false).GetAwaiter().GetResult();

                return Create(vocab, vocabReverse, options);
            }
            finally
            {
                if (disposeStream)
                {
                    vocabStream.Dispose();
                }
            }
        }

        private static BertTokenizer Create(
                    Dictionary<StringSpanOrdinalKey, int> vocab,
                    Dictionary<int, string> vocabReverse,
                    BertOptions? options)
        {
            options ??= new();

            options.Normalizer ??= options.ApplyBasicTokenization ? new BertNormalizer(options.LowerCaseBeforeTokenization, options.IndividuallyTokenizeCjk, options.RemoveNonSpacingMarks) : null;

            if (options.SplitOnSpecialTokens)
            {
                bool lowerCase = options.ApplyBasicTokenization && options.LowerCaseBeforeTokenization;
                if (options.SpecialTokens is not null)
                {
                    if (lowerCase)
                    {
                        Dictionary<string, int> dic = options.SpecialTokens.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
                        foreach (var kvp in options.SpecialTokens)
                        {
                            if (!vocab.TryGetValue(new StringSpanOrdinalKey(kvp.Key), out int id) || id != kvp.Value)
                            {
                                throw new ArgumentException($"The special token '{kvp.Key}' is not in the vocabulary or assigned id value {id} different than the value {kvp.Value} in the special tokens.");
                            }

                            // Ensure that the special tokens are lowercased.
                            dic[kvp.Key.ToLowerInvariant()] = kvp.Value;
                        }

                        options.SpecialTokens = dic;
                    }
                }
                else
                {
                    // Create a dictionary with the special tokens.
                    Dictionary<string, int> specialTokens = new Dictionary<string, int>();
                    options.SpecialTokens = specialTokens;

                    AddSpecialToken(vocab, specialTokens, options.UnknownToken, lowerCase);
                    AddSpecialToken(vocab, specialTokens, options.SeparatorToken, lowerCase);
                    AddSpecialToken(vocab, specialTokens, options.PaddingToken, lowerCase);
                    AddSpecialToken(vocab, specialTokens, options.ClassificationToken, lowerCase);
                    AddSpecialToken(vocab, specialTokens, options.MaskingToken, lowerCase);
                }
            }

            options.PreTokenizer ??= options.ApplyBasicTokenization ? PreTokenizer.CreateWordOrPunctuation(options.SplitOnSpecialTokens ? options.SpecialTokens : null) : PreTokenizer.CreateWhiteSpace();

            return new BertTokenizer(vocab, vocabReverse, options);
        }

        private static void AddSpecialToken(Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<string, int> specialTokens, string token, bool lowerCase)
        {
            if (token is null || !vocab.TryGetValue(new StringSpanOrdinalKey(token), out int id))
            {
                throw new ArgumentException($"The special token '{token}' is not in the vocabulary.");
            }

            string normalizedToken = token;
            if (lowerCase)
            {
                // Lowercase the special tokens to have the pre-tokenization can find them as we lowercase the input text.
                // we don't even need to do case-insensitive comparisons as we are lowercasing the input text.
                normalizedToken = token.ToLowerInvariant();

                // Add lowercased special tokens to the vocab if they are not already there.
                // This will allow matching during the encoding process.
                vocab[new StringSpanOrdinalKey(normalizedToken)] = id;
            }

            specialTokens[normalizedToken] = id;
        }
    }
}
