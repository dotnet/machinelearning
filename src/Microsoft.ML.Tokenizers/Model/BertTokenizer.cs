// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
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
                    PreTokenizer? preTokenizer,
                    Normalizer? normalizer,
                    IReadOnlyDictionary<string, int>? specialTokens,
                    bool doLowerCase,
                    bool doBasicTokenization,
                    bool splitOnSpecialTokens,
                    string unknownToken,
                    string sepToken,
                    string padToken,
                    string clsToken,
                    string maskToken,
                    bool tokenizeChineseChars,
                    bool stripAccents) : base(vocab, vocabReverse, preTokenizer, normalizer, specialTokens, unknownToken)
        {
            DoLowerCase = doLowerCase;
            DoBasicTokenization = doBasicTokenization;
            SplitOnSpecialTokens = splitOnSpecialTokens;

            SepToken = sepToken;
            SepTokenId = vocab[new StringSpanOrdinalKey(sepToken)];

            PadToken = padToken;
            PadTokenId = vocab[new StringSpanOrdinalKey(padToken)];

            ClsToken = clsToken;
            ClsTokenId = vocab[new StringSpanOrdinalKey(clsToken)];

            MaskToken = maskToken;
            MaskTokenId = vocab[new StringSpanOrdinalKey(maskToken)];

            TokenizeChineseChars = tokenizeChineseChars;
            StripAccents = stripAccents;
        }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should lowercase the input text.
        /// </summary>
        public bool DoLowerCase { get; }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should do basic tokenization. Like clean text, normalize it, lowercasing, etc.
        /// </summary>
        public bool DoBasicTokenization { get; }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should split on the special tokens or treat special tokens as normal text.
        /// </summary>
        public bool SplitOnSpecialTokens { get; }

        /// <summary>
        /// Gets the separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering.
        /// It is also used as the last token of a sequence built with special tokens.
        /// </summary>
        public string SepToken { get; }

        /// <summary>
        /// Gets the separator token Id
        /// </summary>
        public int SepTokenId { get; }

        /// <summary>
        /// Gets the token used for padding, for example when batching sequences of different lengths
        /// </summary>
        public string PadToken { get; }

        /// <summary>
        /// Gets padding token Id
        /// </summary>
        public int PadTokenId { get; }

        /// <summary>
        /// Gets the classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification).
        /// It is the first token of the sequence when built with special tokens.
        /// </summary>
        public string ClsToken { get; }

        /// <summary>
        /// Gets the classifier token Id
        /// </summary>
        public int ClsTokenId { get; }

        /// <summary>
        /// Gets the mask token used for masking values. This is the token used when training this model with masked language modeling.
        /// This is the token which the model will try to predict.
        /// </summary>
        public string MaskToken { get; }

        /// <summary>
        /// Gets the mask token Id
        /// </summary>
        public int MaskTokenId { get; }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should split the Chinese characters into tokens.
        /// </summary>
        public bool TokenizeChineseChars { get; }

        /// <summary>
        /// Gets a value indicating whether the tokenizer should strip accents characters.
        /// </summary>
        public bool StripAccents { get; }

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

                list.Insert(0, ClsTokenId);
                list.Add(SepTokenId);

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

                list.Insert(0, ClsTokenId);
                list.Add(SepTokenId);

                return list;
            }

            return ids;
        }

        /// <summary>
        /// Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:
        ///     - single sequence: `[CLS] tokenIds0 [SEP]`
        ///     - pair of sequences: `[CLS] tokenIds0 [SEP] tokenIds1 [SEP]`
        /// </summary>
        /// <param name="tokenIds0">List of IDs to which the special tokens will be added.</param>
        /// <param name="tokenIds1">Optional second list of IDs for sequence pairs.</param>
        /// <returns>The list of IDs with special tokens added.</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="tokenIds0"/> is null.</exception>
        public IReadOnlyList<int> BuildInputsWithSpecialTokens(IEnumerable<int> tokenIds0, IEnumerable<int>? tokenIds1 = null)
        {
            if (tokenIds0 is null)
            {
                throw new ArgumentNullException(nameof(tokenIds0));
            }

            List<int> ids;

            if (tokenIds0 is ICollection<int> c1)
            {
                int capacity = c1.Count + 2;    // Add 2 for [CLS] and two [SEP] tokens.

                if (tokenIds1 is not null)
                {
                    capacity += tokenIds1 is ICollection<int> c2 ? c2.Count + 1 : c1.Count + 1;
                }

                ids = new(capacity) { ClsTokenId };
            }
            else
            {
                // slow path
                ids = new List<int>(10) { ClsTokenId };
            }

            ids.AddRange(tokenIds0);
            ids.Add(SepTokenId);

            if (tokenIds1 is not null)
            {
                ids.AddRange(tokenIds1);
                ids.Add(SepTokenId);
            }

            return ids;
        }

        /// <summary>
        /// Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:
        ///     - single sequence: `[CLS] tokenIds0 [SEP]`
        ///     - pair of sequences: `[CLS] tokenIds0 [SEP] tokenIds1 [SEP]`
        /// </summary>
        /// <param name="tokenIds0">List of IDs to which the special tokens will be added.</param>
        /// <param name="buffer">The buffer to write the token IDs with special tokens added.</param>
        /// <param name="written">The number of elements written to the buffer.</param>
        /// <param name="tokenIds1">Optional second list of IDs for sequence pairs.</param>
        /// <returns>The status of the operation.</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="tokenIds0"/> is null.</exception>
        public OperationStatus BuildInputsWithSpecialTokens(IEnumerable<int> tokenIds0, Span<int> buffer, out int written, IEnumerable<int>? tokenIds1 = null)
        {
            if (tokenIds0 is null)
            {
                throw new ArgumentNullException(nameof(tokenIds0));
            }

            written = 0;
            if (buffer.Length < 1)
            {
                return OperationStatus.DestinationTooSmall;
            }

            buffer[written++] = ClsTokenId;
            foreach (int id in tokenIds0)
            {
                if (buffer.Length <= written)
                {
                    written = 0;
                    return OperationStatus.DestinationTooSmall;
                }

                buffer[written++] = id;
            }

            if (buffer.Length <= written)
            {
                written = 0;
                return OperationStatus.DestinationTooSmall;
            }
            buffer[written++] = SepTokenId;

            if (tokenIds1 is not null)
            {
                foreach (int id in tokenIds1)
                {
                    if (buffer.Length <= written)
                    {
                        written = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    buffer[written++] = id;
                }

                if (buffer.Length <= written)
                {
                    written = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                buffer[written++] = SepTokenId;
            }

            return OperationStatus.Done;
        }

        /// <summary>
        /// Retrieve sequence tokens mask from a IDs list.
        /// </summary>
        /// <param name="tokenIds0">List of IDs.</param>
        /// <param name="tokenIds1">Optional second list of IDs for sequence pairs.</param>
        /// <param name="alreadyHasSpecialTokens">Indicate whether or not the token list is already formatted with special tokens for the model.</param>
        /// <returns>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</returns>
        /// <exception cref="ArgumentNullException"></exception>
        public IReadOnlyList<int> GetSpecialTokensMask(IEnumerable<int> tokenIds0, IEnumerable<int>? tokenIds1 = null, bool alreadyHasSpecialTokens = false)
        {
            if (tokenIds0 is null)
            {
                throw new ArgumentNullException(nameof(tokenIds0));
            }

            List<int> mask;
            if (tokenIds0 is ICollection<int> c1)
            {
                int capcity = c1.Count + 2;

                if (tokenIds1 is not null)
                {
                    capcity += tokenIds1 is ICollection<int> c2 ? c2.Count + 1 : c1.Count + 1;
                }

                mask = new List<int>(capcity);
            }
            else
            {
                mask = new List<int>(10);
            }

            if (!alreadyHasSpecialTokens)
            {
                mask.Add(1); // CLS
                mask.AddRange(Enumerable.Repeat(0, tokenIds0.Count()));
                mask.Add(1); // SEP
                if (tokenIds1 is not null)
                {
                    mask.AddRange(Enumerable.Repeat(0, tokenIds1.Count()));
                    mask.Add(1); // SEP
                }

                return mask;
            }

            foreach (int id in tokenIds0)
            {
                mask.Add(id == ClsTokenId || id == SepTokenId || id == PadTokenId || id == MaskTokenId || id == UnknownTokenId ? 1 : 0);
            }

            if (tokenIds1 is not null)
            {
                foreach (int id in tokenIds1)
                {
                    mask.Add(id == ClsTokenId || id == SepTokenId || id == PadTokenId || id == MaskTokenId || id == UnknownTokenId ? 1 : 0);
                }
            }

            return mask;
        }

        /// <summary>
        /// Retrieve sequence tokens mask from a IDs list.
        /// </summary>
        /// <param name="tokenIds0">List of IDs.</param>
        /// <param name="buffer">The buffer to write the mask. The integers written values are in the range [0, 1]: 1 for a special token, 0 for a sequence token.</param>
        /// <param name="written">The number of elements written to the buffer.</param>
        /// <param name="tokenIds1">Optional second list of IDs for sequence pairs.</param>
        /// <param name="alreadyHasSpecialTokens">Indicate whether or not the token list is already formatted with special tokens for the model.</param>
        /// <returns>The status of the operation.</returns>
        /// <exception cref="ArgumentNullException"></exception>
        public OperationStatus GetSpecialTokensMask(IEnumerable<int> tokenIds0, Span<int> buffer, out int written, IEnumerable<int>? tokenIds1 = null, bool alreadyHasSpecialTokens = false)
        {
            if (tokenIds0 is null)
            {
                throw new ArgumentNullException(nameof(tokenIds0));
            }

            written = 0;
            if (!alreadyHasSpecialTokens)
            {
                if (buffer.Length < 1)
                {
                    return OperationStatus.DestinationTooSmall;
                }
                buffer[written++] = 1; // CLS

                foreach (int id in tokenIds0)
                {
                    if (buffer.Length <= written)
                    {
                        written = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    buffer[written++] = 0;
                }

                if (buffer.Length <= written)
                {
                    written = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                buffer[written++] = 1; // SEP

                if (tokenIds1 is not null)
                {
                    foreach (int id in tokenIds1)
                    {
                        if (buffer.Length <= written)
                        {
                            written = 0;
                            return OperationStatus.DestinationTooSmall;
                        }
                        buffer[written++] = 0;
                    }

                    if (buffer.Length <= written)
                    {
                        written = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    buffer[written++] = 1; // SEP
                }

                return OperationStatus.Done;
            }

            foreach (int id in tokenIds0)
            {
                if (buffer.Length <= written)
                {
                    written = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                buffer[written++] = id == ClsTokenId || id == SepTokenId || id == PadTokenId || id == MaskTokenId || id == UnknownTokenId ? 1 : 0;
            }

            if (tokenIds1 is not null)
            {
                foreach (int id in tokenIds1)
                {
                    if (buffer.Length <= written)
                    {
                        written = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    buffer[written++] = id == ClsTokenId || id == SepTokenId || id == PadTokenId || id == MaskTokenId || id == UnknownTokenId ? 1 : 0;
                }
            }

            return OperationStatus.Done;
        }

        /// <summary>
        /// Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence pair mask has the following format:
        ///         0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        ///         | first sequence    | second sequence |
        /// If <paramref name="tokenIds1"/> is null, this method only returns the first portion of the type ids (0s).
        /// </summary>
        /// <param name="tokenIds0">List of token IDs for the first sequence.</param>
        /// <param name="tokenIds1">Optional list of token IDs for the second sequence.</param>
        /// <returns>List of token type IDs according to the given sequence(s).</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="tokenIds0"/> is null.</exception>
        public IReadOnlyList<int> CreateTokenTypeIdsFromSequences(IEnumerable<int> tokenIds0, IEnumerable<int>? tokenIds1 = null)
        {
            if (tokenIds0 is null)
            {
                throw new ArgumentNullException(nameof(tokenIds0));
            }

            List<int> typeIds;
            if (tokenIds0 is ICollection<int> c1)
            {
                int capacity = c1.Count + 2;    // Add 2 for [CLS] and [SEP] tokens.

                if (tokenIds1 is not null)
                {
                    capacity += tokenIds1 is ICollection<int> c2 ? c2.Count + 1 : c1.Count + 1;
                }

                typeIds = new List<int>(capacity);
            }
            else
            {
                typeIds = new List<int>(10);
            }

            foreach (var id in tokenIds0)
            {
                typeIds.Add(0);
            }
            typeIds.Add(0); // [CLS]
            typeIds.Add(0); // [SEP]

            if (tokenIds1 is not null)
            {
                foreach (int id in tokenIds1)
                {
                    typeIds.Add(1);
                }

                typeIds.Add(1); // [SEP]
            }

            return typeIds;
        }

        public OperationStatus CreateTokenTypeIdsFromSequences(IEnumerable<int> tokenIds0, Span<int> buffer, out int written, IEnumerable<int>? tokenIds1 = null)
        {
            if (tokenIds0 is null)
            {
                throw new ArgumentNullException(nameof(tokenIds0));
            }

            written = 0;

            // Add 2 for [CLS] and [SEP] tokens. Add 1 for [SEP] token if tokenIds1 is not null.
            int capacity = tokenIds0.Count() + 2 + (tokenIds1 is null ? 0 : tokenIds1.Count() + 1);
            if (buffer.Length < 2)
            {
                return OperationStatus.DestinationTooSmall;
            }
            buffer[written++] = 0; // [CLS]
            buffer[written++] = 0; // [SEP]

            foreach (int id in tokenIds0)
            {
                if (buffer.Length <= written)
                {
                    written = 0;
                    return OperationStatus.DestinationTooSmall;
                }
                buffer[written++] = 0;
            }

            if (tokenIds1 is not null)
            {
                foreach (int id in tokenIds1)
                {
                    if (buffer.Length <= written)
                    {
                        written = 0;
                        return OperationStatus.DestinationTooSmall;
                    }
                    buffer[written++] = 1;
                }

                if (buffer.Length < written)
                {
                    return OperationStatus.DestinationTooSmall;
                }
                buffer[written++] = 1; // [SEP]
            }

            return OperationStatus.Done;
        }

        /// <summary>
        /// Create a new instance of the <see cref="BertTokenizer"/> class.
        /// </summary>
        /// <param name="vocabFilePath">The path to the vocabulary file.</param>
        /// <param name="doLowerCase">A value indicating whether the tokenizer should lowercase the input text.</param>
        /// <param name="doBasicTokenization">A value indicating whether the tokenizer should do basic tokenization. Like clean text, normalize it, lowercasing, etc.</param>
        /// <param name="splitOnSpecialTokens">A value indicating whether the tokenizer should split on special tokens.</param>
        /// <param name="unknownToken">The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.</param>
        /// <param name="sepToken">The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.</param>
        /// <param name="padToken">The token used for padding, for example when batching sequences of different lengths.</param>
        /// <param name="clsToken">The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.</param>
        /// <param name="maskToken">The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.</param>
        /// <param name="tokenizeChineseChars">A value indicating whether the tokenizer should split the Chinese characters into tokens.</param>
        /// <param name="stripAccents">A value indicating whether the tokenizer should strip accents characters.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        public static BertTokenizer Create(
                    string vocabFilePath,
                    bool doLowerCase = true,
                    bool doBasicTokenization = true,
                    bool splitOnSpecialTokens = true,
                    string unknownToken = "[UNK]",
                    string sepToken = "[SEP]",
                    string padToken = "[PAD]",
                    string clsToken = "[CLS]",
                    string maskToken = "[MASK]",
                    bool tokenizeChineseChars = true,
                    bool stripAccents = false) =>
            Create(
                string.IsNullOrEmpty(vocabFilePath) ? throw new ArgumentNullException(nameof(vocabFilePath)) : File.OpenRead(vocabFilePath),
                doLowerCase, doBasicTokenization, splitOnSpecialTokens, unknownToken, sepToken, padToken, clsToken, maskToken, tokenizeChineseChars, stripAccents, disposeStream: true);

        /// <summary>
        /// Create a new instance of the <see cref="BertTokenizer"/> class.
        /// </summary>
        /// <param name="vocabStream">The stream containing the vocabulary file.</param>
        /// <param name="doLowerCase">A value indicating whether the tokenizer should lowercase the input text.</param>
        /// <param name="doBasicTokenization">A value indicating whether the tokenizer should do basic tokenization. Like clean text, normalize it, lowercasing, etc.</param>
        /// <param name="splitOnSpecialTokens">A value indicating whether the tokenizer should split on special tokens.</param>
        /// <param name="unknownToken">The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.</param>
        /// <param name="sepToken">The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.</param>
        /// <param name="padToken">The token used for padding, for example when batching sequences of different lengths.</param>
        /// <param name="clsToken">The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.</param>
        /// <param name="maskToken">The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.</param>
        /// <param name="tokenizeChineseChars">A value indicating whether the tokenizer should split the Chinese characters into tokens.</param>
        /// <param name="stripAccents">A value indicating whether the tokenizer should strip accents characters.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        public static BertTokenizer Create(
                    Stream vocabStream,
                    bool doLowerCase = true,
                    bool doBasicTokenization = true,
                    bool splitOnSpecialTokens = true,
                    string unknownToken = "[UNK]",
                    string sepToken = "[SEP]",
                    string padToken = "[PAD]",
                    string clsToken = "[CLS]",
                    string maskToken = "[MASK]",
                    bool tokenizeChineseChars = true,
                    bool stripAccents = false) =>
            Create(vocabStream, doLowerCase, doBasicTokenization, splitOnSpecialTokens, unknownToken, sepToken, padToken, clsToken, maskToken, tokenizeChineseChars, stripAccents, disposeStream: false);

        /// <summary>
        /// Create a new instance of the <see cref="BertTokenizer"/> class asynchronously.
        /// </summary>
        /// <param name="vocabStream">The stream containing the vocabulary file.</param>
        /// <param name="doLowerCase">A value indicating whether the tokenizer should lowercase the input text.</param>
        /// <param name="doBasicTokenization">A value indicating whether the tokenizer should do basic tokenization. Like clean text, normalize it, lowercasing, etc.</param>
        /// <param name="splitOnSpecialTokens">A value indicating whether the tokenizer should split on special tokens.</param>
        /// <param name="unknownToken">The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.</param>
        /// <param name="sepToken">The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.</param>
        /// <param name="padToken">The token used for padding, for example when batching sequences of different lengths.</param>
        /// <param name="clsToken">The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.</param>
        /// <param name="maskToken">The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.</param>
        /// <param name="tokenizeChineseChars">A value indicating whether the tokenizer should split the Chinese characters into tokens.</param>
        /// <param name="stripAccents">A value indicating whether the tokenizer should strip accents characters.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        public static async Task<BertTokenizer> CreateAsync(
                    Stream vocabStream,
                    bool doLowerCase = true,
                    bool doBasicTokenization = true,
                    bool splitOnSpecialTokens = true,
                    string unknownToken = "[UNK]",
                    string sepToken = "[SEP]",
                    string padToken = "[PAD]",
                    string clsToken = "[CLS]",
                    string maskToken = "[MASK]",
                    bool tokenizeChineseChars = true,
                    bool stripAccents = false)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            (Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, string> vocabReverse) = await LoadVocabAsync(vocabStream, useAsync: true).ConfigureAwait(false);

            return Create(vocab, vocabReverse, doLowerCase, doBasicTokenization, splitOnSpecialTokens, unknownToken, sepToken, padToken, clsToken, maskToken, tokenizeChineseChars, stripAccents);
        }

        private static BertTokenizer Create(
                            Stream vocabStream,
                            bool doLowerCase,
                            bool doBasicTokenization,
                            bool splitOnSpecialTokens,
                            string unknownToken,
                            string sepToken,
                            string padToken,
                            string clsToken,
                            string maskToken,
                            bool tokenizeChineseChars,
                            bool stripAccents,
                            bool disposeStream)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            try
            {
                (Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, string> vocabReverse) = LoadVocabAsync(vocabStream, useAsync: false).GetAwaiter().GetResult();

                return Create(vocab, vocabReverse, doLowerCase, doBasicTokenization, splitOnSpecialTokens, unknownToken, sepToken, padToken, clsToken, maskToken, tokenizeChineseChars, stripAccents);
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
                    bool doLowerCase,
                    bool doBasicTokenization,
                    bool splitOnSpecialTokens,
                    string unknownToken,
                    string sepToken,
                    string padToken,
                    string clsToken,
                    string maskToken,
                    bool tokenizeChineseChars,
                    bool stripAccents)
        {
            Normalizer? normalizer = doBasicTokenization ? new BertNormalizer(doLowerCase, tokenizeChineseChars, stripAccents) : null;

            Dictionary<string, int>? specialTokens = new();
            bool lowerCase = doBasicTokenization && doLowerCase && splitOnSpecialTokens;

            AddSpecialToken(vocab, specialTokens, unknownToken, lowerCase);
            AddSpecialToken(vocab, specialTokens, sepToken, lowerCase);
            AddSpecialToken(vocab, specialTokens, padToken, lowerCase);
            AddSpecialToken(vocab, specialTokens, clsToken, lowerCase);
            AddSpecialToken(vocab, specialTokens, maskToken, lowerCase);

            PreTokenizer? preTokenizer = doBasicTokenization ?
                                            PreTokenizer.CreateWhiteSpaceOrPunctuationPreTokenizer(splitOnSpecialTokens ? specialTokens : null) :
                                            PreTokenizer.CreateWhiteSpacePreTokenizer();

            return new BertTokenizer(vocab, vocabReverse, preTokenizer, normalizer, specialTokens, doLowerCase, doBasicTokenization,
                                    splitOnSpecialTokens, unknownToken, sepToken, padToken, clsToken, maskToken, tokenizeChineseChars, stripAccents);
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
