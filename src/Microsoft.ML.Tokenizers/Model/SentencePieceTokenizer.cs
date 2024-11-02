// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    // SentencePieceBpe is implementing the BPE algorithm based on the SentencePiece https://github.com/google/sentencepiece.
    // SentencePiece is under the Apache License 2.0 https://github.com/google/sentencepiece/blob/master/LICENSE

    /// <summary>
    /// SentencePieceBpe is a tokenizer that splits the input into tokens using the SentencePiece Bpe model.
    /// </summary>
    public class SentencePieceTokenizer : Tokenizer
    {
        private const int UninitializedId = -2; // indicate if the symbol contains uninitialized id.
        private readonly Dictionary<StringSpanOrdinalKey, (int Id, float Score, byte Type)> _vocab = new();
        private readonly Dictionary<int, string> _vocabReverse = new();
        private IReadOnlyDictionary<string, int>? _publicVocab;
        private readonly int _maxByteId;
        private readonly int _byteCodeToIdOffset; // offset of mapping byte code to the to the Ids.
        private readonly int _oneByteUtf8EncodingMaxId; // the maximum value of the one byte UTF-8 character.
        private readonly Normalizer? _normalizer;
        private readonly Regex? _specialTokensRegex;
        private readonly Dictionary<StringSpanOrdinalKey, int>? _specialTokens;
        private readonly Dictionary<int, string>? _specialTokensReverse;

        internal SentencePieceTokenizer(ModelProto modelProto, bool addBos, bool addEos, IReadOnlyDictionary<string, int>? specialTokens = null) :
            this(modelProto is null ? throw new ArgumentNullException(nameof(modelProto)) : modelProto, specialTokens)
        {
            AddBeginningOfSentence = addBos;
            AddEndOfSentence = addEos;
        }

        private SentencePieceTokenizer(ModelProto modelProto, IReadOnlyDictionary<string, int>? specialTokens)
        {
            for (int i = 0; i < modelProto.Pieces.Count; i++)
            {
                var piece = modelProto.Pieces[i];
                _vocab.Add(new StringSpanOrdinalKey(piece.Piece), (i, piece.Score, (byte)piece.Type));
                _vocabReverse.Add(i, piece.Piece);

                if (piece.Type == ModelProto.Types.SentencePiece.Types.Type.Byte)
                {
                    _maxByteId = i;
                }
            }

            _byteCodeToIdOffset = _vocab.TryGetValue("<0x00>", out (int Id, float Score, byte Type) value) ? value.Id : _maxByteId;
            _oneByteUtf8EncodingMaxId = _byteCodeToIdOffset + 0x7F; // 0x7F is the maximum value of the one byte UTF-8 character.

            BeginningOfSentenceToken = modelProto.TrainerSpec.BosPiece ?? "<s>";
            BeginningOfSentenceId = modelProto.TrainerSpec.BosId <= 0 ? 1 : modelProto.TrainerSpec.BosId;
            EndOfSentenceToken = modelProto.TrainerSpec.EosPiece ?? "</s>";
            EndOfSentenceId = modelProto.TrainerSpec.EosId <= 0 ? 1 : modelProto.TrainerSpec.EosId;
            UnknownToken = modelProto.TrainerSpec.UnkPiece ?? "<unk>";
            UnknownId = modelProto.TrainerSpec.UnkId < 0 ? 0 : modelProto.TrainerSpec.UnkId;

            AddDummyPrefix = modelProto.NormalizerSpec.AddDummyPrefix;
            EscapeWhiteSpaces = modelProto.NormalizerSpec.EscapeWhitespaces;
            TreatWhitespaceAsSuffix = modelProto.TrainerSpec.TreatWhitespaceAsSuffix;
            ByteFallback = modelProto.TrainerSpec.ByteFallback;

            SpecialTokens = specialTokens;
            _normalizer = new SentencePieceNormalizer(modelProto.NormalizerSpec.RemoveExtraWhitespaces, AddDummyPrefix, EscapeWhiteSpaces, modelProto.TrainerSpec.TreatWhitespaceAsSuffix, specialTokens);

            if (specialTokens is not null && specialTokens.Count > 0)
            {
                _specialTokens = new Dictionary<StringSpanOrdinalKey, int>();
                _specialTokensReverse = new Dictionary<int, string>();

                foreach (var item in specialTokens)
                {
                    _specialTokens.Add(new StringSpanOrdinalKey(item.Key), item.Value);
                    _specialTokensReverse.Add(item.Value, item.Key);
                }

                // We create this Regex object without a timeout because we know it will perform the match operation in O(N) time complexity.
                // Therefore, confidently process any text within a reasonable time.
                _specialTokensRegex = new Regex(string.Join("|", specialTokens.Keys.Select(s => Regex.Escape(s))), RegexOptions.Compiled);
            }
        }

        public IReadOnlyDictionary<string, int>? SpecialTokens { get; }

        /// <summary>
        /// Specifies whether the model will do a byte fallback when it encounters unknown tokens during the encoding process.
        /// </summary>
        public bool ByteFallback { get; }

        /// <summary>
        /// Indicate emitting the prefix character U+2581 at the beginning of sentence token during the normalization and encoding.
        /// </summary>
        public bool AddDummyPrefix { get; }

        /// <summary>
        /// Indicate if the spaces should be replaced with character U+2581 during the normalization and encoding.
        /// </summary>
        public bool EscapeWhiteSpaces { get; }

        /// <summary>
        /// Indicate emitting the character U+2581 at the end of the last sentence token instead beginning of sentence token during the normalization and encoding.
        /// </summary>
        public bool TreatWhitespaceAsSuffix { get; private set; }

        /// <summary>
        /// Indicate emitting the beginning of sentence token during the encoding.
        /// </summary>
        public bool AddBeginningOfSentence { get; }

        /// <summary>
        /// Indicate emitting the end of sentence token during the encoding.
        /// </summary>
        public bool AddEndOfSentence { get; }

        /// <summary>
        /// The beginning of sentence token.
        /// </summary>
        public string BeginningOfSentenceToken { get; }

        /// <summary>
        /// The end of sentence token.
        /// </summary>
        public string EndOfSentenceToken { get; }

        /// <summary>
        /// The unknown token.
        /// </summary>
        public string UnknownToken { get; }

        /// <summary>
        /// The id of the beginning of sentence token.
        /// </summary>
        public int BeginningOfSentenceId { get; }

        /// <summary>
        /// The id of the end of sentence token.
        /// </summary>
        public int EndOfSentenceId { get; }

        /// <summary>
        /// The id of the unknown token.
        /// </summary>
        public int UnknownId { get; }

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => null;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _normalizer;

        /// <summary>
        /// The vocabulary of the model.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocabulary
        {
            get
            {
                IReadOnlyDictionary<string, int>? publicVocab = Volatile.Read(ref _publicVocab);
                if (publicVocab is null)
                {
                    var vocab = new Dictionary<string, int>();
                    foreach (var item in _vocab)
                    {
                        vocab.Add(item.Key.ToString(), item.Value.Id);
                    }

                    Interlocked.CompareExchange(ref _publicVocab, new ReadOnlyDictionary<string, int>(vocab), null);
                    publicVocab = _publicVocab;
                }

                return publicVocab;
            }
        }

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            return new EncodeResults<EncodedToken>
            {
                Tokens = EncodeToTokens(text, textSpan, out string? normalizedString, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization, settings.ConsiderNormalization),
                NormalizedText = normalizedString,
                CharsConsumed = normalizedString?.Length ?? text?.Length ?? textSpan.Length
            };
        }

        /// <summary>
        /// Encodes input text a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(string text, out string? normalizedString, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToTokens(text, Span<char>.Empty, out normalizedString, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(ReadOnlySpan<char> text, out string? normalizedString, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToTokens(null, text, out normalizedString, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization);

        private IReadOnlyList<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, out string? normalizedString, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization, bool considerNormalization)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                return [];
            }

            ReadOnlySpan<char> textToEncode = text is null ? textSpan : text.AsSpan();
            if (considerNormalization && _normalizer is not null)
            {
                normalizedString = text is not null ? _normalizer.Normalize(text) : _normalizer.Normalize(textSpan);
                textToEncode = normalizedString.AsSpan();
            }
            else
            {
                normalizedString = null;
            }

            if (textToEncode.Length == 0)
            {
                return [];
            }

            List<EncodedToken>? tokens = new();

            if (_specialTokensRegex is not null)
            {
                EncodeWithSpecialTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, tokens);
            }
            else
            {
                EncodeInternal(textToEncode, addBeginningOfSentence, addEndOfSentence, tokens);
            }

            return tokens;
        }

        private void EncodeWithSpecialTokens(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, List<EncodedToken> tokens)
        {
            Debug.Assert(_specialTokensRegex is not null);

            if (addBeginOfSentence)
            {
                tokens.Add(new EncodedToken(BeginningOfSentenceId, BeginningOfSentenceToken, new Range(0, 0)));
            }

            int currentOffset = 0;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, _specialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    EncodeInternal(text.Slice(currentOffset, Offset - currentOffset), addBeginOfSentence: false, addEndOfSentence: false, tokens);
                }

                if (_specialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
                {
                    tokens.Add(new EncodedToken(id, _specialTokensReverse![id], new Range(Offset, Offset + Length)));
                }

                currentOffset = Offset + Length;
            }

            if (currentOffset < text.Length)
            {
                EncodeInternal(text.Slice(currentOffset), addBeginOfSentence: false, addEndOfSentence: false, tokens);
            }

            if (addEndOfSentence)
            {
                tokens.Add(new EncodedToken(EndOfSentenceId, EndOfSentenceToken, new Range(text.Length, text.Length)));
            }
        }

        /// <summary>
        /// Encode a text to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="tokens">A collection to store the encoded tokens.</param>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        private void EncodeInternal(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, List<EncodedToken> tokens)
        {
            BpeSymbol[] symbols = ArrayPool<BpeSymbol>.Shared.Rent(text.Length);

            Dictionary<(int Index, int Len), (int LeftIndex, int LeftLen, int RightIndex, int RightLen)>? revMerge = Encode(text, symbols);

            if (addBeginOfSentence)
            {
                tokens.Add(new EncodedToken(BeginningOfSentenceId, BeginningOfSentenceToken, new Range(0, 0)));
            }

            for (int index = 0; (uint)index < (uint)symbols.Length; index = symbols[index].next)
            {
                int id = symbols[index].id;
                byte type = symbols[index].type;

                if (id == UninitializedId)
                {
                    if (_vocab.TryGetValue(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), out (int Id, float Score, byte Type) tokenInfo))
                    {
                        id = tokenInfo.Id;
                        type = tokenInfo.Type;
                    }
                    else
                    {
                        id = UnknownId;
                        type = 0;
                    }
                }

                if (type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused)
                {
                    if (id == UnknownId && ByteFallback)
                    {
                        EncodeAsBytes(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), symbols[index].pieceSpan.Index);
                    }
                    else
                    {
                        tokens.Add(new EncodedToken(
                                    id,
                                    GetTokenString(id, symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length, text),
                                    new Range(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Index + symbols[index].pieceSpan.Length)));
                    }
                    continue;
                }

                Segment(symbols[index].pieceSpan, text);
            }

            ArrayPool<BpeSymbol>.Shared.Return(symbols);

            if (addEndOfSentence)
            {
                tokens.Add(new EncodedToken(EndOfSentenceId, EndOfSentenceToken, new Range(text.Length, text.Length)));
            }

            return;

            // Encode the Unknown token to bytes.
            void EncodeAsBytes(ReadOnlySpan<char> text, int index)
            {
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (c <= 0x7F)
                    {
                        int id = (int)c + _byteCodeToIdOffset; // byte code is mapped to the to the Ids starting from 4.

                        if (_vocabReverse.TryGetValue(id, out string? token))
                        {
                            tokens.Add(new EncodedToken(id, token, new Range(index + i, index + i + 1)));
                        }
                    }
                    else
                    {
                        Span<byte> utf8Bytes = stackalloc byte[256];
                        byte[]? arrayPoolArray = null;

                        int len = Encoding.UTF8.GetMaxByteCount(text.Length - i);
                        if (len > utf8Bytes.Length)
                        {
                            arrayPoolArray = ArrayPool<byte>.Shared.Rent(len);
                            utf8Bytes = arrayPoolArray;
                        }

                        // Need to convert the text into UTF-8 bytes and then encode the bytes.
                        int bytesWritten = Helpers.GetUtf8Bytes(text.Slice(i), utf8Bytes);
                        int length = text.Length - i;
                        for (int j = 0; j < bytesWritten; j++)
                        {
                            int id = (int)utf8Bytes[j] + _byteCodeToIdOffset; // byte code is mapped to the to the Ids starting from 4.

                            if (_vocabReverse.TryGetValue(id, out string? token))
                            {
                                tokens.Add(new EncodedToken(id, token, new Range(index + i, index + i + length)));
                            }

                            length = 0;
                        }

                        if (arrayPoolArray is not null)
                        {
                            ArrayPool<byte>.Shared.Return(arrayPoolArray);
                        }

                        break;
                    }
                }
            }

            void Segment((int Index, int Length) pieceSpan, ReadOnlySpan<char> text)
            {
                if (!_vocab.TryGetValue(text.Slice(pieceSpan.Index, pieceSpan.Length), out (int Id, float Score, byte Type) id))
                {
                    EncodeAsBytes(text.Slice(pieceSpan.Index, pieceSpan.Length), pieceSpan.Index);
                    return;
                }

                if (id.Type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused ||
                    revMerge is null ||
                    !revMerge.TryGetValue((pieceSpan.Index, pieceSpan.Length), out (int LeftIndex, int LeftLen, int RightIndex, int RightLen) merge))
                {
                    tokens.Add(new EncodedToken(id.Id, text.Slice(pieceSpan.Index, pieceSpan.Length).ToString(), new Range(pieceSpan.Index, pieceSpan.Index + pieceSpan.Length)));
                    return;
                }

                Segment((merge.LeftIndex, merge.LeftLen), text);
                Segment((merge.RightIndex, merge.RightLen), text);
            }
        }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The encoded results containing the list of encoded Ids.</returns>
        protected override EncodeResults<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            return new EncodeResults<int>
            {
                Tokens = EncodeToIds(text, textSpan, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderNormalization, out string? normalizedString, out int charsConsumed, settings.MaxTokenCount),
                NormalizedText = normalizedString,
                CharsConsumed = charsConsumed
            };
        }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, addBeginningOfSentence, addEndOfSentence, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedString, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedString, out charsConsumed, maxTokenCount);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedString, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedString, out charsConsumed, maxTokenCount);


        private IReadOnlyList<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, bool addBeginningOfSentence, bool addEndOfSentence, bool considerNormalization, out string? normalizedString, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                charsConsumed = 0;
                return [];
            }

            return EncodeToIds(text is null ? textSpan : text.AsSpan(), addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedString, out charsConsumed, maxTokenCount);
        }

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The list of encoded Ids.</returns>
        private IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerNormalization,
                                                out string? normalizedString, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (text.IsEmpty)
            {
                normalizedString = null;
                charsConsumed = 0;
                return [];
            }

            ReadOnlySpan<char> textToEncode;

            if (considerNormalization && _normalizer is not null)
            {
                normalizedString = _normalizer.Normalize(text);
                textToEncode = normalizedString.AsSpan();
            }
            else
            {
                normalizedString = null;
                textToEncode = text;
            }

            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than 0.");
            }

            List<int> ids = new();

            if (_specialTokensRegex is not null)
            {
                EncodeToIdsWithAddedToken(textToEncode, addBeginningOfSentence, addEndOfSentence, ids, out charsConsumed, maxTokenCount);
            }
            else
            {
                EncodeToIds(textToEncode, addBeginningOfSentence, addEndOfSentence, ids, out charsConsumed, maxTokenCount);
            }

            return ids;
        }

        private int EncodeToIdsWithAddedToken(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, IList<int> accumulatedIds, out int charsConsumed, int maxTokens = int.MaxValue)
        {
            Debug.Assert(_specialTokensRegex is not null);
            Debug.Assert(maxTokens > 0);

            charsConsumed = 0;
            int idsCount = 0;

            if (addBeginOfSentence)
            {
                accumulatedIds.Add(BeginningOfSentenceId);
                idsCount++;
            }

            int currentOffset = 0;

            int charsWritten;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, _specialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    idsCount += EncodeToIds(text.Slice(currentOffset, Offset - currentOffset), addBeginOfSentence: false, addEndOfSentence: false, accumulatedIds, out charsWritten, maxTokens - idsCount);
                    charsConsumed += charsWritten;
                }

                if (idsCount < maxTokens && _specialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
                {
                    accumulatedIds.Add(id);
                    idsCount++;
                    charsConsumed += Length;
                }

                currentOffset = Offset + Length;
            }

            if (currentOffset < text.Length && idsCount < maxTokens)
            {
                idsCount += EncodeToIds(text.Slice(currentOffset), addBeginOfSentence: false, addEndOfSentence: false, accumulatedIds, out charsWritten, maxTokens - idsCount);
                charsConsumed += charsWritten;
            }

            if (addEndOfSentence && idsCount < maxTokens)
            {
                accumulatedIds.Add(EndOfSentenceId);
                idsCount++;
            }

            return idsCount;
        }

        /// <summary>
        /// Encode a text to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        private int EncodeToIds(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, IList<int> accumulatedIds, out int charsConsumed, int maxTokens = int.MaxValue)
        {
            charsConsumed = 0;
            if (text.IsEmpty)
            {
                return 0;
            }

            int idsCount = 0;

            if (addBeginOfSentence)
            {
                accumulatedIds.Add(BeginningOfSentenceId);
                idsCount++;
            }

            BpeSymbol[] symbols = ArrayPool<BpeSymbol>.Shared.Rent(text.Length);

            Dictionary<(int Index, int Len), (int LeftIndex, int LeftLen, int RightIndex, int RightLen)>? revMerge = Encode(text, symbols);

            for (int index = 0; index != -1 && index < symbols.Length; index = symbols[index].next)
            {
                int id = symbols[index].id;
                byte type = symbols[index].type;

                if (id == UninitializedId)
                {
                    if (_vocab.TryGetValue(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), out (int Id, float Score, byte Type) tokenInfo))
                    {
                        id = tokenInfo.Id;
                        type = tokenInfo.Type;
                    }
                    else
                    {
                        id = UnknownId;
                        type = 0;
                    }
                }

                if (type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused)
                {
                    if (id == UnknownId && ByteFallback)
                    {
                        if (!EncodeAsBytes(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), symbols[index].pieceSpan.Index, ref charsConsumed))
                        {
                            ArrayPool<BpeSymbol>.Shared.Return(symbols);
                            return idsCount;
                        }
                    }
                    else
                    {
                        if (idsCount < maxTokens)
                        {
                            accumulatedIds.Add(id);
                            charsConsumed += symbols[index].pieceSpan.Length;
                            idsCount++;
                        }
                        else
                        {
                            ArrayPool<BpeSymbol>.Shared.Return(symbols);
                            return idsCount;
                        }
                    }
                    continue;
                }

                if (!Segment(symbols[index].pieceSpan, text, ref charsConsumed))
                {
                    break;
                }
            }

            ArrayPool<BpeSymbol>.Shared.Return(symbols);

            if (addEndOfSentence)
            {
                if (idsCount < maxTokens)
                {
                    accumulatedIds.Add(EndOfSentenceId);
                    idsCount++;
                }
            }

            return idsCount;

            // Encode the Unknown token to bytes.
            bool EncodeAsBytes(ReadOnlySpan<char> text, int index, ref int charsConsumed)
            {
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (c <= 0x7F)
                    {
                        if (idsCount < maxTokens)
                        {
                            charsConsumed++;
                            accumulatedIds.Add((int)c + _byteCodeToIdOffset); // byte code is mapped to the to the Ids starting from 4.
                            idsCount++;
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        Span<byte> utf8Bytes = stackalloc byte[100];
                        byte[]? arrayPoolArray = null;

                        int len = Encoding.UTF8.GetMaxByteCount(text.Length - i);
                        if (len > utf8Bytes.Length)
                        {
                            arrayPoolArray = ArrayPool<byte>.Shared.Rent(len);
                            utf8Bytes = arrayPoolArray;
                        }

                        // Need to convert the text into UTF-8 bytes and then encode the bytes.
                        int bytesWritten = Helpers.GetUtf8Bytes(text.Slice(i), utf8Bytes);

                        bool ret;
                        if (idsCount + bytesWritten <= maxTokens)
                        {
                            for (int j = 0; j < bytesWritten; j++)
                            {
                                accumulatedIds.Add((int)utf8Bytes[j] + _byteCodeToIdOffset); // byte code is mapped to the to the Ids starting from 4.
                            }

                            charsConsumed += text.Length - i;
                            ret = true;
                        }
                        else
                        {
                            ret = false;
                        }

                        if (arrayPoolArray is not null)
                        {
                            ArrayPool<byte>.Shared.Return(arrayPoolArray);
                        }

                        return ret;
                    }
                }

                return true;
            }

            bool Segment((int Index, int Length) pieceSpan, ReadOnlySpan<char> text, ref int charsConsumed)
            {
                if (!_vocab.TryGetValue(text.Slice(pieceSpan.Index, pieceSpan.Length), out (int Id, float Score, byte Type) id))
                {
                    return EncodeAsBytes(text.Slice(pieceSpan.Index, pieceSpan.Length), pieceSpan.Index, ref charsConsumed);
                }

                if (id.Type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused ||
                    revMerge is null ||
                    !revMerge.TryGetValue((pieceSpan.Index, pieceSpan.Length), out (int LeftIndex, int LeftLen, int RightIndex, int RightLen) merge))
                {
                    if (idsCount < maxTokens)
                    {
                        accumulatedIds.Add(id.Id);
                        charsConsumed += pieceSpan.Length;
                        idsCount++;
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                return Segment((merge.LeftIndex, merge.LeftLen), text, ref charsConsumed) && Segment((merge.RightIndex, merge.RightLen), text, ref charsConsumed);
            }
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        protected override int CountTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            return CountTokens(text, textSpan, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out _, out _, settings.MaxTokenCount);
        }

        private int CountTokens(string? text, ReadOnlySpan<char> textSpan, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int charsConsumed, int maxTokenCount = int.MaxValue)
            => CountTokens(text is null ? textSpan : text.AsSpan(), addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out charsConsumed, maxTokenCount);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public int CountTokens(string text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text, Span<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public int CountTokens(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(null, text, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public int CountTokens(string text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int charsConsumed, int maxTokenCount = int.MaxValue)
            => CountTokens(text, Span<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out charsConsumed, maxTokenCount);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public int CountTokens(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (text.IsEmpty)
            {
                normalizedString = null;
                charsConsumed = 0;
                return 0;
            }

            ReadOnlySpan<char> textToEncode;
            if (considerNormalization && _normalizer is not null)
            {
                normalizedString = _normalizer.Normalize(text);
                textToEncode = normalizedString.AsSpan();
            }
            else
            {
                normalizedString = null;
                textToEncode = text;
            }

            return _specialTokensRegex is not null ?
                CountTokensWithSpecialTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, out charsConsumed, maxTokenCount) :
                CountTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, out charsConsumed, maxTokenCount);
        }

        private int CountTokensWithSpecialTokens(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, out int charsConsumed, int maxTokens = int.MaxValue)
        {
            Debug.Assert(_specialTokensRegex is not null);
            Debug.Assert(maxTokens > 0);

            charsConsumed = 0;
            int idsCount = 0;

            if (addBeginOfSentence)
            {
                idsCount++;
            }

            int currentOffset = 0;

            int charsWritten;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, _specialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    idsCount += CountTokens(text.Slice(currentOffset, Offset - currentOffset), addBeginOfSentence: false, addEndOfSentence: false, out charsWritten, maxTokens - idsCount);
                    charsConsumed += charsWritten;
                }

                if (idsCount < maxTokens && _specialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
                {
                    idsCount++;
                    charsConsumed += Length;
                }

                currentOffset = Offset + Length;
            }

            if (currentOffset < text.Length && idsCount < maxTokens)
            {
                idsCount += CountTokens(text.Slice(currentOffset), addBeginOfSentence: false, addEndOfSentence: false, out charsWritten, maxTokens - idsCount);
                charsConsumed += charsWritten;
            }

            if (addEndOfSentence && idsCount < maxTokens)
            {
                idsCount++;
            }

            return idsCount;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        private int CountTokens(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, out int charsConsumed, int maxTokens = int.MaxValue)
        {
            charsConsumed = 0;
            if (text.IsEmpty)
            {
                return 0;
            }

            int tokenCount = addBeginOfSentence ? 1 : 0;

            BpeSymbol[] symbols = ArrayPool<BpeSymbol>.Shared.Rent(text.Length);

            Dictionary<(int Index, int Len), (int LeftIndex, int LeftLen, int RightIndex, int RightLen)>? revMerge = Encode(text, symbols);

            for (int index = 0; index != -1 && index < symbols.Length; index = symbols[index].next)
            {
                int id = symbols[index].id;
                byte type = symbols[index].type;

                if (id == UninitializedId)
                {
                    if (_vocab.TryGetValue(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), out (int Id, float Score, byte Type) tokenInfo))
                    {
                        id = tokenInfo.Id;
                        type = tokenInfo.Type;
                    }
                    else
                    {
                        id = UnknownId;
                        type = 0;
                    }
                }

                if (type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused)
                {
                    if (id == UnknownId && ByteFallback)
                    {
                        if (!EncodeAsBytes(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), symbols[index].pieceSpan.Index, ref charsConsumed))
                        {
                            break;
                        }
                    }
                    else
                    {
                        if (tokenCount < maxTokens)
                        {
                            tokenCount++;
                            charsConsumed += symbols[index].pieceSpan.Length;
                        }
                        else
                        {
                            break;
                        }
                    }
                    continue;
                }

                if (!Segment(symbols[index].pieceSpan, text, ref charsConsumed))
                {
                    break;
                }
            }

            ArrayPool<BpeSymbol>.Shared.Return(symbols);

            if (addEndOfSentence)
            {
                if (tokenCount < maxTokens)
                {
                    tokenCount++;
                }
            }

            return tokenCount;

            // Encode the Unknown token to bytes.
            bool EncodeAsBytes(ReadOnlySpan<char> text, int index, ref int charsConsumed)
            {
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (c <= 0x7F)
                    {
                        if (tokenCount < maxTokens)
                        {
                            tokenCount++;
                            charsConsumed++;
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        Span<byte> utf8Bytes = stackalloc byte[100];
                        byte[]? arrayPoolArray = null;

                        int len = Encoding.UTF8.GetMaxByteCount(text.Length - i);
                        if (len > utf8Bytes.Length)
                        {
                            arrayPoolArray = ArrayPool<byte>.Shared.Rent(len);
                            utf8Bytes = arrayPoolArray;
                        }

                        // Need to convert the text into UTF-8 bytes and then encode the bytes.
                        int encodedCount = Helpers.GetUtf8Bytes(text.Slice(i), utf8Bytes);
                        bool ret;

                        if (tokenCount + encodedCount <= maxTokens)
                        {
                            tokenCount += encodedCount;
                            charsConsumed += text.Length - i;
                            ret = true;
                        }
                        else
                        {
                            ret = false;
                        }

                        if (arrayPoolArray is not null)
                        {
                            ArrayPool<byte>.Shared.Return(arrayPoolArray);
                        }

                        return ret;
                    }
                }

                return true;
            }

            bool Segment((int Index, int Length) pieceSpan, ReadOnlySpan<char> text, ref int charsConsumed)
            {
                if (!_vocab.TryGetValue(text.Slice(pieceSpan.Index, pieceSpan.Length), out (int Id, float Score, byte Type) id))
                {
                    return EncodeAsBytes(text.Slice(pieceSpan.Index, pieceSpan.Length), pieceSpan.Index, ref charsConsumed);
                }

                if (id.Type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused ||
                    revMerge is null ||
                    !revMerge.TryGetValue((pieceSpan.Index, pieceSpan.Length), out (int LeftIndex, int LeftLen, int RightIndex, int RightLen) merge))
                {
                    if (tokenCount < maxTokens)
                    {
                        tokenCount++;
                        charsConsumed += pieceSpan.Length;
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                return Segment((merge.LeftIndex, merge.LeftLen), text, ref charsConsumed) && Segment((merge.RightIndex, merge.RightLen), text, ref charsConsumed);
            }
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <param name="fromEnd">Indicate whether to find the index from the end of the text.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="settings" /> has <see cref="EncodeSettings.ConsiderNormalization"/> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// If <paramRef name="fromEnd" /> is <see langword="false"/>, it represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the input text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// If <paramRef name="fromEnd" /> is <see langword="true"/>, it represents the index of the first character to be included. In cases where no tokens fit, the result will be the text length; conversely,
        /// if all tokens fit, the result will be zero.
        /// </returns>
        protected override int GetIndexByTokenCount(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings, bool fromEnd, out string? normalizedString, out int tokenCount)
        {
            if (fromEnd)
            {
                return GetIndexByTokenCountFromEnd(text, textSpan, settings.MaxTokenCount, settings.ConsiderNormalization, out normalizedString, out tokenCount);
            }

            tokenCount = CountTokens(text, textSpan, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out normalizedString, out int charsConsumed, settings.MaxTokenCount);
            return charsConsumed;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(text, Span<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out int charsConsumed, maxTokenCount);
            return charsConsumed;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(null, text, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out int charsConsumed, maxTokenCount);
            return charsConsumed;
        }

        private int GetIndexByTokenCountFromEnd(string? text, ReadOnlySpan<char> textSpan, int maxTokenCount, bool considerNormalization, out string? normalizedString, out int tokenCount)
            => GetIndexByTokenCountFromEnd(text is null ? textSpan : text.AsSpan(), AddBeginningOfSentence, AddEndOfSentence, maxTokenCount, considerNormalization, out normalizedString, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedString"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public int GetIndexByTokenCountFromEnd(string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, bool considerNormalization, out string? normalizedString, out int tokenCount)
            => GetIndexByTokenCountFromEnd(text is null ? ReadOnlySpan<char>.Empty : text.AsSpan(), addBeginningOfSentence, addEndOfSentence, maxTokenCount, considerNormalization, out normalizedString, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedString"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public int GetIndexByTokenCountFromEnd(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, bool considerNormalization, out string? normalizedString, out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            if (text.IsEmpty)
            {
                normalizedString = null;
                tokenCount = 0;
                return 0;
            }

            ReadOnlySpan<char> textToEncode;
            if (considerNormalization && _normalizer is not null)
            {
                normalizedString = _normalizer.Normalize(text);
                textToEncode = normalizedString.AsSpan();
            }
            else
            {
                normalizedString = null;
                textToEncode = text;
            }

            int textIndex;
            if (_specialTokensRegex is not null)
            {
                tokenCount = CountTokensFromEndWithSpecialTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, out textIndex, maxTokenCount);
            }
            else
            {
                tokenCount = CountTokensFromEnd(textToEncode, addBeginningOfSentence, addEndOfSentence, out textIndex, maxTokenCount);
            }

            return textIndex;
        }

        private int CountTokensFromEndWithSpecialTokens(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, out int textIndex, int maxTokens)
        {
            Debug.Assert(_specialTokensRegex is not null);
            Debug.Assert(maxTokens > 0);
            Debug.Assert(text.Length > 0);

            textIndex = text.Length;
            int idsCount = 0;

            if (addEndOfSentence)
            {
                idsCount++;
            }

            (int Offset, int Length)[] splits = PreTokenizer.SplitText(text, _specialTokensRegex!).ToArray();

            if (splits.Length == 0)
            {
                return CountTokensFromEnd(text, addBeginOfSentence, addEndOfSentence, out textIndex, maxTokens);
            }

            (int Offset, int Length) current = splits[splits.Length - 1];

            int splitTextIndex;
            ReadOnlySpan<char> splitText;

            if (current.Offset + current.Length < text.Length)
            {
                splitText = text.Slice(current.Offset + current.Length);
                idsCount += CountTokensFromEnd(splitText, addBeginOfSentence: false, addEndOfSentence: false, out splitTextIndex, maxTokens - idsCount);
                textIndex -= splitText.Length - splitTextIndex;
            }

            for (int i = splits.Length - 1; i >= 0 && idsCount < maxTokens; i--)
            {
                current = splits[i];

                if (_specialTokens!.TryGetValue(text.Slice(current.Offset, current.Length), out int id))
                {
                    idsCount++;
                }
                textIndex -= current.Length;

                if (current.Offset > 0 && idsCount < maxTokens)
                {
                    int start = i > 0 ? splits[i - 1].Offset + splits[i - 1].Length : 0;
                    splitText = text.Slice(start, current.Offset - start);
                    idsCount += CountTokensFromEnd(splitText, addBeginOfSentence: false, addEndOfSentence: false, out splitTextIndex, maxTokens - idsCount);
                    textIndex -= splitText.Length - splitTextIndex;
                }
            }

            if (addBeginOfSentence && idsCount < maxTokens)
            {
                idsCount++;
            }

            return idsCount;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="textIndex">Starting from this index to the end of the text will encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        private int CountTokensFromEnd(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, out int textIndex, int maxTokens = int.MaxValue)
        {
            textIndex = text.Length;
            if (text.IsEmpty)
            {
                return 0;
            }

            int tokenCount = addEndOfSentence ? 1 : 0;

            BpeSymbol[] symbols = ArrayPool<BpeSymbol>.Shared.Rent(text.Length);

            Dictionary<(int Index, int Len), (int LeftIndex, int LeftLen, int RightIndex, int RightLen)>? revMerge = Encode(text, symbols);

            // Move to the last symbol.
            int lastSymbolIndex = 0;
            while (symbols[lastSymbolIndex].next != -1 && lastSymbolIndex < symbols.Length)
            {
                lastSymbolIndex = symbols[lastSymbolIndex].next;
            }

            for (int index = lastSymbolIndex; index >= 0; index = symbols[index].prev)
            {
                int id = symbols[index].id;
                byte type = symbols[index].type;

                if (id == UninitializedId)
                {
                    if (_vocab.TryGetValue(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), out (int Id, float Score, byte Type) tokenInfo))
                    {
                        id = tokenInfo.Id;
                        type = tokenInfo.Type;
                    }
                    else
                    {
                        id = UnknownId;
                        type = 0;
                    }
                }

                if (type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused)
                {
                    if (id == UnknownId && ByteFallback)
                    {
                        if (!EncodeAsBytesFromEnd(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), symbols[index].pieceSpan.Index, ref textIndex))
                        {
                            break;
                        }
                    }
                    else
                    {
                        if (tokenCount < maxTokens)
                        {
                            tokenCount++;
                            textIndex -= symbols[index].pieceSpan.Length;
                        }
                        else
                        {
                            break;
                        }
                    }
                    continue;
                }

                if (!SegmentFromEnd(symbols[index].pieceSpan, text, ref textIndex))
                {
                    break;
                }
            }

            ArrayPool<BpeSymbol>.Shared.Return(symbols);

            if (addBeginOfSentence)
            {
                if (tokenCount < maxTokens)
                {
                    tokenCount++;
                }
            }

            return tokenCount;

            // Encode the Unknown token to bytes.
            bool EncodeAsBytesFromEnd(ReadOnlySpan<char> text, int index, ref int textIndex)
            {
                for (int i = text.Length - 1; i >= 0; i--)
                {
                    char c = text[i];
                    if (c <= 0x7F)
                    {
                        if (tokenCount < maxTokens)
                        {
                            tokenCount++;
                            textIndex--;
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        Span<byte> utf8Bytes = stackalloc byte[100];
                        byte[]? arrayPoolArray = null;

                        int len = Encoding.UTF8.GetMaxByteCount(text.Length - i);
                        if (len > utf8Bytes.Length)
                        {
                            arrayPoolArray = ArrayPool<byte>.Shared.Rent(len);
                            utf8Bytes = arrayPoolArray;
                        }

                        // Need to convert the text into UTF-8 bytes and then encode the bytes.
                        int encodedCount = Helpers.GetUtf8Bytes(text.Slice(0, i + 1), utf8Bytes);
                        bool ret;

                        if (tokenCount + encodedCount <= maxTokens)
                        {
                            tokenCount += encodedCount;
                            textIndex -= i + 1;
                            ret = true;
                        }
                        else
                        {
                            ret = false;
                        }

                        if (arrayPoolArray is not null)
                        {
                            ArrayPool<byte>.Shared.Return(arrayPoolArray);
                        }

                        return ret;
                    }
                }

                return true;
            }

            bool SegmentFromEnd((int Index, int Length) pieceSpan, ReadOnlySpan<char> text, ref int textIndex)
            {
                if (!_vocab.TryGetValue(text.Slice(pieceSpan.Index, pieceSpan.Length), out (int Id, float Score, byte Type) id))
                {
                    return EncodeAsBytesFromEnd(text.Slice(pieceSpan.Index, pieceSpan.Length), pieceSpan.Index, ref textIndex);
                }

                if (id.Type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused ||
                    revMerge is null ||
                    !revMerge.TryGetValue((pieceSpan.Index, pieceSpan.Length), out (int LeftIndex, int LeftLen, int RightIndex, int RightLen) merge))
                {
                    if (tokenCount < maxTokens)
                    {
                        tokenCount++;
                        textIndex -= pieceSpan.Length;
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                // Segment the right part first.
                return SegmentFromEnd((merge.RightIndex, merge.RightLen), text, ref textIndex) && SegmentFromEnd((merge.LeftIndex, merge.LeftLen), text, ref textIndex);
            }
        }

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string Decode(IEnumerable<int> ids)
            => Decode(ids, considerSpecialTokens: false);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens during decoding.</param>
        /// <returns>The decoded string.</returns>
        public string Decode(IEnumerable<int> ids, bool considerSpecialTokens)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            using IEnumerator<int> enumerator = ids.GetEnumerator();
            if (!enumerator.MoveNext())
            {
                return string.Empty;
            }

            ValueStringBuilder sb = new(stackalloc char[256]);

            int bytesCount = -1;
            byte[]? bytesPoolArray = null;
            bool prefixRemoved = false;
            int suffixIndex = -1;
            char prefixSuffixChar = EscapeWhiteSpaces ? SentencePieceNormalizer.DummyPrefix : ' ';

            if (enumerator.Current <= _maxByteId)
            {
                // First token is a byte token.

                while (enumerator.Current < _byteCodeToIdOffset)
                {
                    // It is possible listing some special tokens before the byte tokens in the tokenizer's data.
                    TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, ref sb);

                    // Skip control tokens.
                    if (!enumerator.MoveNext())
                    {
                        return sb.ToString();
                    }
                }

                if (enumerator.Current <= _maxByteId)
                {
                    EncodeByte(enumerator.Current, _oneByteUtf8EncodingMaxId, _byteCodeToIdOffset, ref bytesCount, ref bytesPoolArray, ref sb);
                }
                else if (_vocabReverse.TryGetValue(enumerator.Current, out string? token))
                {
                    AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token, prefixSuffixChar, ref sb, ref prefixRemoved, ref suffixIndex);
                }
                else
                {
                    TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, ref sb);
                }
            }
            else if (_vocabReverse.TryGetValue(enumerator.Current, out string? token))
            {
                AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token, prefixSuffixChar, ref sb, ref prefixRemoved, ref suffixIndex);
            }
            else
            {
                TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, ref sb);
            }

            char[]? charPoolArray = null;

            while (enumerator.MoveNext())
            {
                if (enumerator.Current < _byteCodeToIdOffset)
                {
                    if (bytesCount >= 1)
                    {
                        FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, ref sb);
                    }

                    // It is possible listing some special tokens before the byte tokens in the tokenizer's data.
                    TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, ref sb);

                    continue;
                }

                if (enumerator.Current <= _maxByteId)
                {
                    if (bytesCount >= 1)
                    {
                        Debug.Assert(bytesPoolArray is not null);

                        if (bytesCount >= bytesPoolArray!.Length)
                        {
                            Helpers.ArrayPoolGrow(ref bytesPoolArray, bytesCount * 2);
                        }

                        bytesPoolArray![bytesCount++] = (byte)(enumerator.Current - _byteCodeToIdOffset);
                    }
                    else
                    {
                        EncodeByte(enumerator.Current, _oneByteUtf8EncodingMaxId, _byteCodeToIdOffset, ref bytesCount, ref bytesPoolArray, ref sb);
                    }
                }
                else
                {
                    if (bytesCount >= 1)
                    {
                        FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, ref sb);
                    }

                    if (_vocabReverse.TryGetValue(enumerator.Current, out string? token))
                    {
                        AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token, prefixSuffixChar, ref sb, ref prefixRemoved, ref suffixIndex);
                    }
                    else
                    {
                        TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, ref sb);
                    }
                }
            }

            if (bytesCount >= 1)
            {
                FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, ref sb);
            }

            if (AddDummyPrefix && TreatWhitespaceAsSuffix && suffixIndex >= 0 && sb.Length > 0)
            {
                Debug.Assert(sb[suffixIndex] == SentencePieceNormalizer.DummyPrefix);
                Debug.Assert(sb.Length > suffixIndex);

                sb.Remove(suffixIndex, 1);
            }

            if (bytesPoolArray is not null)
            {
                ArrayPool<byte>.Shared.Return(bytesPoolArray);
            }

            if (charPoolArray is not null)
            {
                ArrayPool<char>.Shared.Return(charPoolArray);
            }

            return EscapeWhiteSpaces ? sb.ToString(SentencePieceNormalizer.DummyPrefix, ' ') : sb.ToString();

            static void FlushBytes(ref int bytesCount, ref byte[]? bytesPoolArray, ref char[]? charPoolArray, ref ValueStringBuilder sb)
            {
                Debug.Assert(bytesCount >= 1);
                Debug.Assert(bytesPoolArray is not null);

                int len = Encoding.UTF8.GetMaxCharCount(bytesCount);

                charPoolArray ??= ArrayPool<char>.Shared.Rent(Math.Max(len, 50));

                if (len > charPoolArray.Length)
                {
                    Helpers.ArrayPoolGrow(ref charPoolArray, len);
                }

                int charCount = Helpers.GetChars(bytesPoolArray.AsSpan(0, bytesCount), charPoolArray);

                sb.Append(charPoolArray.AsSpan(0, charCount));
                bytesCount = -1;
            }

            static void EncodeByte(int id, int oneByteUtf8EncodingMaxId, int byteCodeToIdOffset, ref int bytesCount, ref byte[]? bytesPoolArray, ref ValueStringBuilder sb)
            {
                if (id <= oneByteUtf8EncodingMaxId)
                {
                    sb.Append((char)(id - byteCodeToIdOffset));
                }
                else
                {
                    bytesCount = 1;
                    bytesPoolArray ??= ArrayPool<byte>.Shared.Rent(50);
                    bytesPoolArray[0] = (byte)(id - byteCodeToIdOffset);
                }
            }

            static void AppendTokenWithCheckingPrefix(bool addDummyPrefix, bool treatWhitespaceAsSuffix, string token, char prefixSuffixChar, ref ValueStringBuilder sb, ref bool prefixRemoved, ref int suffixIndex)
            {
                if (token.Length == 0)
                {
                    return;
                }

                if (!addDummyPrefix)
                {
                    sb.Append(token);
                    return;
                }

                if (treatWhitespaceAsSuffix)
                {
                    sb.Append(token);
                    if (token[token.Length - 1] == prefixSuffixChar)
                    {
                        suffixIndex = sb.Length - 1;
                    }
                }
                else
                {
                    sb.Append(!prefixRemoved && token[0] == prefixSuffixChar ? token.AsSpan(1) : token.AsSpan());
                }

                prefixRemoved = true;
            }

            static void TryDecodeAsSpecialToken(SentencePieceTokenizer tokenizer, int id, bool considerSpecialTokens, ref ValueStringBuilder sb)
            {
                if (!considerSpecialTokens)
                {
                    return;
                }

                if (id == tokenizer.BeginningOfSentenceId)
                {
                    sb.Append(tokenizer.BeginningOfSentenceToken);
                }
                else if (id == tokenizer.EndOfSentenceId)
                {
                    sb.Append(tokenizer.EndOfSentenceToken);
                }
                else if (id == tokenizer.UnknownId)
                {
                    sb.Append(tokenizer.UnknownToken);
                }
                else if (tokenizer._specialTokensReverse?.TryGetValue(id, out string? specialToken) is true)
                {
                    sb.Append(specialToken);
                }
            }
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
            => Decode(ids, destination, considerSpecialTokens: false, out idsConsumed, out charsWritten);

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// /// <param name="considerSpecialTokens">Indicate whether to consider special tokens during decoding.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, bool considerSpecialTokens, out int idsConsumed, out int charsWritten)
        {
            idsConsumed = 0;
            charsWritten = 0;

            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            using IEnumerator<int> enumerator = ids.GetEnumerator();
            if (!enumerator.MoveNext())
            {
                return OperationStatus.Done;
            }

            Span<char> buffer = destination;

            int bytesCount = -1;
            byte[]? bytesPoolArray = null;
            bool prefixRemoved = false;
            int suffixIndex = -1;
            char prefixSuffixChar = EscapeWhiteSpaces ? SentencePieceNormalizer.DummyPrefix : ' ';

            if (enumerator.Current <= _maxByteId)
            {
                // First token is a byte token.
                while (enumerator.Current < _byteCodeToIdOffset)
                {
                    OperationStatus status = TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, buffer, ref charsWritten);
                    if (status != OperationStatus.Done)
                    {
                        return status;
                    }
                    buffer = destination.Slice(charsWritten);

                    // Skip control tokens.
                    idsConsumed++;
                    if (!enumerator.MoveNext())
                    {
                        return OperationStatus.Done;
                    }
                }

                if (enumerator.Current <= _maxByteId)
                {
                    if (!EncodeByte(enumerator.Current, _oneByteUtf8EncodingMaxId, _byteCodeToIdOffset, ref bytesCount, buffer, ref charsWritten, ref idsConsumed, ref bytesPoolArray))
                    {
                        return OperationStatus.DestinationTooSmall;
                    }
                }
                else if (_vocabReverse.TryGetValue(enumerator.Current, out string? token))
                {
                    if (!AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token, prefixSuffixChar, destination, ref prefixRemoved, ref suffixIndex, ref idsConsumed, ref charsWritten))
                    {
                        return OperationStatus.DestinationTooSmall;
                    }
                }
                else
                {
                    OperationStatus status = TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, buffer, ref charsWritten);
                    if (status != OperationStatus.Done)
                    {
                        return status;
                    }

                    idsConsumed++;
                }
            }
            else if (_vocabReverse.TryGetValue(enumerator.Current, out string? token))
            {
                if (!AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token, prefixSuffixChar, destination, ref prefixRemoved, ref suffixIndex, ref idsConsumed, ref charsWritten))
                {
                    return OperationStatus.DestinationTooSmall;
                }
            }
            else
            {
                OperationStatus status = TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, buffer, ref charsWritten);
                if (status != OperationStatus.Done)
                {
                    return status;
                }

                idsConsumed++;
            }

            char[]? charPoolArray = null;

            while (enumerator.MoveNext())
            {
                buffer = destination.Slice(charsWritten);

                if (enumerator.Current < _byteCodeToIdOffset)
                {
                    if (bytesCount >= 1)
                    {
                        if (!FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, buffer, ref charsWritten, ref idsConsumed))
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                    }

                    OperationStatus status = TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, buffer, ref charsWritten);
                    if (status != OperationStatus.Done)
                    {
                        return status;
                    }

                    idsConsumed++;
                    continue;
                }

                if (enumerator.Current <= _maxByteId)
                {
                    if (bytesCount >= 1)
                    {
                        Debug.Assert(bytesPoolArray is not null);

                        if (bytesCount >= bytesPoolArray!.Length)
                        {
                            Helpers.ArrayPoolGrow(ref bytesPoolArray, bytesCount * 2);
                        }

                        bytesPoolArray![bytesCount++] = (byte)(enumerator.Current - _byteCodeToIdOffset);
                    }
                    else
                    {
                        if (!EncodeByte(enumerator.Current, _oneByteUtf8EncodingMaxId, _byteCodeToIdOffset, ref bytesCount, buffer, ref charsWritten, ref idsConsumed, ref bytesPoolArray))
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                    }
                }
                else
                {
                    if (bytesCount >= 1)
                    {
                        if (!FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, buffer, ref charsWritten, ref idsConsumed))
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                    }

                    if (_vocabReverse.TryGetValue(enumerator.Current, out string? token))
                    {
                        if (!AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token, prefixSuffixChar, destination, ref prefixRemoved, ref suffixIndex, ref idsConsumed, ref charsWritten))
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                    }
                    else
                    {
                        OperationStatus status = TryDecodeAsSpecialToken(this, enumerator.Current, considerSpecialTokens, buffer, ref charsWritten);
                        if (status != OperationStatus.Done)
                        {
                            return status;
                        }

                        idsConsumed++;
                    }
                }
            }

            buffer = destination.Slice(charsWritten);

            if (bytesCount >= 1)
            {
                if (!FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, buffer, ref charsWritten, ref idsConsumed))
                {
                    return OperationStatus.DestinationTooSmall;
                }
            }

            if (suffixIndex >= 0)
            {
                Debug.Assert(destination[suffixIndex] == ' ');

                if (suffixIndex < charsWritten - 1)
                {
                    destination.Slice(suffixIndex + 1, charsWritten - suffixIndex - 1).CopyTo(destination.Slice(suffixIndex));
                }

                charsWritten--;
            }

            if (bytesPoolArray is not null)
            {
                ArrayPool<byte>.Shared.Return(bytesPoolArray);
            }

            if (charPoolArray is not null)
            {
                ArrayPool<char>.Shared.Return(charPoolArray);
            }

            return OperationStatus.Done;

            static OperationStatus TryDecodeAsSpecialToken(SentencePieceTokenizer tokenizer, int id, bool considerSpecialTokens, Span<char> buffer, ref int charsWritten)
            {
                string? specialToken = null;

                if (id == tokenizer.BeginningOfSentenceId)
                {
                    specialToken = tokenizer.BeginningOfSentenceToken;
                }
                else if (id == tokenizer.EndOfSentenceId)
                {
                    specialToken = tokenizer.EndOfSentenceToken;
                }
                else if (id == tokenizer.UnknownId)
                {
                    specialToken = tokenizer.UnknownToken;
                }
                else if (!tokenizer._specialTokensReverse?.TryGetValue(id, out specialToken) is true)
                {
                    return OperationStatus.InvalidData;
                }

                if (considerSpecialTokens && specialToken is not null)
                {
                    if (buffer.Length < specialToken!.Length)
                    {
                        return OperationStatus.DestinationTooSmall;
                    }

                    specialToken.AsSpan().CopyTo(buffer);
                    charsWritten += specialToken.Length;
                }

                return OperationStatus.Done;
            }

            static bool FlushBytes(ref int bytesCount, ref byte[]? bytesPoolArray, ref char[]? charPoolArray, Span<char> buffer, ref int charsWritten, ref int idsConsumed)
            {
                Debug.Assert(bytesCount >= 1);
                Debug.Assert(bytesPoolArray is not null);

                int len = Encoding.UTF8.GetMaxCharCount(bytesCount);

                charPoolArray ??= ArrayPool<char>.Shared.Rent(Math.Max(len, 50));

                if (len > charPoolArray.Length)
                {
                    Helpers.ArrayPoolGrow(ref charPoolArray, len);
                }

                int charCount = Helpers.GetChars(bytesPoolArray.AsSpan(0, bytesCount), charPoolArray);

                if (charCount > buffer.Length)
                {
                    return false;
                }

                charPoolArray.AsSpan(0, charCount).CopyTo(buffer);
                charsWritten += charCount;
                idsConsumed += bytesCount;
                bytesCount = -1;

                return true;
            }

            static bool EncodeByte(int id, int oneByteUtf8EncodingMaxId, int byteCodeToIdOffset, ref int bytesCount, Span<char> buffer, ref int charsWritten, ref int idsConsumed, ref byte[]? bytesPoolArray)
            {
                if (id <= oneByteUtf8EncodingMaxId)
                {
                    if (buffer.Length < 1)
                    {
                        return false;
                    }

                    buffer[0] = (char)(id - byteCodeToIdOffset);
                    charsWritten++;
                    idsConsumed++;
                }
                else
                {
                    bytesCount = 1;
                    bytesPoolArray ??= ArrayPool<byte>.Shared.Rent(50);
                    bytesPoolArray[0] = (byte)(id - byteCodeToIdOffset);
                }

                return true;
            }

            static bool AppendTokenWithCheckingPrefix(bool addDummyPrefix, bool treatWhitespaceAsSuffix, string token, char prefixSuffixChar, Span<char> destination, ref bool prefixRemoved, ref int suffixIndex, ref int idsConsumed, ref int charsConsumed)
            {
                if (token.Length == 0)
                {
                    return true;
                }

                Span<char> buffer = destination.Slice(charsConsumed);

                ReadOnlySpan<char> tokenSpan = token.AsSpan();

                if (!addDummyPrefix)
                {
                    if (tokenSpan.Length > buffer.Length)
                    {
                        return false;
                    }

                    if (prefixSuffixChar != ' ')
                    {
                        for (int i = 0; i < tokenSpan.Length; i++)
                        {
                            buffer[i] = tokenSpan[i] == prefixSuffixChar ? ' ' : tokenSpan[i];
                        }
                    }
                    else
                    {
                        tokenSpan.CopyTo(buffer);
                    }

                    buffer = buffer.Slice(tokenSpan.Length);
                    charsConsumed += tokenSpan.Length;
                    idsConsumed++;
                    return true;
                }

                if (treatWhitespaceAsSuffix)
                {
                    if (tokenSpan[tokenSpan.Length - 1] == prefixSuffixChar)
                    {
                        suffixIndex = charsConsumed + tokenSpan.Length - 1;
                    }

                    if (tokenSpan.Length > buffer.Length)
                    {
                        return false;
                    }

                    if (prefixSuffixChar != ' ')
                    {
                        for (int i = 0; i < tokenSpan.Length; i++)
                        {
                            buffer[i] = tokenSpan[i] == prefixSuffixChar ? ' ' : tokenSpan[i];
                        }
                    }
                    else
                    {
                        tokenSpan.CopyTo(buffer);
                    }

                    charsConsumed += tokenSpan.Length;

                    idsConsumed++;
                }
                else
                {
                    int delta = !prefixRemoved && token[0] == prefixSuffixChar ? 1 : 0;
                    if (buffer.Length < token.Length - delta)
                    {
                        return false;
                    }

                    tokenSpan = tokenSpan.Slice(delta);
                    if (prefixSuffixChar != ' ')
                    {
                        for (int i = 0; i < tokenSpan.Length; i++)
                        {
                            buffer[i] = tokenSpan[i] == prefixSuffixChar ? ' ' : tokenSpan[i];
                        }
                    }
                    else
                    {
                        tokenSpan.CopyTo(buffer);
                    }

                    charsConsumed += tokenSpan.Length;
                    idsConsumed++;

                    if (!prefixRemoved && delta == 1)
                    {
                        prefixRemoved = true;
                    }
                }

                return true;
            }
        }

        // Tries to avoid string allocations if possible.
        private string GetTokenString(int id, int index, int length, ReadOnlySpan<char> text)
            => _vocabReverse.TryGetValue(id, out string? token) ? token : text.Slice(index, length).ToString();

        private Dictionary<(int Index, int Len), (int LeftIndex, int LeftLen, int RightIndex, int RightLen)>? Encode(ReadOnlySpan<char> text, BpeSymbol[] symbols)
        {
            Debug.Assert(text.Length > 0);
            Debug.Assert(symbols.Length >= text.Length);

            int symbolIndex = 0;
            int spanIndex = 0;

            while (spanIndex < text.Length)
            {
                int len = (Char.IsHighSurrogate(text[spanIndex]) && spanIndex < text.Length - 1 && Char.IsLowSurrogate(text[spanIndex + 1])) ? 2 : 1;

                BpeSymbol s = new(
                            prev: symbolIndex == 0 ? -1 : symbolIndex - 1,
                            next: spanIndex + len >= text.Length ? -1 : symbolIndex + 1,
                            pieceSpan: (spanIndex, len),
                            id: UninitializedId,
                            type: 0);

                symbols[symbolIndex++] = s;
                spanIndex += len;
            }

            PriorityQueue<SymbolPair> agenda = new(symbolIndex);
            Dictionary<(int Index, int Len), (int LeftIndex, int LeftLen, int RightIndex, int RightLen)>? revMerge = null;

            for (int i = 1; i < symbolIndex; i++)
            {
                TryMerge(i - 1, i, text);
            }

            while (agenda.Count > 0)
            {
                SymbolPair top = agenda.Dequeue();

                if (symbols[top.Left].pieceSpan.Length == 0 || symbols[top.Right].pieceSpan.Length == 0 ||
                    symbols[top.Left].pieceSpan.Length + symbols[top.Right].pieceSpan.Length != top.Length)
                {
                    continue;
                }

                // Replaces symbols with `top` rule.
                symbols[top.Left].pieceSpan = (symbols[top.Left].pieceSpan.Index, symbols[top.Left].pieceSpan.Length + symbols[top.Right].pieceSpan.Length);
                symbols[top.Left].id = top.Id;

                // Updates prev/next pointers.
                symbols[top.Left].next = symbols[top.Right].next;

                if (symbols[top.Right].next >= 0)
                {
                    symbols[symbols[top.Right].next].prev = top.Left;
                }
                symbols[top.Right].pieceSpan = (0, 0);

                // Adds new symbol pairs which are newly added after symbol replacement.
                TryMerge(symbols[top.Left].prev, top.Left, text);
                TryMerge(top.Left, symbols[top.Left].next, text);
            }

            return revMerge;

            void TryMerge(int left, int right, ReadOnlySpan<char> textSpan)
            {
                if (left == -1 || right == -1)
                {
                    return;
                }

                int pieceLength = symbols[left].pieceSpan.Length + symbols[right].pieceSpan.Length;
                if (!_vocab.TryGetValue(textSpan.Slice(symbols[left].pieceSpan.Index, pieceLength), out (int Id, float Score, byte Type) leftId))
                {
                    return;
                }

                symbols[left].type = leftId.Type;

                SymbolPair pair = new(left, right, leftId.Score, pieceLength, leftId.Id);
                agenda.Enqueue(pair);

                if (leftId.Type == (byte)ModelProto.Types.SentencePiece.Types.Type.Unused)
                {
                    revMerge ??= new();
                    revMerge.Add((symbols[left].pieceSpan.Index, pieceLength), (symbols[left].pieceSpan.Index, symbols[left].pieceSpan.Length, symbols[right].pieceSpan.Index, symbols[right].pieceSpan.Length));
                }
            }
        }

        private struct SymbolPair : IEquatable<SymbolPair>, IComparable<SymbolPair>
        {
            public int Left { get; set; }
            public int Right { get; set; }
            public int Length { get; set; }
            public float Score { get; set; }
            public int Id { get; set; }

            public SymbolPair(int left, int right, float score, int length, int id)
            {
                Left = left;
                Right = right;
                Score = score;
                Length = length;
                Id = id;
            }

            public int CompareTo(SymbolPair other)
            {
                if (Score != other.Score)
                {
                    return other.Score.CompareTo(Score);
                }

                return other.Left.CompareTo(Left);
            }

            public override int GetHashCode()
            {
                int hashCode = 23;
                hashCode = (hashCode * 37) + Score.GetHashCode();
                hashCode = (hashCode * 37) + Left.GetHashCode();
                return hashCode;
            }

            public bool Equals(SymbolPair other) => Left == other.Left && Score == other.Score;
        }

        private record struct BpeSymbol(int prev, int next, (int Index, int Length) pieceSpan, int id, byte type);
    }
}
