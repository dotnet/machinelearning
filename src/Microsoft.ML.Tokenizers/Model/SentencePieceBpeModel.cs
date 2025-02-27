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
    internal sealed class SentencePieceBpeModel : SentencePieceBaseModel
    {
        private const int UninitializedId = -2; // indicate if the symbol contains uninitialized id.
        private readonly Dictionary<StringSpanOrdinalKey, (int Id, float Score, byte Type)> _vocab = new();
        private readonly Dictionary<int, string> _vocabReverse = new();
        private IReadOnlyDictionary<string, int>? _publicVocab;

        internal SentencePieceBpeModel(ModelProto modelProto, bool addBos, bool addEos, IReadOnlyDictionary<string, int>? specialTokens = null) : base(modelProto, addBos, addEos, specialTokens)
        {
            for (int i = 0; i < modelProto.Pieces.Count; i++)
            {
                var piece = modelProto.Pieces[i];
                _vocab.Add(new StringSpanOrdinalKey(piece.Piece), (i, piece.Score, (byte)piece.Type));
                _vocabReverse.Add(i, piece.Piece);

                if (piece.Type == ModelProto.Types.SentencePiece.Types.Type.Byte)
                {
                    MaxByteId = i;
                }
            }

            ByteCodeToIdOffset = _vocab.TryGetValue("<0x00>", out (int Id, float Score, byte Type) value) ? value.Id : MaxByteId;
            OneByteUtf8EncodingMaxId = ByteCodeToIdOffset + 0x7F; // 0x7F is the maximum value of the one byte UTF-8 character.
        }

        internal SentencePieceBpeModel(SentencePieceOptions options) : base(options)
        {
            if (options.PrecompiledNormalizationData is not null)
            {
                throw new NotSupportedException("Normalization data is not supported for SentencePieceBpeModel.");
            }

            Debug.Assert(options.Vocabulary is not null);

            int id = 0;
            foreach (var item in options.Vocabulary!)
            {
                _vocab.Add(new StringSpanOrdinalKey(item.Token), (id, item.Score, (byte)ModelProto.Types.SentencePiece.Types.Type.Normal));
                _vocabReverse.Add(id++, item.Token);
            }

            if (options.ByteFallback)
            {
                if (!_vocab.TryGetValue("<0x00>", out (int Id, float Score, byte Type) value))
                {
                    throw new ArgumentException("'ByteFallback' is enabled but the vocabulary must include a special token for each byte value (0-255) in the format <0xNN>, where NN represents the byte's hexadecimal value.");
                }

                ByteCodeToIdOffset = value.Id;
                OneByteUtf8EncodingMaxId = ByteCodeToIdOffset + 0x7F; // 0x7F is the maximum value of the one byte UTF-8 character.
            }

            if (!_vocab.TryGetValue(options.UnknownToken, out (int Id, float Score, byte Type) unknownToken))
            {
                throw new ArgumentException($"The vocabulary must include the unknown token '{options.UnknownToken}'.");
            }
            UnknownId = unknownToken.Id;

            if (!_vocab.TryGetValue(options.BeginningOfSentenceToken, out (int Id, float Score, byte Type) beginOfSentenceToken))
            {
                throw new ArgumentException($"The vocabulary must include the beginning of sentence token '{options.BeginningOfSentenceToken}'.");
            }
            BeginningOfSentenceId = beginOfSentenceToken.Id;

            if (!_vocab.TryGetValue(options.EndOfSentenceToken, out (int Id, float Score, byte Type) endOfSentenceToken))
            {
                throw new ArgumentException($"The vocabulary must include the end of sentence token '{options.EndOfSentenceToken}'.");
            }
            EndOfSentenceId = endOfSentenceToken.Id;
        }

        public override IReadOnlyDictionary<string, int> Vocabulary
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

        public override bool TryMapIdToToken(int id, out string? token) => _vocabReverse.TryGetValue(id, out token);

        public override IReadOnlyList<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, out string? normalizedText, bool addBeginningOfSentence, bool addEndOfSentence, bool considerNormalization)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedText = null;
                return [];
            }

            ReadOnlySpan<char> textToEncode = text is null ? textSpan : text.AsSpan();
            if (considerNormalization && Normalizer is not null)
            {
                normalizedText = text is not null ? Normalizer.Normalize(text) : Normalizer.Normalize(textSpan);
                textToEncode = normalizedText.AsSpan();
            }
            else
            {
                normalizedText = null;
            }

            if (textToEncode.Length == 0)
            {
                return [];
            }

            List<EncodedToken> tokens = new();

            if (SpecialTokensRegex is not null)
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
            Debug.Assert(SpecialTokensRegex is not null);

            if (addBeginOfSentence)
            {
                tokens.Add(new EncodedToken(BeginningOfSentenceId, BeginningOfSentenceToken, new Range(0, 0)));
            }

            int currentOffset = 0;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, SpecialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    EncodeInternal(text.Slice(currentOffset, Offset - currentOffset), addBeginOfSentence: false, addEndOfSentence: false, tokens);
                }

                if (InternalSpecialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
                {
                    tokens.Add(new EncodedToken(id, SpecialTokensReverse![id], new Range(Offset, Offset + Length)));
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
                        int id = (int)c + ByteCodeToIdOffset; // byte code is mapped to the to the Ids starting from 4.

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
                            int id = (int)utf8Bytes[j] + ByteCodeToIdOffset; // byte code is mapped to the to the Ids starting from 4.

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

        public override IReadOnlyList<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, bool addBeginningOfSentence, bool addEndOfSentence, bool considerNormalization,
                                        out string? normalizedText, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedText = null;
                charsConsumed = 0;
                return [];
            }

            return EncodeToIds(text is null ? textSpan : text.AsSpan(), addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedText, out charsConsumed, maxTokenCount);
        }

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The list of encoded Ids.</returns>
        private IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerNormalization,
                                                out string? normalizedText, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (text.IsEmpty)
            {
                normalizedText = null;
                charsConsumed = 0;
                return [];
            }

            ReadOnlySpan<char> textToEncode;

            if (considerNormalization && Normalizer is not null)
            {
                normalizedText = Normalizer.Normalize(text);
                textToEncode = normalizedText.AsSpan();
            }
            else
            {
                normalizedText = null;
                textToEncode = text;
            }

            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than 0.");
            }

            List<int> ids = new();

            if (SpecialTokensRegex is not null)
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
            Debug.Assert(SpecialTokensRegex is not null);
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

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, SpecialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    idsCount += EncodeToIds(text.Slice(currentOffset, Offset - currentOffset), addBeginOfSentence: false, addEndOfSentence: false, accumulatedIds, out charsWritten, maxTokens - idsCount);
                    charsConsumed += charsWritten;
                }

                if (idsCount < maxTokens && InternalSpecialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
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
                            accumulatedIds.Add((int)c + ByteCodeToIdOffset); // byte code is mapped to the to the Ids starting from 4.
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
                                accumulatedIds.Add((int)utf8Bytes[j] + ByteCodeToIdOffset); // byte code is mapped to the to the Ids starting from 4.
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

        public override int CountTokens(
                        string? text,
                        ReadOnlySpan<char> textSpan,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerNormalization,
                        out string? normalizedText,
                        out int charsConsumed,
                        int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            textSpan = text is null ? textSpan : text.AsSpan();

            if (textSpan.IsEmpty)
            {
                normalizedText = null;
                charsConsumed = 0;
                return 0;
            }

            ReadOnlySpan<char> textToEncode;
            if (considerNormalization && Normalizer is not null)
            {
                normalizedText = Normalizer.Normalize(textSpan);
                textToEncode = normalizedText.AsSpan();
            }
            else
            {
                normalizedText = null;
                textToEncode = textSpan;
            }

            return SpecialTokensRegex is not null ?
                CountTokensWithSpecialTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, out charsConsumed, maxTokenCount) :
                CountTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, out charsConsumed, maxTokenCount);
        }

        private int CountTokensWithSpecialTokens(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, out int charsConsumed, int maxTokens = int.MaxValue)
        {
            Debug.Assert(SpecialTokensRegex is not null);
            Debug.Assert(maxTokens > 0);

            charsConsumed = 0;
            int idsCount = 0;

            if (addBeginOfSentence)
            {
                idsCount++;
            }

            int currentOffset = 0;

            int charsWritten;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, SpecialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    idsCount += CountTokens(text.Slice(currentOffset, Offset - currentOffset), addBeginOfSentence: false, addEndOfSentence: false, out charsWritten, maxTokens - idsCount);
                    charsConsumed += charsWritten;
                }

                if (idsCount < maxTokens && InternalSpecialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
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

        public override int GetIndexByTokenCountFromEnd(string? text, ReadOnlySpan<char> textSpan, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, bool considerNormalization, out string? normalizedText, out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            textSpan = text is null ? textSpan : text.AsSpan();

            if (textSpan.IsEmpty)
            {
                normalizedText = null;
                tokenCount = 0;
                return 0;
            }

            ReadOnlySpan<char> textToEncode;
            if (considerNormalization && Normalizer is not null)
            {
                normalizedText = Normalizer.Normalize(textSpan);
                textToEncode = normalizedText.AsSpan();
            }
            else
            {
                normalizedText = null;
                textToEncode = textSpan;
            }

            int textIndex;
            if (SpecialTokensRegex is not null)
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
            Debug.Assert(SpecialTokensRegex is not null);
            Debug.Assert(maxTokens > 0);
            Debug.Assert(text.Length > 0);

            textIndex = text.Length;
            int idsCount = 0;

            if (addEndOfSentence)
            {
                idsCount++;
            }

            (int Offset, int Length)[] splits = PreTokenizer.SplitText(text, SpecialTokensRegex!).ToArray();

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

                if (InternalSpecialTokens!.TryGetValue(text.Slice(current.Offset, current.Length), out int id))
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

        // Tries to avoid string allocations if possible.
        private string GetTokenString(int id, int index, int length, ReadOnlySpan<char> text)
            => _vocabReverse.TryGetValue(id, out string? token) ? token : text.Slice(index, length).ToString();

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
