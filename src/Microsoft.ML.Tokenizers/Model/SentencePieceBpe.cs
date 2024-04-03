﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    // SentencePieceBpe is implementing the BPE algorithm based on the SentencePiece https://github.com/google/sentencepiece.
    // SentencePiece is under the Apache License 2.0 https://github.com/google/sentencepiece/blob/master/LICENSE

    /// <summary>
    /// SentencePieceBpe is a tokenizer that splits the input into tokens using the SentencePiece Bpe model.
    /// </summary>
    public sealed class SentencePieceBpe : Model
    {
        private const int UninitializedId = -2; // indicate if the symbol contains uninitialized id.

        private readonly Dictionary<StringSpanOrdinalKey, (int Id, float Score, byte Type)> _vocab = new();
        private readonly Dictionary<int, string> _vocabReverse = new();
        private Dictionary<string, int>? _publicVocab;
        private readonly int _maxByteId;
        private readonly int _byteCodeToIdOffset; // offset of mapping byte code to the to the Ids.
        private readonly int _oneByteUtf8EncodingMaxId; // the maximum value of the one byte UTF-8 character.

        internal SentencePieceBpe(ModelProto modelProto, bool addBos, bool addEos) :
                this(modelProto is null ? throw new ArgumentNullException(nameof(modelProto)) : modelProto)
        {
            AddBeginningOfSentence = addBos;
            AddEndOfSentence = addEos;
        }

        private SentencePieceBpe(ModelProto modelProto)
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
        }

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
        public bool TreatWhitespaceAsSuffix { get; }

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
        /// The vocabulary of the model.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocab
        {
            get
            {
                Dictionary<string, int>? publicVocab = Volatile.Read(ref _publicVocab);
                if (publicVocab is null)
                {
                    var vocab = new Dictionary<string, int>();
                    foreach (var item in _vocab)
                    {
                        vocab.Add(item.Key.ToString(), item.Value.Id);
                    }

                    Interlocked.CompareExchange(ref _publicVocab, vocab, null);
                    publicVocab = _publicVocab;
                }

                return publicVocab;
            }
        }

        /// <summary>
        /// Encode a text to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text) => Encode(text, AddBeginningOfSentence, AddEndOfSentence);

        /// <summary>
        /// Encode a text to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        public IReadOnlyList<Token> Encode(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence)
        {
            if (text.Length == 0)
            {
                return Array.Empty<Token>();
            }

            BpeSymbol[] symbols = ArrayPool<BpeSymbol>.Shared.Rent(text.Length);

            Dictionary<(int Index, int Len), (int LeftIndex, int LeftLen, int RightIndex, int RightLen)>? revMerge = Encode(text, symbols);

            List<Token> tokens = new();

            if (addBeginOfSentence)
            {
                tokens.Add(new Token(BeginningOfSentenceId, BeginningOfSentenceToken, (0, 0)));
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
                        tokens.Add(new Token(
                                    id,
                                    GetTokenString(id, symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length, text),
                                    (symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length)));
                    }
                    continue;
                }

                Segment(symbols[index].pieceSpan, text);
            }

            ArrayPool<BpeSymbol>.Shared.Return(symbols);

            if (addEndOfSentence)
            {
                tokens.Add(new Token(EndOfSentenceId, EndOfSentenceToken, (text.Length, 0)));
            }

            return tokens;

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
                            tokens.Add(new Token(id, token, (index + i, 1)));
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
                                tokens.Add(new Token(id, token, (index + i, length)));
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
                    tokens.Add(new Token(id.Id, text.Slice(pieceSpan.Index, pieceSpan.Length).ToString(), (pieceSpan.Index, pieceSpan.Length)));
                    return;
                }

                Segment((merge.LeftIndex, merge.LeftLen), text);
                Segment((merge.RightIndex, merge.RightLen), text);
            }
        }

        /// <summary>
        /// Encode a text to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        public override int EncodeToIds(ReadOnlySpan<char> text, IList<int> accumulatedIds, out int textLength, int maxTokens = int.MaxValue)
            => EncodeToIds(text, AddBeginningOfSentence, AddEndOfSentence, accumulatedIds, out textLength, maxTokens);

        /// <summary>
        /// Encode a text to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        public int EncodeToIds(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, IList<int> accumulatedIds, out int textLength, int maxTokens = int.MaxValue)
        {
            if (maxTokens <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokens), "The maximum number of tokens must be greater than 0.");
            }

            textLength = 0;
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
                        if (!EncodeAsBytes(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), symbols[index].pieceSpan.Index, ref textLength))
                        {
                            break;
                        }
                    }
                    else
                    {
                        if (idsCount < maxTokens)
                        {
                            accumulatedIds.Add(id);
                            textLength += symbols[index].pieceSpan.Length;
                            idsCount++;
                        }
                        else
                        {
                            return idsCount;
                        }
                    }
                    continue;
                }

                if (!Segment(symbols[index].pieceSpan, text, ref textLength))
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
            bool EncodeAsBytes(ReadOnlySpan<char> text, int index, ref int textLength)
            {
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (c <= 0x7F)
                    {
                        if (idsCount < maxTokens)
                        {
                            textLength++;
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

                            textLength += text.Length - i;
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

            bool Segment((int Index, int Length) pieceSpan, ReadOnlySpan<char> text, ref int textLength)
            {
                if (!_vocab.TryGetValue(text.Slice(pieceSpan.Index, pieceSpan.Length), out (int Id, float Score, byte Type) id))
                {
                    return EncodeAsBytes(text.Slice(pieceSpan.Index, pieceSpan.Length), pieceSpan.Index, ref textLength);
                }

                if (id.Type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused ||
                    revMerge is null ||
                    !revMerge.TryGetValue((pieceSpan.Index, pieceSpan.Length), out (int LeftIndex, int LeftLen, int RightIndex, int RightLen) merge))
                {
                    if (idsCount < maxTokens)
                    {
                        accumulatedIds.Add(id.Id);
                        textLength += pieceSpan.Length;
                        idsCount++;
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                return Segment((merge.LeftIndex, merge.LeftLen), text, ref textLength) && Segment((merge.RightIndex, merge.RightLen), text, ref textLength);
            }
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        public override int CountTokens(ReadOnlySpan<char> text, out int textLength, int maxTokens = int.MaxValue) => CountTokens(text, AddBeginningOfSentence, AddEndOfSentence, out textLength, maxTokens);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textIndex">Starting from this index to the end of the text will encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int CountTokensFromEnd(ReadOnlySpan<char> text, out int textIndex, int maxTokens = int.MaxValue) => CountTokensFromEnd(text, AddBeginningOfSentence, AddEndOfSentence, out textIndex, maxTokens);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>The input text has to be normalized before calling this method.</remarks>
        public int CountTokens(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, out int textLength, int maxTokens = int.MaxValue)
        {
            textLength = 0;
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
                        if (!EncodeAsBytes(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), symbols[index].pieceSpan.Index, ref textLength))
                        {
                            break;
                        }
                    }
                    else
                    {
                        if (tokenCount < maxTokens)
                        {
                            tokenCount++;
                            textLength += symbols[index].pieceSpan.Length;
                        }
                        else
                        {
                            break;
                        }
                    }
                    continue;
                }

                if (!Segment(symbols[index].pieceSpan, text, ref textLength))
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
            bool EncodeAsBytes(ReadOnlySpan<char> text, int index, ref int textLength)
            {
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (c <= 0x7F)
                    {
                        if (tokenCount < maxTokens)
                        {
                            tokenCount++;
                            textLength++;
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
                            textLength += text.Length - i;
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

            bool Segment((int Index, int Length) pieceSpan, ReadOnlySpan<char> text, ref int textLength)
            {
                if (!_vocab.TryGetValue(text.Slice(pieceSpan.Index, pieceSpan.Length), out (int Id, float Score, byte Type) id))
                {
                    return EncodeAsBytes(text.Slice(pieceSpan.Index, pieceSpan.Length), pieceSpan.Index, ref textLength);
                }

                if (id.Type != (byte)ModelProto.Types.SentencePiece.Types.Type.Unused ||
                    revMerge is null ||
                    !revMerge.TryGetValue((pieceSpan.Index, pieceSpan.Length), out (int LeftIndex, int LeftLen, int RightIndex, int RightLen) merge))
                {
                    if (tokenCount < maxTokens)
                    {
                        tokenCount++;
                        textLength += pieceSpan.Length;
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                return Segment((merge.LeftIndex, merge.LeftLen), text, ref textLength) && Segment((merge.RightIndex, merge.RightLen), text, ref textLength);
            }
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
        public int CountTokensFromEnd(ReadOnlySpan<char> text, bool addBeginOfSentence, bool addEndOfSentence, out int textIndex, int maxTokens = int.MaxValue)
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

            if (AddBeginningOfSentence)
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
        /// Map the token to encoded id with the option to skip the special tokens.
        /// </summary>
        /// <param name="token">The token to map to Id</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? MapTokenToId(ReadOnlySpan<char> token)
            => _vocab.TryGetValue(token, out (int Id, float Score, byte Type) value) ? value.Id : null;

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? MapIdToToken(int id)
            => _vocabReverse.TryGetValue(id, out string? value) ? value : null;

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="decoder">The optional Decoder to merge the given list of tokens in a string.</param>
        /// <returns>The decoded string.</returns>
        /// <remarks>
        /// The decoder is not used here because the SentencePiece Bpe model knows how to decode the ids in additions to avoid any performance overhead.
        /// </remarks>
        public override string? Decode(IEnumerable<int> ids, TokenizerDecoder? decoder = null)
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

            if (enumerator.Current == BeginningOfSentenceId)
            {
                // escape prefix control tokens.
                if (!enumerator.MoveNext())
                {
                    return string.Empty;
                }
            }

            int bytesCount = -1;
            byte[]? bytesPoolArray = null;
            ValueStringBuilder sb = new(stackalloc char[256]);

            if (enumerator.Current <= _maxByteId)
            {
                // First token is a byte token.

                while (enumerator.Current < _byteCodeToIdOffset)
                {
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
                    sb.Append(token);
                }
            }
            else if (_vocabReverse.TryGetValue(enumerator.Current, out string? token))
            {
                // escape the dummy prefix if needed.
                sb.Append(AddDummyPrefix && !TreatWhitespaceAsSuffix && token.Length > 0 && token[0] == LlamaNormalizer.DummyPrefix ?
                                    token.AsSpan(1) :
                                    token.AsSpan());
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
                        sb.Append(token);
                    }
                }
            }

            if (bytesCount >= 1)
            {
                FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, ref sb);
            }

            if (AddDummyPrefix && TreatWhitespaceAsSuffix && sb.Length > 0 && sb[sb.Length - 1] == LlamaNormalizer.DummyPrefix)
            {
                sb.RemoveLastChar();
            }

            if (bytesPoolArray is not null)
            {
                ArrayPool<byte>.Shared.Return(bytesPoolArray);
            }

            if (charPoolArray is not null)
            {
                ArrayPool<char>.Shared.Return(charPoolArray);
            }

            return sb.ToString(LlamaNormalizer.DummyPrefix, ' ');

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
