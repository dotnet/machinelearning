// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    internal sealed class SentencePieceUnigramModel : SentencePieceBaseModel
    {
        private readonly SortedDictionary<string, int> _vocab;
        private readonly (string Piece, float Score, ModelProto.Types.SentencePiece.Types.Type Type)[] _vocabReverse;
        private readonly DoubleArrayTrie _trie;
        private readonly float _minScore;
        private readonly float _maxScore;
        private const float UnkPenalty = 10.0f;

        public SentencePieceUnigramModel(ModelProto modelProto, bool addBos, bool addEos, IReadOnlyDictionary<string, int>? specialTokens = null) : base(modelProto, addBos, addEos, specialTokens)
        {
            _vocab = new SortedDictionary<string, int>(OrdinalUtf8StringComparer.Instance);

            if (modelProto.TrainerSpec.BosId >= modelProto.Pieces.Count ||
                modelProto.TrainerSpec.EosId >= modelProto.Pieces.Count ||
                modelProto.TrainerSpec.UnkId >= modelProto.Pieces.Count)
            {
                throw new ArgumentException("The BOS, EOS, or UNK token is not present in the vocabulary.");
            }

            _vocabReverse = new (string Piece, float Score, ModelProto.Types.SentencePiece.Types.Type Type)[modelProto.Pieces.Count];

            _minScore = float.MaxValue;
            _maxScore = float.MinValue;

            for (int i = 0; i < modelProto.Pieces.Count; i++)
            {
                if (modelProto.Pieces[i].Type == ModelProto.Types.SentencePiece.Types.Type.Normal ||
                    modelProto.Pieces[i].Type == ModelProto.Types.SentencePiece.Types.Type.UserDefined ||
                    modelProto.Pieces[i].Type == ModelProto.Types.SentencePiece.Types.Type.Unused)
                {
                    string piece = modelProto.Pieces[i].Piece;
                    float score = modelProto.Pieces[i].Score;
                    _vocabReverse[i] = (piece, score, modelProto.Pieces[i].Type);
                    _vocab.Add(piece, i);
                    _minScore = Math.Min(_minScore, score);
                    _maxScore = Math.Max(_maxScore, score);
                }
                else if (modelProto.Pieces[i].Type == ModelProto.Types.SentencePiece.Types.Type.Byte)
                {
                    MaxByteId = i;
                }
                else if (modelProto.Pieces[i].Type == ModelProto.Types.SentencePiece.Types.Type.Unknown)
                {
                    // Ensure the unknown token is cached
                    _vocabReverse[i] = (modelProto.Pieces[i].Piece, modelProto.Pieces[i].Score, ModelProto.Types.SentencePiece.Types.Type.Unknown);
                }
            }

            ByteCodeToIdOffset = _vocab.TryGetValue("<0x00>", out int id) ? id : MaxByteId;
            OneByteUtf8EncodingMaxId = ByteCodeToIdOffset + 0x7F; // 0x7F is the maximum value of the one byte UTF-8 character.
            MaxIdByteFallbackId = ByteCodeToIdOffset + 0xFF; // from <0x00> to <0xFF>.

            _trie = new DoubleArrayTrie(_vocab);

            // Once the trie is built, we need to add the special tokens to the vocabulary.
            // Including these special tokens ensures they are mapped like regular tokens.
            // SentencePiece specifically handles the BOS, EOS, and UNK tokens, while the PAD token is optional.

            Debug.Assert(modelProto.TrainerSpec.UnkId >= 0);
            Debug.Assert(modelProto.TrainerSpec.BosId >= 0);
            Debug.Assert(modelProto.TrainerSpec.EosId >= 0);

            _vocab[modelProto.TrainerSpec.UnkPiece] = modelProto.TrainerSpec.UnkId;
            _vocab[modelProto.TrainerSpec.BosPiece] = modelProto.TrainerSpec.BosId;
            _vocab[modelProto.TrainerSpec.EosPiece] = modelProto.TrainerSpec.EosId;

            _vocabReverse[modelProto.TrainerSpec.BosId] = (modelProto.TrainerSpec.BosPiece, 0f, ModelProto.Types.SentencePiece.Types.Type.Control);
            _vocabReverse[modelProto.TrainerSpec.EosId] = (modelProto.TrainerSpec.EosPiece, 0f, ModelProto.Types.SentencePiece.Types.Type.Control);
            _vocabReverse[modelProto.TrainerSpec.UnkId] = (modelProto.TrainerSpec.UnkPiece, 0f, ModelProto.Types.SentencePiece.Types.Type.Unknown);

            if (modelProto.TrainerSpec.PadId >= 0)
            {
                _vocab[modelProto.TrainerSpec.PadPiece] = modelProto.TrainerSpec.PadId;
                _vocabReverse[modelProto.TrainerSpec.PadId] = (modelProto.TrainerSpec.PadPiece, 0f, ModelProto.Types.SentencePiece.Types.Type.Control);
            }
        }

        public override IReadOnlyDictionary<string, int> Vocabulary => new ReadOnlyDictionary<string, int>(_vocab);

        public int MaxIdByteFallbackId { get; }

        public override IReadOnlyList<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, out string? normalizedText, bool addBeginningOfSentence, bool addEndOfSentence, bool considerNormalization)
        {
            ReadOnlySpan<char> textToEncode = string.IsNullOrEmpty(text) ? textSpan : text.AsSpan();
            if (textToEncode.IsEmpty)
            {
                normalizedText = string.Empty;
                return Array.Empty<EncodedToken>();
            }

            List<EncodedToken> tokens = new();

            // Rent a buffer that approximately enough to hold the Utf8 encoded bytes, the normalization of the encoded buffer, and some extra memory to for encoding results.
            int[] buffer = ArrayPool<int>.Shared.Rent(textToEncode.Length * 3);

            // Hold the Utf16 normalized string.
            char[] normalizedString = ArrayPool<char>.Shared.Rent(textToEncode.Length + 2);

            if (SpecialTokensRegex is not null)
            {
                EncodeToTokensWithSpecialTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, considerNormalization, tokens, buffer, ref normalizedString, out normalizedText);
            }
            else
            {
                EncodeToTokensWithoutSpecialTokens(textToEncode, addBeginningOfSentence, addEndOfSentence, considerNormalization, tokens, buffer, ref normalizedString, out normalizedText);
            }

            ArrayPool<char>.Shared.Return(normalizedString);
            ArrayPool<int>.Shared.Return(buffer);

            return tokens;
        }

        public override bool TryMapIdToToken(int id, out string? token)
        {
            if ((uint)id >= (uint)(_vocabReverse.Length))
            {
                token = null;
                return false;
            }

            token = _vocabReverse[id].Piece;
            return true;
        }

        private void StoreNormalizedTextFromEnd(ReadOnlySpan<char> text, ref char[] normalizedString, ref int normalizedStringCountFromEnd)
        {
            int remainingLength = normalizedString.Length - normalizedStringCountFromEnd;
            if (text.Length > remainingLength)
            {
                char[] utf16NormalizedString = ArrayPool<char>.Shared.Rent(normalizedString.Length << 1);
                normalizedString.AsSpan().Slice(normalizedString.Length - normalizedStringCountFromEnd).CopyTo(utf16NormalizedString.AsSpan(utf16NormalizedString.Length - normalizedStringCountFromEnd));
                ArrayPool<char>.Shared.Return(normalizedString);
                normalizedString = utf16NormalizedString;
            }

            text.CopyTo(normalizedString.AsSpan(normalizedString.Length - normalizedStringCountFromEnd - text.Length));
            normalizedStringCountFromEnd += text.Length;
        }

        private void StoreNormalizedTextFromEnd(ReadOnlySpan<byte> utf8Bytes, ref char[] normalizedString, ref int normalizedStringCountFromEnd)
        {
            int remainingLength = normalizedString.Length - normalizedStringCountFromEnd;
            int expectedCount = Helpers.GetUtf16LengthFromUtf8Bytes(utf8Bytes);

            if (expectedCount > remainingLength)
            {
                char[] utf16NormalizedString = ArrayPool<char>.Shared.Rent(normalizedString.Length << 1);
                normalizedString.AsSpan().Slice(normalizedString.Length - normalizedStringCountFromEnd).CopyTo(utf16NormalizedString.AsSpan(utf16NormalizedString.Length - normalizedStringCountFromEnd));
                ArrayPool<char>.Shared.Return(normalizedString);
                normalizedString = utf16NormalizedString;
            }

            bool res = Helpers.ConvertUtf8ToUtf16(utf8Bytes, normalizedString.AsSpan(normalizedString.Length - normalizedStringCountFromEnd - expectedCount), out int bytesConsumed, out int charsWritten);
            Debug.Assert(res);
            Debug.Assert(bytesConsumed == utf8Bytes.Length);
            Debug.Assert(charsWritten == expectedCount);
            normalizedStringCountFromEnd += expectedCount;
        }

        private void StoreNormalizedText(ReadOnlySpan<char> text, ref char[] normalizedString, ref int normalizedStringIndex)
        {
            Span<char> utf16NormalizedString = normalizedString.AsSpan().Slice(normalizedStringIndex);

            if (text.Length > utf16NormalizedString.Length)
            {
                Helpers.ArrayPoolGrow(ref normalizedString, normalizedString.Length << 1);
                utf16NormalizedString = normalizedString.AsSpan().Slice(normalizedStringIndex);
            }

            text.CopyTo(utf16NormalizedString);
            normalizedStringIndex += text.Length;
        }

        private void StoreNormalizedText(ReadOnlySpan<byte> normalizationSpan, ref char[] normalizedString, ref int normalizedStringIndex)
        {
            Span<char> normalizedUtf16Span = normalizedString.AsSpan().Slice(normalizedStringIndex);
            if (Encoding.UTF8.GetMaxCharCount(normalizationSpan.Length) > normalizedUtf16Span.Length)
            {
                Helpers.ArrayPoolGrow(ref normalizedString, normalizedString.Length << 1);
                normalizedUtf16Span = normalizedString.AsSpan().Slice(normalizedStringIndex);
            }

            bool res = Helpers.ConvertUtf8ToUtf16(normalizationSpan, normalizedUtf16Span, out int bytesConsumed, out int charsWritten);
            Debug.Assert(res);
            normalizedStringIndex += charsWritten;
        }

        private void EncodeToTokensWithSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerNormalization,
                        List<EncodedToken> tokens,
                        int[] buffer,
                        ref char[] normalizedString,
                        out string? normalizedText)
        {
            Debug.Assert(SpecialTokensRegex is not null);

            if (addBeginningOfSentence)
            {
                tokens.Add(new EncodedToken(BeginningOfSentenceId, BeginningOfSentenceToken, new Range(0, 0)));
            }

            int currentOffset = 0;
            int progressOffset = 0;
            int normalizedStringIndex = 0;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, SpecialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    EncodeToTokensInternal(text.Slice(currentOffset, Offset - currentOffset), considerNormalization, ref progressOffset, tokens, buffer, ref normalizedString, ref normalizedStringIndex);
                }

                if (InternalSpecialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
                {
                    tokens.Add(new EncodedToken(id, SpecialTokensReverse![id], new Range(progressOffset, progressOffset + Length)));
                    progressOffset += Length;

                    StoreNormalizedText(text.Slice(Offset, Length), ref normalizedString, ref normalizedStringIndex);
                }

                currentOffset = Offset + Length;
            }

            if (currentOffset < text.Length)
            {
                EncodeToTokensInternal(text.Slice(currentOffset), considerNormalization, ref progressOffset, tokens, buffer, ref normalizedString, ref normalizedStringIndex);
            }

            if (addEndOfSentence)
            {
                tokens.Add(new EncodedToken(EndOfSentenceId, EndOfSentenceToken, new Range(progressOffset, progressOffset)));
            }

            normalizedText = normalizedString.AsSpan().Slice(0, normalizedStringIndex).ToString();
        }

        private void EncodeToTokensWithoutSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerNormalization,
                        List<EncodedToken> tokens,
                        int[] buffer,
                        ref char[] normalizedString,
                        out string? normalizedText)
        {
            if (addBeginningOfSentence)
            {
                tokens.Add(new EncodedToken(BeginningOfSentenceId, BeginningOfSentenceToken, new Range(0, 0)));
            }

            int progressOffset = 0;
            int normalizedStringIndex = 0;

            EncodeToTokensInternal(text, considerNormalization, ref progressOffset, tokens, buffer, ref normalizedString, ref normalizedStringIndex);

            if (addEndOfSentence)
            {
                tokens.Add(new EncodedToken(EndOfSentenceId, EndOfSentenceToken, new Range(progressOffset, progressOffset)));
            }

            normalizedText = normalizedString.AsSpan().Slice(0, normalizedStringIndex).ToString();
        }

        private void NormalizeText(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        int[] buffer,
                        out byte[]? normalizedArrayPool,
                        out Span<byte> normalizationSpan)
        {
            Debug.Assert(Encoding.UTF8.GetMaxByteCount(text.Length) * 3 <= buffer.Length * sizeof(int));
            Span<byte> byteSpan = MemoryMarshal.AsBytes(buffer.AsSpan());

            // Unigram is currently working with Utf8 encoded bytes.
            // if considerNormalization is true, the utf8 encoded bytes will be normalized to utf8 bytes too.
            int byteCount = Helpers.GetUtf8Bytes(text, byteSpan);
            normalizationSpan = byteSpan.Slice(byteCount);

            Debug.Assert(normalizationSpan.Length >= (byteCount << 1));
            normalizedArrayPool = null;

            if (considerNormalization)
            {
                int normalizationCount = Normalizer!.Normalize(byteSpan.Slice(0, byteCount), ref normalizationSpan, ref normalizedArrayPool);
                normalizationSpan = normalizationSpan.Slice(0, normalizationCount);
                if (normalizationCount == 0)
                {
                    if (normalizedArrayPool is not null)
                    {
                        ArrayPool<byte>.Shared.Return(normalizedArrayPool);
                        normalizedArrayPool = null;
                    }

                    return;
                }
            }
            else
            {
                normalizationSpan = byteSpan.Slice(0, byteCount);
            }
        }

        private void EncodeToTokensInternal(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        ref int tokensOffset,
                        List<EncodedToken> tokens,
                        int[] buffer,
                        ref char[] normalizedString,
                        ref int normalizedStringIndex)
        {
            //
            // Normalize text
            //

            NormalizeText(text, considerNormalization, buffer, out byte[]? normalizedArrayPool, out Span<byte> normalizationSpan);

            //
            // Encode using Unigram algorithm
            //

            BestPathNode[] bestPathEndsAt = ArrayPool<BestPathNode>.Shared.Rent(normalizationSpan.Length + 1);

            Encode(normalizationSpan, bestPathEndsAt);

            //
            // Fill the results
            //

            // Backtrack to identify the best path.
            int insertionStartPosition = tokens.Count;
            int endsAt = normalizationSpan.Length;
            bool unknownEncountered = false;

            while (endsAt > 0)
            {
                ref BestPathNode node = ref bestPathEndsAt[endsAt];

                string stringToken = node.Id == UnknownId ? Helpers.GetString(normalizationSpan.Slice(node.StartsAt, endsAt - node.StartsAt)) : _vocabReverse[node.Id].Piece;
                int tokenLength = stringToken.Length;

                tokens.Add(new EncodedToken(node.Id, stringToken, new Range(0, tokenLength))); // we will update the range later.
                endsAt = node.StartsAt;
                unknownEncountered = unknownEncountered || node.Id == UnknownId;
            }

            int start = insertionStartPosition;
            int end = tokens.Count - 1;

            // Reverse the stored tokens and fix the encoded tokens offset.
            while (start < end)
            {
                EncodedToken temp = tokens[start];
                tokens[start] = tokens[end];
                tokens[end] = temp;

                int tokenLength = tokens[start].Offset.End.Value;
                // Fix the offsets
                tokens[start] = new EncodedToken(tokens[start].Id, tokens[start].Value, new Range(tokensOffset, tokensOffset + tokenLength));
                tokensOffset += tokenLength;

                start++;
                end--;
            }

            while (start < tokens.Count)
            {
                int tokenLength = tokens[start].Offset.End.Value;
                // Fix the offsets
                tokens[start] = new EncodedToken(tokens[start].Id, tokens[start].Value, new Range(tokensOffset, tokensOffset + tokenLength));
                tokensOffset += tokenLength;
                start++;
            }

            StoreNormalizedText(normalizationSpan, ref normalizedString, ref normalizedStringIndex);

            if (ByteFallback && unknownEncountered)
            {
                FallbackToByteEncoding(normalizedString, tokens, insertionStartPosition);
            }

            ArrayPool<BestPathNode>.Shared.Return(bestPathEndsAt);
            if (normalizedArrayPool is not null)
            {
                ArrayPool<byte>.Shared.Return(normalizedArrayPool);
            }
        }

        private void FallbackToByteEncoding(ReadOnlySpan<char> normalizationSpan, List<EncodedToken> tokens, int insertionStartPosition)
        {
            Span<byte> destination = stackalloc byte[4];

            while (insertionStartPosition < tokens.Count)
            {
                if (tokens[insertionStartPosition].Id == UnknownId)
                {
                    int offsetStart = tokens[insertionStartPosition].Offset.Start.Value;
                    int tokenLength = tokens[insertionStartPosition].Offset.End.Value - offsetStart;

                    tokens.RemoveAt(insertionStartPosition);

                    int charLength = 0;
                    for (int i = 0; i < tokenLength; i += charLength)
                    {
                        int codepointLength = Helpers.EncodeNextUtf8(normalizationSpan.Slice(offsetStart), destination);
                        charLength = codepointLength == 4 ? 2 : 1;

                        Debug.Assert(codepointLength > 0);

                        int id = ByteCodeToIdOffset + destination[0];
                        tokens.Insert(insertionStartPosition++, new EncodedToken(id, _vocabReverse[id].Piece, new Range(offsetStart, offsetStart + charLength)));

                        for (int j = 1; j < codepointLength; j++)
                        {
                            id = ByteCodeToIdOffset + destination[j];
                            tokens.Insert(insertionStartPosition++, new EncodedToken(id, _vocabReverse[id].Piece, new Range(offsetStart + charLength, offsetStart + charLength)));
                        }

                        offsetStart += charLength;
                    }

                    continue;
                }

                insertionStartPosition++;
            }
        }

        private struct BestPathNode
        {
            public BestPathNode()
            {
                Id = -1;
                BestPathScore = 0f;
                StartsAt = -1;
            }

            // The vocab id. (maybe -1 for UNK)
            public int Id { get; set; }

            // The total score of the best path ending at this node.
            public float BestPathScore { get; set; }

            // The starting position (in utf-8) of this node. The entire best path can be constructed by backtracking along this link.
            public int StartsAt { get; set; }
        };

        private void Encode(ReadOnlySpan<byte> normalized, Span<BestPathNode> bestPathEndsAt)
        {
            Debug.Assert(normalized.Length > 0);

            int size = normalized.Length;
            float unkScore = _minScore - UnkPenalty;

            Debug.Assert(bestPathEndsAt.Length >= size + 1);

            // The ends are exclusive.
            for (int i = 0; i < size + 1; i++)
            {
                bestPathEndsAt[i] = new BestPathNode();
            }

            // Generate lattice on-the-fly (not stored) and update best_path_ends_at.
            int startsAt = 0;

            while (startsAt < size)
            {
                int nodePos = 0;
                int keyPos = startsAt;
                float bestPathScoreTillHere = bestPathEndsAt[startsAt].BestPathScore;
                bool hasSingleNode = false;
                int mbLen = Helpers.OneCharLen(normalized[startsAt]);
                while (keyPos < size)
                {
                    int ret = _trie.Traverse(normalized, ref nodePos, ref keyPos, keyPos + 1);
                    if (ret == -2)
                    {
                        break;
                    }

                    if (ret >= 0)
                    {
                        if (_vocabReverse[ret].Type == ModelProto.Types.SentencePiece.Types.Type.Unused)
                        {
                            continue;
                        }

                        // Update the best path node.
                        ref BestPathNode targetNode = ref bestPathEndsAt[keyPos];
                        int length = keyPos - startsAt;

                        // User defined symbol receives extra bonus to always be selected.
                        float score = _vocabReverse[ret].Type == ModelProto.Types.SentencePiece.Types.Type.UserDefined ? length * _maxScore - 0.1f : _vocabReverse[ret].Score;
                        float candidateBestPathScore = score + bestPathScoreTillHere;

                        if (targetNode.StartsAt == -1 || candidateBestPathScore > targetNode.BestPathScore)
                        {
                            targetNode.BestPathScore = candidateBestPathScore;
                            targetNode.StartsAt = startsAt;
                            targetNode.Id = ret;
                        }

                        if (!hasSingleNode && length == mbLen)
                        {
                            hasSingleNode = true;
                        }
                    }
                }

                if (!hasSingleNode)
                {
                    ref BestPathNode targetNode = ref bestPathEndsAt[startsAt + mbLen];
                    float candidateBestPathScore = unkScore + bestPathScoreTillHere;

                    if (targetNode.StartsAt == -1 || candidateBestPathScore > targetNode.BestPathScore)
                    {
                        targetNode.BestPathScore = candidateBestPathScore;
                        targetNode.StartsAt = startsAt;
                        targetNode.Id = UnknownId;
                    }
                }

                // Move by one unicode character.
                startsAt += mbLen;
            }
        }

        public override IReadOnlyList<int> EncodeToIds(
                                            string? text,
                                            ReadOnlySpan<char> textSpan,
                                            bool addBeginningOfSentence,
                                            bool addEndOfSentence,
                                            bool considerNormalization,
                                            out string? normalizedText,
                                            out int charsConsumed,
                                            int maxTokenCount = int.MaxValue)
        {
            ReadOnlySpan<char> textToEncode = string.IsNullOrEmpty(text) ? textSpan : text.AsSpan();

            if (textToEncode.IsEmpty || maxTokenCount <= 0)
            {
                normalizedText = null;
                charsConsumed = 0;
                return Array.Empty<int>();
            }

            List<int>? ids = new();

            if (addBeginningOfSentence)
            {
                ids.Add(BeginningOfSentenceId);
                if (maxTokenCount == 1)
                {
                    normalizedText = null;
                    charsConsumed = 0;
                    return ids; // done. no more space for anything else.
                }
            }

            // Rent a buffer that approximately enough to hold the Utf8 encoded bytes, the normalization of the encoded buffer, and some extra memory to for encoding results.
            int[] buffer = ArrayPool<int>.Shared.Rent(textToEncode.Length * 3);

            // when maxTokenCount == int.MaxValue we don't need to return the normalized string as most likely we can handle the whole input text without need to continuation.
            char[]? normalizedString = maxTokenCount == int.MaxValue ? null : ArrayPool<char>.Shared.Rent(textToEncode.Length + 2);

            if (SpecialTokensRegex is not null)
            {
                EncodeToIdsWithSpecialTokens(textToEncode, considerNormalization, ids, buffer, ref normalizedString, out normalizedText, out charsConsumed, maxTokenCount);
            }
            else
            {
                EncodeToIdsWithoutSpecialTokens(textToEncode, considerNormalization, ids, buffer, ref normalizedString, out normalizedText, out charsConsumed, maxTokenCount);
            }

            if (addEndOfSentence && ids.Count < maxTokenCount)
            {
                ids.Add(EndOfSentenceId);
            }

            if (normalizedString is not null)
            {
                ArrayPool<char>.Shared.Return(normalizedString);
            }

            ArrayPool<int>.Shared.Return(buffer);

            return ids;
        }

        private void StoreNormalizedText(ReadOnlySpan<char> text, bool considerNormalization, int[] buffer, ref char[]? normalizedString, ref int normalizedStringIndex)
        {
            Debug.Assert(normalizedString is not null);

            if (!considerNormalization)
            {
                StoreNormalizedText(text, ref normalizedString!, ref normalizedStringIndex);
            }
            else
            {
                NormalizeText(text, considerNormalization, buffer, out byte[]? normalizedArrayPool, out Span<byte> normalizationSpan);
                StoreNormalizedText(normalizationSpan, ref normalizedString!, ref normalizedStringIndex);
                if (normalizedArrayPool is not null)
                {
                    ArrayPool<byte>.Shared.Return(normalizedArrayPool);
                }
            }
        }

        private void EncodeToIdsWithSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        List<int> ids,
                        int[] buffer,
                        ref char[]? normalizedString,
                        out string? normalizedText,
                        out int charsConsumed,
                        int maxTokenCount)
        {
            Debug.Assert(SpecialTokensRegex is not null);
            Debug.Assert(maxTokenCount > 0);

            charsConsumed = 0;
            normalizedText = null;

            int currentOffset = 0;
            int normalizedStringIndex = 0;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, SpecialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    if (ids.Count >= maxTokenCount)
                    {
                        if (normalizedString is not null)
                        {
                            StoreNormalizedText(text.Slice(currentOffset, Offset - currentOffset), considerNormalization, buffer, ref normalizedString, ref normalizedStringIndex);
                        }
                    }
                    else
                    {
                        EncodeToIdsInternal(text.Slice(currentOffset, Offset - currentOffset), considerNormalization, ids, buffer, ref normalizedString, ref normalizedStringIndex, ref charsConsumed, maxTokenCount);
                    }
                }

                if (InternalSpecialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
                {
                    if (normalizedString is not null)
                    {
                        StoreNormalizedText(text.Slice(Offset, Length), ref normalizedString, ref normalizedStringIndex);
                    }

                    if (ids.Count < maxTokenCount)
                    {
                        ids.Add(id); // special token id

                        charsConsumed += Length;
                    }
                }

                currentOffset = Offset + Length;
            }

            if (currentOffset < text.Length)
            {
                if (ids.Count < maxTokenCount)
                {
                    EncodeToIdsInternal(text.Slice(currentOffset), considerNormalization, ids, buffer, ref normalizedString, ref normalizedStringIndex, ref charsConsumed, maxTokenCount);
                }
                else if (normalizedString is not null)
                {
                    StoreNormalizedText(text.Slice(currentOffset), considerNormalization, buffer, ref normalizedString, ref normalizedStringIndex);
                }
            }

            if (normalizedString is not null)
            {
                normalizedText = normalizedString.AsSpan().Slice(0, normalizedStringIndex).ToString();
            }
        }

        private void EncodeToIdsWithoutSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        List<int> ids,
                        int[] buffer,
                        ref char[]? normalizedString,
                        out string? normalizedText,
                        out int charsConsumed,
                        int maxTokenCount)
        {
            charsConsumed = 0;
            normalizedText = null;
            int normalizedStringIndex = 0;

            EncodeToIdsInternal(text, considerNormalization, ids, buffer, ref normalizedString, ref normalizedStringIndex, ref charsConsumed, maxTokenCount);

            if (normalizedString is not null)
            {
                normalizedText = normalizedString.AsSpan().Slice(0, normalizedStringIndex).ToString();
            }
        }

        private void FallbackToByteEncoding(List<int> ids, ReadOnlySpan<byte> normalizationSpan, (int IdsIndex, int Utf8Index, int Utf8Length)[] unknownTokensTracking, int unknownTokensCount)
        {
            Debug.Assert(unknownTokensCount > 0);
            Debug.Assert(unknownTokensTracking is not null && unknownTokensTracking.Length >= unknownTokensCount);

            // validate reverse ordered.
            Debug.Assert(unknownTokensCount == 1 || unknownTokensTracking![0].IdsIndex > unknownTokensTracking![1].IdsIndex);

            int accumulatedOffsets = 0;
            for (int i = unknownTokensCount - 1; i >= 0; i--)
            {
                unknownTokensTracking![i].IdsIndex += accumulatedOffsets;
                (int IdsIndex, int Utf8Index, int Utf8Length) = unknownTokensTracking![i];

                if (IdsIndex >= ids.Count)
                {
                    continue; // already removed.
                }

                Debug.Assert(ids[IdsIndex] == UnknownId);

                // Replace the Unknown id entry with the byte encoding.
                ids.RemoveAt(IdsIndex);

                for (int j = Utf8Length - 1; j >= 0; j--)
                {
                    ids.Insert(IdsIndex, ByteCodeToIdOffset + normalizationSpan[Utf8Index + j]);
                }

                // -1 because we removed the Unknown id entry.
                accumulatedOffsets += Utf8Length - 1;
            }
        }

        private void EncodeToIdsInternal(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        List<int> ids,
                        int[] buffer,
                        ref char[]? normalizedString,
                        ref int normalizedStringIndex,
                        ref int charsConsumed,
                        int maxTokenCount)
        {
            if (ids.Count >= maxTokenCount)
            {
                return;
            }

            //
            // Normalize the input text.
            //

            NormalizeText(text, considerNormalization, buffer, out byte[]? normalizedArrayPool, out Span<byte> normalizationSpan);

            //
            // Do the actual encoding
            //

            BestPathNode[] bestPathEndsAt = ArrayPool<BestPathNode>.Shared.Rent(normalizationSpan.Length + 1);

            Encode(normalizationSpan, bestPathEndsAt);

            // Backtrack to identify the best path.
            int insertionStartPosition = ids.Count;
            int endsAt = normalizationSpan.Length;

            int unknownTokensCount = 0;
            (int IdsIndex, int Utf8Index, int Utf8Length)[]? unknownTokensTracking = null;
            bool needToTrackUnknown = ByteFallback || maxTokenCount != int.MaxValue;

            while (endsAt > 0)
            {
                ref BestPathNode node = ref bestPathEndsAt[endsAt];

                ids.Add(node.Id);

                if (node.Id == UnknownId && needToTrackUnknown)
                {
                    unknownTokensCount++;
                    if (unknownTokensTracking is null)
                    {
                        unknownTokensTracking = ArrayPool<(int IdsIndex, int Utf8Index, int Utf8Length)>.Shared.Rent(10);
                    }
                    else if (unknownTokensTracking.Length == unknownTokensCount)
                    {
                        Helpers.ArrayPoolGrow(ref unknownTokensTracking, unknownTokensCount << 1);
                    }

                    unknownTokensTracking[unknownTokensCount - 1] = (ids.Count - 1, node.StartsAt, endsAt - node.StartsAt);
                }

                endsAt = node.StartsAt;
            }

            ArrayPool<BestPathNode>.Shared.Return(bestPathEndsAt);

            ids.Reverse(insertionStartPosition, ids.Count - insertionStartPosition);

            if (unknownTokensCount > 0)
            {
                Debug.Assert(unknownTokensTracking is not null && unknownTokensTracking.Length >= unknownTokensCount);

                int end = ids.Count - 1;

                // Fix the id indexes after swapping
                for (int i = 0; i < unknownTokensCount; i++)
                {
                    unknownTokensTracking![i].IdsIndex = insertionStartPosition + (end - unknownTokensTracking![i].IdsIndex);
                }
            }

            //
            // Handle maxTokenCount
            //

            if (maxTokenCount == int.MaxValue)
            {
                Debug.Assert(unknownTokensCount == 0 && unknownTokensTracking is null);

                if (ByteFallback && unknownTokensCount > 0)
                {
                    Debug.Assert(unknownTokensTracking is not null && unknownTokensTracking.Length >= unknownTokensCount);
                    FallbackToByteEncoding(ids, normalizationSpan, unknownTokensTracking!, unknownTokensCount);
                }

                // sure we should be consumed the whole text.
                charsConsumed += text.Length;

                if (normalizedArrayPool is not null)
                {
                    ArrayPool<byte>.Shared.Return(normalizedArrayPool);
                }

                // done't bother storing the normalized string as we return null when we can handle the whole input text.
                Debug.Assert(normalizedString is null);

                return;
            }

            // Check if we need to truncate the tokens. and calculate the accurate consumed characters count.
            int index = insertionStartPosition;
            int addedTokensCount = 0;

            while (index < ids.Count && index + addedTokensCount < maxTokenCount)
            {
                if (ids[index] == UnknownId)
                {
                    Debug.Assert(unknownTokensCount > 0 && unknownTokensTracking is not null && unknownTokensTracking.Length >= unknownTokensCount);

                    int j = 0;
                    for (; j < unknownTokensCount; j++)
                    {
                        if (unknownTokensTracking![j].IdsIndex == index)
                        {
                            break;
                        }
                    }

                    Debug.Assert(j < unknownTokensCount);

                    ReadOnlySpan<byte> utf8UnknownBytes = normalizationSpan.Slice(unknownTokensTracking![j].Utf8Index, unknownTokensTracking![j].Utf8Length);

                    if (ByteFallback)
                    {
                        if (index + utf8UnknownBytes.Length > maxTokenCount)
                        {
                            break; // not enough space
                        }

                        addedTokensCount += utf8UnknownBytes.Length - 1;
                    }

                    charsConsumed += Helpers.GetUtf16LengthFromUtf8Bytes(utf8UnknownBytes);
                }
                else
                {
                    charsConsumed += _vocabReverse[ids[index]].Piece.Length;
                }

                index++;
            }

            if (index < ids.Count)
            {
                ids.RemoveRange(index, ids.Count - index);
            }

            if (unknownTokensCount > 0 && ByteFallback)
            {
                Debug.Assert(unknownTokensTracking is not null && unknownTokensTracking.Length >= unknownTokensCount);
                FallbackToByteEncoding(ids, normalizationSpan, unknownTokensTracking!, unknownTokensCount);
            }

            //
            // Create the normalized string.
            //

            if (normalizedString is not null)
            {
                StoreNormalizedText(normalizationSpan, ref normalizedString, ref normalizedStringIndex);
            }

            if (unknownTokensTracking is not null)
            {
                ArrayPool<(int IdsIndex, int Utf8Index, int Utf8Length)>.Shared.Return(unknownTokensTracking);
            }

            if (normalizedArrayPool is not null)
            {
                ArrayPool<byte>.Shared.Return(normalizedArrayPool);
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
            ReadOnlySpan<char> textToEncode = string.IsNullOrEmpty(text) ? textSpan : text.AsSpan();

            if (textToEncode.IsEmpty || maxTokenCount <= 0)
            {
                normalizedText = null;
                charsConsumed = 0;
                return 0;
            }

            int tokenCount = 0;

            if (addBeginningOfSentence)
            {
                tokenCount++;

                if (maxTokenCount == 1)
                {
                    normalizedText = null;
                    charsConsumed = 0;
                    return tokenCount;
                }
            }

            // Rent a buffer that approximately enough to hold the Utf8 encoded bytes, the normalization of the encoded buffer, and some extra memory to for encoding results.
            int[] buffer = ArrayPool<int>.Shared.Rent(textToEncode.Length * 3);

            // when maxTokenCount == int.MaxValue we don't need to return the normalized string as most likely we can handle the whole input text without need to continuation.
            char[]? normalizedString = maxTokenCount == int.MaxValue ? null : ArrayPool<char>.Shared.Rent(textToEncode.Length + 2);

            if (SpecialTokensRegex is not null)
            {
                CountTokensWithSpecialTokens(textToEncode, considerNormalization, ref tokenCount, buffer, ref normalizedString, out normalizedText, out charsConsumed, maxTokenCount);
            }
            else
            {
                CountTokensWithoutSpecialTokens(textToEncode, considerNormalization, ref tokenCount, buffer, ref normalizedString, out normalizedText, out charsConsumed, maxTokenCount);
            }

            if (addEndOfSentence && tokenCount < maxTokenCount)
            {
                tokenCount++;
            }

            if (normalizedString is not null)
            {
                ArrayPool<char>.Shared.Return(normalizedString);
            }

            ArrayPool<int>.Shared.Return(buffer);

            return tokenCount;
        }

        private void CountTokensWithSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        ref int tokenCount,
                        int[] buffer,
                        ref char[]? normalizedString,
                        out string? normalizedText,
                        out int charsConsumed,
                        int maxTokenCount)
        {
            Debug.Assert(SpecialTokensRegex is not null);
            Debug.Assert(maxTokenCount > 0);

            charsConsumed = 0;
            normalizedText = null;

            int currentOffset = 0;
            int normalizedStringIndex = 0;

            foreach ((int Offset, int Length) in PreTokenizer.SplitText(text, SpecialTokensRegex!))
            {
                if (Offset > currentOffset)
                {
                    if (tokenCount >= maxTokenCount)
                    {
                        if (normalizedString is not null)
                        {
                            StoreNormalizedText(text.Slice(currentOffset, Offset - currentOffset), considerNormalization, buffer, ref normalizedString, ref normalizedStringIndex);
                        }
                    }
                    else
                    {
                        CountTokensInternal(text.Slice(currentOffset, Offset - currentOffset), considerNormalization, ref tokenCount, buffer, ref normalizedString, ref normalizedStringIndex, ref charsConsumed, maxTokenCount);
                    }
                }

                if (InternalSpecialTokens!.TryGetValue(text.Slice(Offset, Length), out int id))
                {
                    if (normalizedString is not null)
                    {
                        StoreNormalizedText(text.Slice(Offset, Length), ref normalizedString, ref normalizedStringIndex);
                    }

                    if (tokenCount < maxTokenCount)
                    {
                        tokenCount++; // special token id
                        charsConsumed += Length;
                    }
                }

                currentOffset = Offset + Length;
            }

            if (currentOffset < text.Length && tokenCount < maxTokenCount)
            {
                if (tokenCount < maxTokenCount)
                {
                    CountTokensInternal(text.Slice(currentOffset), considerNormalization, ref tokenCount, buffer, ref normalizedString, ref normalizedStringIndex, ref charsConsumed, maxTokenCount);
                }
                else if (normalizedString is not null)
                {
                    StoreNormalizedText(text.Slice(currentOffset), considerNormalization, buffer, ref normalizedString, ref normalizedStringIndex);
                }
            }

            if (normalizedString is not null)
            {
                normalizedText = normalizedString.AsSpan().Slice(0, normalizedStringIndex).ToString();
            }
        }

        private void CountTokensWithoutSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        ref int tokenCount,
                        int[] buffer,
                        ref char[]? normalizedString,
                        out string? normalizedText,
                        out int charsConsumed,
                        int maxTokenCount)
        {
            charsConsumed = 0;
            normalizedText = null;
            int normalizedStringIndex = 0;

            CountTokensInternal(text, considerNormalization, ref tokenCount, buffer, ref normalizedString, ref normalizedStringIndex, ref charsConsumed, maxTokenCount);

            if (normalizedString is not null)
            {
                normalizedText = normalizedString.AsSpan().Slice(0, normalizedStringIndex).ToString();
            }
        }

        private void CountTokensInternal(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        ref int tokenCount,
                        int[] buffer,
                        ref char[]? normalizedString,
                        ref int normalizedStringIndex,
                        ref int charsConsumed,
                        int maxTokenCount)
        {
            //
            // Normalize the input text.
            //

            NormalizeText(text, considerNormalization, buffer, out byte[]? normalizedArrayPool, out Span<byte> normalizationSpan);

            //
            // Do the actual encoding
            //

            BestPathNode[] bestPathEndsAt = ArrayPool<BestPathNode>.Shared.Rent(normalizationSpan.Length + 1);

            Encode(normalizationSpan, bestPathEndsAt);

            // Need to check for unknown tokens and update the charsConsumed.

            (int Id, int UtfStartOffset, int Utf8Length)[] ids = ArrayPool<(int Id, int UtfStartOffset, int Utf8Length)>.Shared.Rent(bestPathEndsAt.Length);

            // Backtrack to identify the best path.
            int idsIndex = ids.Length - 1;
            int endsAt = normalizationSpan.Length;

            bool unknownEncountered = false;
            while (endsAt > 0)
            {
                ref BestPathNode node = ref bestPathEndsAt[endsAt];

                ids[idsIndex--] = (node.Id, node.StartsAt, endsAt - node.StartsAt);

                unknownEncountered = unknownEncountered || node.Id == UnknownId;

                endsAt = node.StartsAt;
            }

            idsIndex++; // Index starting the collected tokens.

            ArrayPool<BestPathNode>.Shared.Return(bestPathEndsAt);

            if ((!ByteFallback || !unknownEncountered) && (maxTokenCount == int.MaxValue || (tokenCount + ids.Length - idsIndex <= maxTokenCount)))
            {
                // sure we should be consumed the whole text.
                charsConsumed += Helpers.GetUtf16LengthFromUtf8Bytes(normalizationSpan);
                tokenCount += ids.Length - idsIndex;

                if (normalizedString is not null)
                {
                    StoreNormalizedText(normalizationSpan, ref normalizedString, ref normalizedStringIndex);
                }

                ArrayPool<(int Id, int UtfStartOffset, int Utf8Length)>.Shared.Return(ids);

                if (normalizedArrayPool is not null)
                {
                    ArrayPool<byte>.Shared.Return(normalizedArrayPool);
                }

                return;
            }

            // Manually count the tokens up to the max.
            for (int i = idsIndex; tokenCount < maxTokenCount && i < ids.Length; i++)
            {
                if (ids[i].Id == UnknownId)
                {
                    if (ByteFallback)
                    {
                        if (tokenCount + ids[i].Utf8Length > maxTokenCount)
                        {
                            break;
                        }

                        tokenCount += ids[i].Utf8Length;
                    }
                    else
                    {
                        tokenCount++;
                    }

                    charsConsumed += Helpers.GetUtf16LengthFromUtf8Bytes(normalizationSpan.Slice(ids[i].UtfStartOffset, ids[i].Utf8Length));
                }
                else
                {
                    charsConsumed += _vocabReverse[ids[i].Id].Piece.Length;
                    tokenCount++;
                }
            }

            //
            // Create the normalized string.
            //

            ArrayPool<(int Id, int UtfStartOffset, int Utf8Length)>.Shared.Return(ids);

            if (normalizedString is not null)
            {
                StoreNormalizedText(normalizationSpan, ref normalizedString, ref normalizedStringIndex);
            }

            if (normalizedArrayPool is not null)
            {
                ArrayPool<byte>.Shared.Return(normalizedArrayPool);
            }
        }

        public override int GetIndexByTokenCountFromEnd(
                        string? text,
                        ReadOnlySpan<char> textSpan,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        int maxTokenCount,
                        bool considerNormalization,
                        out string? normalizedText,
                        out int tokenCount)
        {
            ReadOnlySpan<char> textToEncode = string.IsNullOrEmpty(text) ? textSpan : text.AsSpan();

            tokenCount = 0;
            if (textToEncode.IsEmpty || maxTokenCount <= 0)
            {
                normalizedText = null;
                return textToEncode.Length;
            }

            if (addEndOfSentence)
            {
                tokenCount++;

                if (maxTokenCount == 1)
                {
                    normalizedText = null;
                    return textToEncode.Length;
                }
            }

            // Rent a buffer that approximately enough to hold the Utf8 encoded bytes, the normalization of the encoded buffer, and some extra memory to for encoding results.
            int[] buffer = ArrayPool<int>.Shared.Rent(textToEncode.Length * 3);

            // when maxTokenCount == int.MaxValue we don't need to return the normalized string as most likely we can handle the whole input text without need to continuation.
            char[]? normalizedString = maxTokenCount == int.MaxValue ? null : ArrayPool<char>.Shared.Rent(textToEncode.Length + 2);

            int charConsumedFromEnd;

            if (SpecialTokensRegex is not null)
            {
                GetIndexByTokenCountFromEndWithSpecialTokens(textToEncode, considerNormalization, ref tokenCount, buffer, ref normalizedString, out charConsumedFromEnd, out normalizedText, maxTokenCount);
            }
            else
            {
                GetIndexByTokenCountFromEndWithoutSpecialTokens(textToEncode, considerNormalization, ref tokenCount, buffer, ref normalizedString, out charConsumedFromEnd, out normalizedText, maxTokenCount);
            }

            if (addBeginningOfSentence && tokenCount < maxTokenCount)
            {
                tokenCount++;
            }

            ArrayPool<int>.Shared.Return(buffer);

            return normalizedText is not null ? normalizedText.Length - charConsumedFromEnd : 0;
        }

        private void GetIndexByTokenCountFromEndWithSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        ref int tokenCount,
                        int[] buffer,
                        ref char[]? normalizedString,
                        out int charConsumedFromEnd,
                        out string? normalizedText,
                        int maxTokenCount)
        {
            Debug.Assert(SpecialTokensRegex is not null);
            Debug.Assert(maxTokenCount > 0);

            charConsumedFromEnd = 0;
            int normalizedStringCountFromEnd = 0;

            (int Offset, int Length)[] splits = PreTokenizer.SplitText(text, SpecialTokensRegex!).ToArray();

            if (splits.Length == 0)
            {
                GetIndexByTokenCountFromEndInternal(text, considerNormalization, ref tokenCount, buffer, ref normalizedString, ref normalizedStringCountFromEnd, ref charConsumedFromEnd, maxTokenCount);
                normalizedText = normalizedString is not null ? normalizedString.AsSpan(normalizedString.Length - normalizedStringCountFromEnd).ToString() : null;
                return;
            }

            (int Offset, int Length) current = splits[splits.Length - 1];

            // Last part is not a special token
            if (current.Offset + current.Length < text.Length)
            {
                GetIndexByTokenCountFromEndInternal(text.Slice(current.Offset + current.Length), considerNormalization, ref tokenCount, buffer, ref normalizedString, ref normalizedStringCountFromEnd, ref charConsumedFromEnd, maxTokenCount);
            }

            for (int i = splits.Length - 1; i >= 0; i--)
            {
                current = splits[i]; // special token

                if (tokenCount < maxTokenCount)
                {
                    if (InternalSpecialTokens!.TryGetValue(text.Slice(current.Offset, current.Length), out int id))
                    {
                        tokenCount++;
                    }

                    charConsumedFromEnd += current.Length;
                }

                if (normalizedString is not null)
                {
                    StoreNormalizedTextFromEnd(text.Slice(current.Offset, current.Length), ref normalizedString, ref normalizedStringCountFromEnd);
                }

                if (current.Offset > 0)
                {
                    int start = i > 0 ? splits[i - 1].Offset + splits[i - 1].Length : 0;
                    GetIndexByTokenCountFromEndInternal(text.Slice(start, current.Offset - start), considerNormalization, ref tokenCount, buffer, ref normalizedString, ref normalizedStringCountFromEnd, ref charConsumedFromEnd, maxTokenCount);
                }
            }

            normalizedText = normalizedString is not null ? normalizedString.AsSpan().Slice(normalizedString.Length - normalizedStringCountFromEnd).ToString() : null;
        }

        private void GetIndexByTokenCountFromEndWithoutSpecialTokens(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        ref int tokenCount,
                        int[] buffer,
                        ref char[]? normalizedString,
                        out int charConsumedFromEnd,
                        out string? normalizedText,
                        int maxTokenCount)
        {
            charConsumedFromEnd = 0;
            int normalizedStringCountFromEnd = 0;

            GetIndexByTokenCountFromEndInternal(text, considerNormalization, ref tokenCount, buffer, ref normalizedString, ref normalizedStringCountFromEnd, ref charConsumedFromEnd, maxTokenCount);

            normalizedText = normalizedString is not null ? normalizedString.AsSpan().Slice(normalizedString.Length - normalizedStringCountFromEnd).ToString() : null;
        }

        private void GetIndexByTokenCountFromEndInternal(
                        ReadOnlySpan<char> text,
                        bool considerNormalization,
                        ref int tokenCount,
                        int[] buffer,
                        ref char[]? normalizedString,
                        ref int normalizedStringCountFromEnd,
                        ref int charConsumedFromEnd,
                        int maxTokenCount)
        {
            //
            // Normalize the input text.
            //

            NormalizeText(text, considerNormalization, buffer, out byte[]? normalizedArrayPool, out Span<byte> normalizationSpan);

            //
            // Do the actual encoding
            //

            BestPathNode[] bestPathEndsAt = ArrayPool<BestPathNode>.Shared.Rent(normalizationSpan.Length + 1);

            Encode(normalizationSpan, bestPathEndsAt);

            int consumedCharacters = 0;
            int endsAt = normalizationSpan.Length;

            while (endsAt > 0 && tokenCount < maxTokenCount)
            {
                ref BestPathNode node = ref bestPathEndsAt[endsAt];

                if (node.Id == UnknownId)
                {
                    int length = endsAt - node.StartsAt;
                    if (ByteFallback)
                    {
                        if (tokenCount + length > maxTokenCount)
                        {
                            break;
                        }

                        tokenCount += length;
                    }
                    else
                    {
                        tokenCount++;
                    }

                    consumedCharacters += Helpers.GetUtf16LengthFromUtf8Bytes(normalizationSpan.Slice(node.StartsAt, length));
                }
                else
                {
                    consumedCharacters += _vocabReverse[node.Id].Piece.Length;
                    tokenCount++;
                }

                endsAt = node.StartsAt;
            }

            charConsumedFromEnd += consumedCharacters;

            if (normalizedString is not null)
            {
                if (considerNormalization)
                {
                    StoreNormalizedTextFromEnd(normalizationSpan, ref normalizedString, ref normalizedStringCountFromEnd);
                }
                else
                {
                    StoreNormalizedTextFromEnd(text, ref normalizedString, ref normalizedStringCountFromEnd);
                }
            }

            ArrayPool<BestPathNode>.Shared.Return(bestPathEndsAt);
            if (normalizedArrayPool is not null)
            {
                ArrayPool<byte>.Shared.Return(normalizedArrayPool);
            }
        }
    }
}
