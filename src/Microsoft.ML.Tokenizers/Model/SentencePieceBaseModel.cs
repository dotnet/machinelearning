// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    internal abstract class SentencePieceBaseModel
    {
        internal SentencePieceBaseModel(ModelProto modelProto, bool addBos = false, bool addEos = false, IReadOnlyDictionary<string, int>? specialTokens = null)
        {
            if (modelProto is null)
            {
                throw new ArgumentNullException(nameof(modelProto));
            }

            AddBeginningOfSentence = addBos;
            AddEndOfSentence = addEos;
            BeginningOfSentenceToken = modelProto.TrainerSpec.BosPiece ?? "<s>";
            BeginningOfSentenceId = Math.Max(0, modelProto.TrainerSpec.BosId);
            EndOfSentenceToken = modelProto.TrainerSpec.EosPiece ?? "</s>";
            EndOfSentenceId = Math.Max(0, modelProto.TrainerSpec.EosId);
            UnknownToken = modelProto.TrainerSpec.UnkPiece ?? "<unk>";
            UnknownId = Math.Max(0, modelProto.TrainerSpec.UnkId);
            AddDummyPrefix = modelProto.NormalizerSpec.AddDummyPrefix;
            EscapeWhiteSpaces = modelProto.NormalizerSpec.EscapeWhitespaces;
            TreatWhitespaceAsSuffix = modelProto.TrainerSpec.TreatWhitespaceAsSuffix;
            ByteFallback = modelProto.TrainerSpec.ByteFallback;
            SpecialTokens = specialTokens;

            if (specialTokens is not null && specialTokens.Count > 0)
            {
                InternalSpecialTokens = new Dictionary<StringSpanOrdinalKey, int>();
                SpecialTokensReverse = new Dictionary<int, string>();

                foreach (var item in specialTokens)
                {
                    InternalSpecialTokens.Add(new StringSpanOrdinalKey(item.Key), item.Value);
                    SpecialTokensReverse.Add(item.Value, item.Key);
                }

                // We create this Regex object without a timeout, as we expect the match operation to complete in O(N) time complexity. Note that `specialTokens` are treated as constants after the tokenizer is created.
                SpecialTokensRegex = new Regex(string.Join("|", specialTokens.Keys.Select(s => Regex.Escape(s))), RegexOptions.Compiled);
            }

            Normalizer = new SentencePieceNormalizer(
                                modelProto.NormalizerSpec.PrecompiledCharsmap.Span,
                                modelProto.NormalizerSpec.RemoveExtraWhitespaces,
                                AddDummyPrefix, EscapeWhiteSpaces,
                                modelProto.TrainerSpec.TreatWhitespaceAsSuffix,
                                specialTokens);
        }

        internal SentencePieceBaseModel(SentencePieceOptions options)
        {
            if (options is null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            if (options.Vocabulary is null)
            {
                throw new ArgumentNullException(nameof(options.Vocabulary));
            }

            if (options.BeginningOfSentenceToken is null)
            {
                throw new ArgumentNullException(nameof(options.BeginningOfSentenceToken));
            }

            if (options.EndOfSentenceToken is null)
            {
                throw new ArgumentNullException(nameof(options.EndOfSentenceToken));
            }

            if (options.UnknownToken is null)
            {
                throw new ArgumentNullException(nameof(options.UnknownToken));
            }

            AddBeginningOfSentence = options.AddBeginningOfSentence;
            AddEndOfSentence = options.AddEndOfSentence;
            BeginningOfSentenceToken = options.BeginningOfSentenceToken;
            EndOfSentenceToken = options.EndOfSentenceToken;
            UnknownToken = options.UnknownToken;
            AddDummyPrefix = options.AddDummyPrefix;
            EscapeWhiteSpaces = options.EscapeWhiteSpaces;
            TreatWhitespaceAsSuffix = options.TreatWhitespaceAsSuffix;
            ByteFallback = options.ByteFallback;
            SpecialTokens = options.SpecialTokens;

            if (SpecialTokens is not null && SpecialTokens.Count > 0)
            {
                InternalSpecialTokens = new Dictionary<StringSpanOrdinalKey, int>();
                SpecialTokensReverse = new Dictionary<int, string>();

                foreach (var item in SpecialTokens)
                {
                    InternalSpecialTokens.Add(new StringSpanOrdinalKey(item.Key), item.Value);
                    SpecialTokensReverse.Add(item.Value, item.Key);
                }

                // We create this Regex object without a timeout, as we expect the match operation to complete in O(N) time complexity. Note that `specialTokens` are treated as constants after the tokenizer is created.
                SpecialTokensRegex = new Regex(string.Join("|", SpecialTokens.Keys.Select(s => Regex.Escape(s))), RegexOptions.Compiled);
            }

            Normalizer = new SentencePieceNormalizer(
                                options.PrecompiledNormalizationData,
                                options.RemoveExtraWhiteSpaces,
                                options.AddDummyPrefix, options.EscapeWhiteSpaces,
                                options.TreatWhitespaceAsSuffix,
                                SpecialTokens);
        }

        internal Regex? SpecialTokensRegex { get; }

        internal Dictionary<StringSpanOrdinalKey, int>? InternalSpecialTokens { get; }

        internal Dictionary<int, string>? SpecialTokensReverse { get; }

        internal int MaxByteId { get; set; } // the maximum value of the byte id.;

        internal int ByteCodeToIdOffset { get; set; } // offset of mapping byte code to the to the Ids.

        internal int OneByteUtf8EncodingMaxId { get; set; } // the maximum value of the one byte UTF-8 character.

        public IReadOnlyDictionary<string, int>? SpecialTokens { get; }

        public bool ByteFallback { get; }

        public bool AddDummyPrefix { get; }

        public bool EscapeWhiteSpaces { get; }

        public bool TreatWhitespaceAsSuffix { get; internal set; }

        public bool AddBeginningOfSentence { get; }

        public bool AddEndOfSentence { get; }

        public string BeginningOfSentenceToken { get; }

        public string EndOfSentenceToken { get; }

        public string UnknownToken { get; }

        public int BeginningOfSentenceId { get; set; }

        public int EndOfSentenceId { get; set; }

        public int UnknownId { get; set; }

        public SentencePieceNormalizer? Normalizer { get; }

        public abstract IReadOnlyDictionary<string, int> Vocabulary { get; }

        public abstract IReadOnlyList<EncodedToken> EncodeToTokens(
                                                        string? text,
                                                        ReadOnlySpan<char> textSpan,
                                                        out string? normalizedText,
                                                        bool addBeginningOfSentence,
                                                        bool addEndOfSentence,
                                                        bool considerNormalization);

        public abstract IReadOnlyList<int> EncodeToIds(
                                            string? text,
                                            ReadOnlySpan<char> textSpan,
                                            bool addBeginningOfSentence,
                                            bool addEndOfSentence,
                                            bool considerNormalization,
                                            out string? normalizedText,
                                            out int charsConsumed,
                                            int maxTokenCount = int.MaxValue);

        public abstract int CountTokens(
                        string? text,
                        ReadOnlySpan<char> textSpan,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerNormalization,
                        out string? normalizedText,
                        out int charsConsumed,
                        int maxTokenCount = int.MaxValue);

        public abstract int GetIndexByTokenCountFromEnd(
                        string? text,
                        ReadOnlySpan<char> textSpan,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        int maxTokenCount,
                        bool considerNormalization,
                        out string? normalizedText,
                        out int tokenCount);

        public abstract bool TryMapIdToToken(int id, out string? token);

        private const int ApproximatedMaxEncodedBytesCount = 50;

        public virtual string Decode(IEnumerable<int> ids, bool considerSpecialTokens)
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

            int current = enumerator.Current;
            if (current <= MaxByteId && ByteFallback)
            {
                // First token is a byte token.

                while (current < ByteCodeToIdOffset)
                {
                    // It is possible listing some special tokens before the byte tokens in the tokenizer's data.
                    TryDecodeAsSpecialToken(this, current, considerSpecialTokens, ref sb);

                    // Skip control tokens.
                    if (!enumerator.MoveNext())
                    {
                        return sb.ToString();
                    }

                    current = enumerator.Current;
                }

                if (current <= MaxByteId && ByteFallback)
                {
                    EncodeByte(current, OneByteUtf8EncodingMaxId, ByteCodeToIdOffset, ref bytesCount, ref bytesPoolArray, ref sb);
                }
                else if (!TryDecodeAsSpecialToken(this, current, considerSpecialTokens, ref sb) && TryMapIdToToken(current, out string? token))
                {
                    AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token!, prefixSuffixChar, ref sb, ref prefixRemoved, ref suffixIndex);
                }
            }
            else if (!TryDecodeAsSpecialToken(this, current, considerSpecialTokens, ref sb) && TryMapIdToToken(current, out string? token))
            {
                AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token!, prefixSuffixChar, ref sb, ref prefixRemoved, ref suffixIndex);
            }

            char[]? charPoolArray = null;

            while (enumerator.MoveNext())
            {
                current = enumerator.Current;
                if (current < ByteCodeToIdOffset)
                {
                    if (bytesCount >= 1)
                    {
                        FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, ref sb);
                    }

                    // It is possible listing some special tokens before the byte tokens in the tokenizer's data.
                    TryDecodeAsSpecialToken(this, current, considerSpecialTokens, ref sb);

                    continue;
                }

                if (current <= MaxByteId && ByteFallback)
                {
                    if (bytesCount >= 1)
                    {
                        Debug.Assert(bytesPoolArray is not null);

                        if (bytesCount >= bytesPoolArray!.Length)
                        {
                            Helpers.ArrayPoolGrow(ref bytesPoolArray, bytesCount * 2);
                        }

                        bytesPoolArray![bytesCount++] = (byte)(current - ByteCodeToIdOffset);
                    }
                    else
                    {
                        EncodeByte(current, OneByteUtf8EncodingMaxId, ByteCodeToIdOffset, ref bytesCount, ref bytesPoolArray, ref sb);
                    }
                }
                else
                {
                    if (bytesCount >= 1)
                    {
                        FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, ref sb);
                    }

                    if (!TryDecodeAsSpecialToken(this, current, considerSpecialTokens, ref sb) && TryMapIdToToken(current, out string? token))
                    {
                        AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token!, prefixSuffixChar, ref sb, ref prefixRemoved, ref suffixIndex);
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

                charPoolArray ??= ArrayPool<char>.Shared.Rent(Math.Max(len, ApproximatedMaxEncodedBytesCount >> 1));

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
                    bytesPoolArray ??= ArrayPool<byte>.Shared.Rent(ApproximatedMaxEncodedBytesCount);
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

            static bool TryDecodeAsSpecialToken(SentencePieceBaseModel model, int id, bool considerSpecialTokens, ref ValueStringBuilder sb)
            {
                string? token = null;

                if (id == model.BeginningOfSentenceId)
                {
                    token = model.BeginningOfSentenceToken;
                }
                else if (id == model.EndOfSentenceId)
                {
                    token = model.EndOfSentenceToken;
                }
                else if (id == model.UnknownId)
                {
                    token = model.UnknownToken;
                }
                else if (model.SpecialTokensReverse?.TryGetValue(id, out string? specialToken) is true)
                {
                    token = specialToken;
                }

                if (token is not null && considerSpecialTokens)
                {
                    sb.Append(token);
                }

                return token is not null;
            }
        }

        public virtual OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, bool considerSpecialTokens, out int idsConsumed, out int charsWritten)
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

            int current = enumerator.Current;
            if (current <= MaxByteId && ByteFallback)
            {
                // First token is a byte token.
                while (current < ByteCodeToIdOffset)
                {
                    OperationStatus status = TryDecodeAsSpecialToken(this, current, considerSpecialTokens, buffer, ref charsWritten, out bool isSpecialToken);
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

                    current = enumerator.Current;
                }

                if (current <= MaxByteId && ByteFallback)
                {
                    if (!EncodeByte(enumerator.Current, OneByteUtf8EncodingMaxId, ByteCodeToIdOffset, ref bytesCount, buffer, ref charsWritten, ref idsConsumed, ref bytesPoolArray))
                    {
                        return OperationStatus.DestinationTooSmall;
                    }
                }
                else
                {
                    OperationStatus status = TryDecodeAsSpecialToken(this, current, considerSpecialTokens, buffer, ref charsWritten, out bool isSpecialToken);
                    if (status != OperationStatus.Done)
                    {
                        return status;
                    }

                    if (!isSpecialToken && TryMapIdToToken(current, out string? token))
                    {
                        if (!AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token!, prefixSuffixChar, destination, ref prefixRemoved, ref suffixIndex, ref idsConsumed, ref charsWritten))
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                    }
                    else
                    {
                        idsConsumed++;
                    }
                }
            }
            else
            {
                OperationStatus status = TryDecodeAsSpecialToken(this, current, considerSpecialTokens, buffer, ref charsWritten, out bool isSpecialToken);
                if (status != OperationStatus.Done)
                {
                    return status;
                }

                if (!isSpecialToken && TryMapIdToToken(current, out string? token))
                {
                    if (!AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token!, prefixSuffixChar, destination, ref prefixRemoved, ref suffixIndex, ref idsConsumed, ref charsWritten))
                    {
                        return OperationStatus.DestinationTooSmall;
                    }
                }
                else
                {
                    idsConsumed++;
                }
            }

            char[]? charPoolArray = null;

            while (enumerator.MoveNext())
            {
                current = enumerator.Current;
                buffer = destination.Slice(charsWritten);

                if (current < ByteCodeToIdOffset)
                {
                    if (bytesCount >= 1)
                    {
                        if (!FlushBytes(ref bytesCount, ref bytesPoolArray, ref charPoolArray, buffer, ref charsWritten, ref idsConsumed))
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                    }

                    OperationStatus status = TryDecodeAsSpecialToken(this, current, considerSpecialTokens, buffer, ref charsWritten, out bool isSpecialToken);
                    if (status != OperationStatus.Done)
                    {
                        return status;
                    }

                    idsConsumed++;
                    continue;
                }

                if (current <= MaxByteId && ByteFallback)
                {
                    if (bytesCount >= 1)
                    {
                        Debug.Assert(bytesPoolArray is not null);

                        if (bytesCount >= bytesPoolArray!.Length)
                        {
                            Helpers.ArrayPoolGrow(ref bytesPoolArray, bytesCount * 2);
                        }

                        bytesPoolArray![bytesCount++] = (byte)(current - ByteCodeToIdOffset);
                    }
                    else
                    {
                        if (!EncodeByte(current, OneByteUtf8EncodingMaxId, ByteCodeToIdOffset, ref bytesCount, buffer, ref charsWritten, ref idsConsumed, ref bytesPoolArray))
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

                    OperationStatus status = TryDecodeAsSpecialToken(this, current, considerSpecialTokens, buffer, ref charsWritten, out bool isSpecialToken);
                    if (status != OperationStatus.Done)
                    {
                        return status;
                    }

                    if (!isSpecialToken && TryMapIdToToken(current, out string? token))
                    {
                        if (!AppendTokenWithCheckingPrefix(AddDummyPrefix, TreatWhitespaceAsSuffix, token!, prefixSuffixChar, destination, ref prefixRemoved, ref suffixIndex, ref idsConsumed, ref charsWritten))
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                    }
                    else
                    {
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

            static OperationStatus TryDecodeAsSpecialToken(SentencePieceBaseModel model, int id, bool considerSpecialTokens, Span<char> buffer, ref int charsWritten, out bool isSpecialToken)
            {
                string? specialToken = null;

                if (id == model.BeginningOfSentenceId)
                {
                    specialToken = model.BeginningOfSentenceToken;
                }
                else if (id == model.EndOfSentenceId)
                {
                    specialToken = model.EndOfSentenceToken;
                }
                else if (id == model.UnknownId)
                {
                    specialToken = model.UnknownToken;
                }
                else
                {
                    model.SpecialTokensReverse?.TryGetValue(id, out specialToken);
                }

                isSpecialToken = specialToken is not null;

                if (considerSpecialTokens && isSpecialToken)
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

                charPoolArray ??= ArrayPool<char>.Shared.Rent(Math.Max(len, ApproximatedMaxEncodedBytesCount >> 1));

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
                    bytesPoolArray ??= ArrayPool<byte>.Shared.Rent(ApproximatedMaxEncodedBytesCount);
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
                        Helpers.Replace(tokenSpan, buffer, prefixSuffixChar, ' ');
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
                        Helpers.Replace(tokenSpan, buffer, prefixSuffixChar, ' ');
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
                        Helpers.Replace(tokenSpan, buffer, prefixSuffixChar, ' ');
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
    }
}