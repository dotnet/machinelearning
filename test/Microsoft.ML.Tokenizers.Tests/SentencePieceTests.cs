// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class SentencePieceTests
    {
        [Fact]
        public void CreateWithNullStreamThrows()
        {
            Assert.ThrowsAny<ArgumentException>(() => SentencePieceTokenizer.Create(null!));
        }

        [Fact]
        public void CreateWithEmptyStreamThrows()
        {
            using MemoryStream empty = new MemoryStream(Array.Empty<byte>());
            Assert.ThrowsAny<ArgumentException>(() => SentencePieceTokenizer.Create(empty));
        }

        [Fact]
        public void CreateWithTruncatedStreamThrows()
        {
            // A protobuf tag claiming a length-delimited field longer than remaining bytes.
            byte[] truncated = new byte[] { 0x0A, 0xFF, 0x01 }; // field 1, length 255 – but only 0 data bytes follow
            using MemoryStream ms = new MemoryStream(truncated);
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void CreateBpeViaSentencePieceTokenizer()
        {
            // Verify that the generic SentencePieceTokenizer.Create() factory method
            // works for BPE models (not just LlamaTokenizer.Create()).
            using Stream stream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(stream);

            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens("Hello", out _);
            Assert.True(tokens.Count > 0);
            Assert.Equal("Hello", tokenizer.Decode(tokens.Select(t => t.Id)));
        }

        [Fact]
        public void CreateFromMemoryStreamUsesFastPath()
        {
            // Verify that loading from a MemoryStream works (exercises
            // the TryGetBuffer fast-path in ModelProtoParser.ParseFrom).
            byte[] modelBytes = File.ReadAllBytes(Path.Combine(@"Llama", "tokenizer.model"));
            using MemoryStream ms = new MemoryStream(modelBytes);
            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(ms);

            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens("Hello", out _);
            Assert.True(tokens.Count > 0);
            Assert.Equal("Hello", tokenizer.Decode(tokens.Select(t => t.Id)));
        }

        [Fact]
        public void BpeModelPropertiesParsedCorrectly()
        {
            // Verify that TrainerSpec and NormalizerSpec fields are correctly parsed
            // from a known BPE model (Llama).
            using Stream stream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(stream);

            Assert.True(tokenizer.ByteFallback);
            Assert.True(tokenizer.AddDummyPrefix);
            Assert.True(tokenizer.EscapeWhiteSpaces);
            Assert.False(tokenizer.TreatWhitespaceAsSuffix);

            Assert.Equal("<unk>", tokenizer.UnknownToken);
            Assert.Equal("<s>", tokenizer.BeginningOfSentenceToken);
            Assert.Equal("</s>", tokenizer.EndOfSentenceToken);
            Assert.Equal(0, tokenizer.UnknownId);
            Assert.Equal(1, tokenizer.BeginningOfSentenceId);
            Assert.Equal(2, tokenizer.EndOfSentenceId);
        }

        [Fact]
        public void UnigramModelPropertiesParsedCorrectly()
        {
            // Verify that TrainerSpec and NormalizerSpec fields are correctly parsed
            // from a known Unigram model (Paraphrase-multilingual-MiniLM-L12-v2).
            using Stream stream = File.OpenRead(Path.Combine(
                @"Paraphrase-multilingual-MiniLM-L12-v2", "sentencepiece.bpe.model"));
            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(stream);

            // Unigram model should not have byte fallback
            Assert.False(tokenizer.ByteFallback);
            Assert.True(tokenizer.AddDummyPrefix);
            Assert.True(tokenizer.EscapeWhiteSpaces);
            Assert.False(tokenizer.TreatWhitespaceAsSuffix);
        }

        [Fact]
        public void ByteFallbackEncodesRareCharacterAsBytes()
        {
            // Llama has byte_fallback=true. Encoding a character that is not in
            // the vocabulary should produce byte-level tokens (<0xNN>) rather than <unk>.
            using Stream stream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(stream,
                addBeginningOfSentence: false, addEndOfSentence: false);

            // U+10342 (Old Italic Letter Re) — a 4-byte UTF-8 character (F0 90 8D 82)
            // that is extremely unlikely to be in the Llama vocabulary.
            string rareChar = "\U00010342";
            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(rareChar, out _);

            // With byte fallback, the character should be encoded as individual byte tokens
            // rather than a single <unk> token.
            Assert.True(tokens.Count > 1, "Byte fallback should produce multiple byte tokens.");
            Assert.DoesNotContain(tokens, t => t.Value == "<unk>");

            // Each byte token should have a name like <0xNN>.
            foreach (EncodedToken token in tokens)
            {
                // The first token is the dummy prefix "▁", the rest should be byte tokens.
                if (token.Value != "\u2581")
                {
                    Assert.StartsWith("<0x", token.Value);
                }
            }

            // Round-trip: decoding should recover the original character.
            string decoded = tokenizer.Decode(tokens.Select(t => t.Id));
            Assert.Equal(rareChar, decoded);
        }

        [Fact]
        public void BpeAndUnigramProduceDifferentTokenizations()
        {
            // Sanity check: BPE and Unigram models produce different tokenizations
            // for the same input, confirming model_type is parsed and routed correctly.
            using Stream bpeStream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            SentencePieceTokenizer bpe = SentencePieceTokenizer.Create(bpeStream,
                addBeginningOfSentence: false, addEndOfSentence: false);

            using Stream unigramStream = File.OpenRead(Path.Combine(
                @"Paraphrase-multilingual-MiniLM-L12-v2", "sentencepiece.bpe.model"));
            SentencePieceTokenizer unigram = SentencePieceTokenizer.Create(unigramStream,
                addBeginningOfSentence: false, addEndOfSentence: false);

            string input = "The quick brown fox jumps over the lazy dog.";
            IReadOnlyList<EncodedToken> bpeTokens = bpe.EncodeToTokens(input, out _);
            IReadOnlyList<EncodedToken> unigramTokens = unigram.EncodeToTokens(input, out _);

            // Both should successfully tokenize the input.
            Assert.True(bpeTokens.Count > 0);
            Assert.True(unigramTokens.Count > 0);

            // But they should produce different token sequences (different vocabs and algorithms).
            Assert.NotEqual(
                string.Join(",", bpeTokens.Select(t => t.Value)),
                string.Join(",", unigramTokens.Select(t => t.Value)));

            // Both should round-trip decode correctly.
            Assert.Equal(input, bpe.Decode(bpeTokens.Select(t => t.Id)));
            Assert.Equal(input, unigram.Decode(unigramTokens.Select(t => t.Id)));
        }

        [Fact]
        public void MalformedVarintInModelThrows()
        {
            // A varint that never terminates (all continuation bits set, exceeding 10 bytes).
            byte[] malformed = new byte[] { 0x0A, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 };
            using MemoryStream ms = new MemoryStream(malformed);
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void UnknownWireTypeInModelThrows()
        {
            // Wire type 7 is invalid in the protobuf spec.
            // Tag byte 0x0F = field 1, wire type 7.
            byte[] badWireType = new byte[] { 0x0F, 0x00 };
            using MemoryStream ms = new MemoryStream(badWireType);
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        // =================================================================
        // Synthetic protobuf tests — construct models from raw wire format
        // =================================================================

        [Fact]
        public void Synthetic_BpeModel_DefaultProperties()
        {
            // Minimal BPE model: only ModelType set, everything else uses proto defaults.
            byte[] model = MakeModelProto(MakeBpeTrainerSpec());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);

            // TrainerSpec defaults
            Assert.False(tokenizer.ByteFallback);
            Assert.False(tokenizer.TreatWhitespaceAsSuffix);
            Assert.Equal("<unk>", tokenizer.UnknownToken);
            Assert.Equal("<s>", tokenizer.BeginningOfSentenceToken);
            Assert.Equal("</s>", tokenizer.EndOfSentenceToken);
            Assert.Equal(0, tokenizer.UnknownId);
            Assert.Equal(1, tokenizer.BeginningOfSentenceId);
            Assert.Equal(2, tokenizer.EndOfSentenceId);

            // NormalizerSpec defaults (proto specifies [default = true] for these)
            Assert.True(tokenizer.AddDummyPrefix);
            Assert.True(tokenizer.EscapeWhiteSpaces);
        }

        [Fact]
        public void Synthetic_BpeModel_AllTrainerSpecFieldsParsed()
        {
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);             // model_type = BPE
            ts.WriteBoolField(24, true);           // treat_whitespace_as_suffix
            ts.WriteBoolField(35, true);           // byte_fallback
            ts.WriteInt32Field(40, 5);             // unk_id
            ts.WriteInt32Field(41, 6);             // bos_id
            ts.WriteInt32Field(42, 7);             // eos_id
            ts.WriteInt32Field(43, 8);             // pad_id
            ts.WriteStringField(45, "[UNK]");      // unk_piece
            ts.WriteStringField(46, "[BOS]");      // bos_piece
            ts.WriteStringField(47, "[EOS]");      // eos_piece
            ts.WriteStringField(48, "[PAD]");      // pad_piece

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);

            Assert.True(tokenizer.ByteFallback);
            Assert.True(tokenizer.TreatWhitespaceAsSuffix);
            Assert.Equal("[UNK]", tokenizer.UnknownToken);
            Assert.Equal("[BOS]", tokenizer.BeginningOfSentenceToken);
            Assert.Equal("[EOS]", tokenizer.EndOfSentenceToken);
            Assert.Equal(5, tokenizer.UnknownId);
            Assert.Equal(6, tokenizer.BeginningOfSentenceId);
            Assert.Equal(7, tokenizer.EndOfSentenceId);
        }

        [Fact]
        public void Synthetic_BpeModel_AllNormalizerSpecFieldsParsed()
        {
            ProtobufWriter ns = new();
            ns.WriteStringField(1, "identity");     // name
            ns.WriteBoolField(3, false);             // add_dummy_prefix (default true)
            ns.WriteBoolField(4, false);             // remove_extra_whitespaces (default true)
            ns.WriteBoolField(5, false);             // escape_whitespaces (default true)

            byte[] model = MakeModelProto(MakeBpeTrainerSpec(), ns.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);

            Assert.False(tokenizer.AddDummyPrefix);
            Assert.False(tokenizer.EscapeWhiteSpaces);
        }

        [Fact]
        public void Synthetic_BpeModel_NegativePadIdParsesSuccessfully()
        {
            // PadId = -1 is encoded as a 10-byte varint (sign-extended to uint64).
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);     // model_type = BPE
            ts.WriteInt32Field(43, -1);   // pad_id = -1

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_BpeModel_AllPieceTypesParseSuccessfully()
        {
            byte[] model = MakeModelProto(
                MakeBpeTrainerSpec(),
                null,
                MakePiece("<unk>", 0f, 2),     // Unknown
                MakePiece("<s>", 0f, 3),        // Control
                MakePiece("</s>", 0f, 3),       // Control
                MakePiece("hello", -1.5f, 1),   // Normal
                MakePiece("world", -2.0f, 4),   // UserDefined
                MakePiece("<pad>", 0f, 5),       // Unused
                MakePiece("<0x00>", 0f, 6));     // Byte

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_BpeModel_FieldsInReverseOrderParseCorrectly()
        {
            // Protobuf allows fields in any order; verify our parser handles it.
            ProtobufWriter ts = new();
            ts.WriteStringField(47, "[EOS]");      // eos_piece (field 47 before field 3)
            ts.WriteStringField(46, "[BOS]");      // bos_piece
            ts.WriteStringField(45, "[UNK]");      // unk_piece
            ts.WriteBoolField(35, true);            // byte_fallback
            ts.WriteInt32Field(3, 2);               // model_type = BPE (written last)

            // ModelProto fields also reversed: normalizer before trainer, trainer before pieces
            ProtobufWriter model = new();
            model.WriteMessageField(3, Array.Empty<byte>()); // empty NormalizerSpec
            model.WriteMessageField(2, ts.ToArray());         // TrainerSpec
            model.WriteMessageField(1, MakePiece("test", -1f)); // piece

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model.ToArray());
            Assert.True(tokenizer.ByteFallback);
            Assert.Equal("[UNK]", tokenizer.UnknownToken);
            Assert.Equal("[BOS]", tokenizer.BeginningOfSentenceToken);
            Assert.Equal("[EOS]", tokenizer.EndOfSentenceToken);
        }

        [Fact]
        public void Synthetic_BpeModel_DuplicateScalarFieldLastValueWins()
        {
            // When a scalar field appears multiple times, protobuf spec says last value wins.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);            // model_type = BPE
            ts.WriteBoolField(35, false);         // byte_fallback = false
            ts.WriteBoolField(35, true);          // byte_fallback = true (should win)
            ts.WriteStringField(45, "first");     // unk_piece = "first"
            ts.WriteStringField(45, "second");    // unk_piece = "second" (should win)

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);

            Assert.True(tokenizer.ByteFallback);
            Assert.Equal("second", tokenizer.UnknownToken);
        }

        [Fact]
        public void Synthetic_BpeModel_UnknownFieldsAtAllLevelsAreSkipped()
        {
            // Unknown field numbers should be silently skipped at every message level.

            // SentencePiece with unknown field
            ProtobufWriter piece = new();
            piece.WriteStringField(1, "test");
            piece.WriteFloatField(2, -1f);
            piece.WriteInt32Field(3, 1);          // type = Normal
            piece.WriteInt32Field(99, 42);        // unknown field

            // TrainerSpec with unknown field
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);             // model_type = BPE
            ts.WriteBoolField(35, true);           // byte_fallback
            ts.WriteInt32Field(99, 42);           // unknown field

            // NormalizerSpec with unknown field
            ProtobufWriter ns = new();
            ns.WriteStringField(1, "identity");
            ns.WriteInt32Field(99, 42);           // unknown field

            // ModelProto with unknown field
            ProtobufWriter model = new();
            model.WriteMessageField(1, piece.ToArray());
            model.WriteMessageField(2, ts.ToArray());
            model.WriteMessageField(3, ns.ToArray());
            model.WriteInt32Field(99, 42);        // unknown field at top level

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model.ToArray());
            Assert.True(tokenizer.ByteFallback);
        }

        [Fact]
        public void Synthetic_BpeModel_AllWireTypesSkippedForUnknownFields()
        {
            // Unknown fields using all four valid wire types should be skipped.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);                                    // model_type = BPE
            ts.WriteInt32Field(90, 42);                                  // unknown varint (wire type 0)
            ts.WriteFixed64Field(91, 0xDEADBEEF);                        // unknown fixed64 (wire type 1)
            ts.WriteBytesField(92, new byte[] { 0x01, 0x02, 0x03 });    // unknown length-delimited (wire type 2)
            ts.WriteFixed32Field(93, 0xCAFE);                            // unknown fixed32 (wire type 5)
            ts.WriteBoolField(35, true);                                  // byte_fallback (after unknown fields)

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);

            // Verify that fields after unknown fields are still parsed correctly.
            Assert.True(tokenizer.ByteFallback);
        }

        [Fact]
        public void Synthetic_BpeModel_EmptyNormalizerSpecUsesDefaults()
        {
            // Zero-length NormalizerSpec submessage → all C# defaults.
            ProtobufWriter model = new();
            model.WriteMessageField(2, MakeBpeTrainerSpec());
            model.WriteMessageField(3, Array.Empty<byte>());

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model.ToArray());
            Assert.True(tokenizer.AddDummyPrefix);
            Assert.True(tokenizer.EscapeWhiteSpaces);
        }

        [Fact]
        public void Synthetic_BpeModel_MultiplePiecesAccumulate()
        {
            // Repeated field: each piece message is independently appended.
            byte[] model = MakeModelProto(
                MakeBpeTrainerSpec(),
                null,
                MakePiece("a", -1f),
                MakePiece("b", -2f),
                MakePiece("c", -3f),
                MakePiece("ab", -0.5f),
                MakePiece("bc", -0.5f));

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_BpeModel_UnicodeStringsParsedCorrectly()
        {
            // Multi-byte UTF-8 strings in piece names.
            byte[] model = MakeModelProto(
                MakeBpeTrainerSpec(),
                null,
                MakePiece("\u2581", -1f),               // ▁ (3-byte UTF-8)
                MakePiece("\u65E5\u672C\u8A9E", -2f),   // 日本語 (CJK)
                MakePiece("\U0001F389", -3f),            // 🎉 (4-byte UTF-8)
                MakePiece("caf\u00E9", -4f));            // café (Latin with diacritics)

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_BpeModel_LargeFieldNumbersSkipped()
        {
            // Field number 1000 requires a multi-byte varint tag.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);        // model_type = BPE
            ts.WriteInt32Field(1000, 42);    // unknown field with large number
            ts.WriteBoolField(35, true);      // byte_fallback (after large field number)

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.True(tokenizer.ByteFallback);
        }

        [Fact]
        public void Synthetic_BpeModel_ZeroLengthStringField()
        {
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);        // model_type = BPE
            ts.WriteStringField(45, "");      // unk_piece = "" (zero-length)

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);

            // Empty string is not null, so ?? "<unk>" does not apply.
            Assert.Equal("", tokenizer.UnknownToken);
        }

        [Fact]
        public void Synthetic_UnigramModel_PropertiesParsedCorrectly()
        {
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 1);             // model_type = Unigram
            ts.WriteBoolField(35, true);           // byte_fallback
            ts.WriteInt32Field(40, 0);             // unk_id = 0
            ts.WriteInt32Field(41, 1);             // bos_id = 1
            ts.WriteInt32Field(42, 2);             // eos_id = 2

            byte[] model = MakeModelProto(
                ts.ToArray(),
                null,
                MakePiece("<unk>", 0f, 2),          // Unknown (index 0)
                MakePiece("<s>", 0f, 3),             // Control (index 1)
                MakePiece("</s>", 0f, 3),            // Control (index 2)
                MakePiece("\u2581hello", -1.5f, 1),  // Normal
                MakePiece("\u2581world", -2.0f, 1)); // Normal

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.True(tokenizer.ByteFallback);
            Assert.Equal(0, tokenizer.UnknownId);
            Assert.Equal(1, tokenizer.BeginningOfSentenceId);
            Assert.Equal(2, tokenizer.EndOfSentenceId);
        }

        [Fact]
        public void Synthetic_BpeModel_ParsedFromNonSeekableStream()
        {
            // Exercises the fallback path in ParseFrom (not MemoryStream.TryGetBuffer).
            byte[] modelBytes = MakeModelProto(
                MakeBpeTrainerSpec(),
                null,
                MakePiece("test", -1f));

            using NonSeekableStream stream = new(modelBytes);
            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(stream);
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_TruncatedVarintThrows()
        {
            // Tag byte with continuation bit set but no more data follows.
            byte[] data = new byte[] { 0x80 };
            using MemoryStream ms = new MemoryStream(data);
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void Synthetic_TruncatedStringFieldThrows()
        {
            // Length-delimited field claiming 100 bytes but only 2 follow.
            ProtobufWriter w = new();
            w.WriteTag(1, 2);       // field 1, length-delimited
            w.WriteVarint(100);     // length = 100
            w.WriteRaw(0x41, 0x42); // only 2 bytes

            using MemoryStream ms = new MemoryStream(w.ToArray());
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void Synthetic_TruncatedFloatInPieceThrows()
        {
            // Float field with only 2 of 4 bytes inside a piece submessage.
            ProtobufWriter piece = new();
            piece.WriteStringField(1, "test");
            piece.WriteTag(2, 5);             // score field tag (float = wire type 5)
            piece.WriteRaw(0x00, 0x00);       // only 2 of 4 bytes

            ProtobufWriter model = new();
            model.WriteMessageField(1, piece.ToArray());
            model.WriteMessageField(2, MakeBpeTrainerSpec());

            using MemoryStream ms = new MemoryStream(model.ToArray());
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void Synthetic_TruncatedFixed64DuringSkipThrows()
        {
            // Unknown field with wire type 1 (fixed64) but only 4 of 8 bytes available.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);          // model_type = BPE
            ts.WriteTag(90, 1);                // unknown field 90, wire type 1 (fixed64)
            ts.WriteRaw(0, 0, 0, 0);           // only 4 of 8 bytes

            ProtobufWriter model = new();
            model.WriteMessageField(2, ts.ToArray());

            using MemoryStream ms = new MemoryStream(model.ToArray());
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void Synthetic_InvalidLengthPrefixThrows()
        {
            // Length-delimited field with length far exceeding available data.
            ProtobufWriter w = new();
            w.WriteTag(1, 2);                               // field 1, length-delimited
            w.WriteRaw(0xFF, 0xFF, 0xFF, 0xFF, 0x07);       // varint = 0x7FFFFFFF

            using MemoryStream ms = new MemoryStream(w.ToArray());
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Theory]
        [InlineData(3)] // Group start (deprecated)
        [InlineData(4)] // Group end (deprecated)
        [InlineData(6)] // Invalid
        public void Synthetic_UnsupportedWireTypeThrows(int wireType)
        {
            byte[] data = new byte[] { (byte)((1 << 3) | wireType), 0x00 };
            using MemoryStream ms = new MemoryStream(data);
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void Synthetic_BpeModel_WireTypeMismatchSkippedGracefully()
        {
            // If a known field number arrives with an unexpected wire type,
            // the parser should treat it as unknown and skip it (forward-compat).
            ProtobufWriter model = new();
            model.WriteInt32Field(1, 42);               // field 1 (pieces) as varint instead of message — skipped
            model.WriteFixed32Field(2, 0);              // field 2 (TrainerSpec) as fixed32 instead of message — skipped
            model.WriteMessageField(2, MakeBpeTrainerSpec()); // real TrainerSpec with correct wire type

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model.ToArray());
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_BpeModel_NonCanonicalVarintParsedCorrectly()
        {
            // Value 2 (BPE) encoded as a maximally-padded 5-byte varint
            // instead of the minimal single byte 0x02.
            ProtobufWriter ts = new();
            ts.WriteTag(3, 0);                            // model_type field tag
            ts.WriteRaw(0x82, 0x80, 0x80, 0x80, 0x00);   // value 2 in 5-byte non-canonical encoding
            ts.WriteBoolField(35, true);                   // byte_fallback (verify parsing continues)

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.True(tokenizer.ByteFallback);
        }

        [Fact]
        public void Synthetic_BpeModel_NegativeBosEosIdClampedToZero()
        {
            // BosId and EosId set to -1 (disabled in SentencePiece).
            // The base constructor applies Math.Max(0, value), clamping to 0.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);      // model_type = BPE
            ts.WriteInt32Field(41, -1);    // bos_id = -1
            ts.WriteInt32Field(42, -1);    // eos_id = -1

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.Equal(0, tokenizer.BeginningOfSentenceId);
            Assert.Equal(0, tokenizer.EndOfSentenceId);
        }

        [Fact]
        public void Synthetic_BpeModel_ExtensionRangeFieldsSkipped()
        {
            // sentencepiece_model.proto defines 'extensions 200 to max;' on several messages.
            // Real files may contain extension fields that must be silently skipped.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);                              // model_type = BPE
            ts.WriteStringField(200, "ext_value");                 // extension field
            ts.WriteInt32Field(300, 99);                           // extension field
            ts.WriteBytesField(500, new byte[] { 0xFF });          // extension field
            ts.WriteBoolField(35, true);                            // byte_fallback

            ProtobufWriter ns = new();
            ns.WriteInt32Field(200, 77);                           // extension in NormalizerSpec
            ns.WriteBoolField(3, false);                            // add_dummy_prefix = false

            byte[] model = MakeModelProto(ts.ToArray(), ns.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.True(tokenizer.ByteFallback);
            Assert.False(tokenizer.AddDummyPrefix);
        }

        [Fact]
        public void Synthetic_BpeModel_FloatSpecialValuesParsed()
        {
            // IEEE 754 special values should not crash the parser.
            byte[] model = MakeModelProto(
                MakeBpeTrainerSpec(),
                null,
                MakePiece("nan_score", float.NaN, 1),
                MakePiece("inf_score", float.PositiveInfinity, 1),
                MakePiece("neg_inf", float.NegativeInfinity, 1),
                MakePiece("subnormal", float.Epsilon, 1));

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_BpeModel_SkipUnknownFieldWithLongVarint()
        {
            // Unknown varint field with value -1 (10-byte varint in the skip path).
            // Exercises the SkipField varint loop through all 10 bytes.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, 2);       // model_type = BPE
            ts.WriteInt32Field(99, -1);     // unknown field, value -1 (10-byte varint)
            ts.WriteBoolField(35, true);     // byte_fallback (verify skip was correct)

            byte[] model = MakeModelProto(ts.ToArray());
            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.True(tokenizer.ByteFallback);
        }

        [Fact]
        public void Synthetic_BpeModel_ZeroLengthPieceSubmessage()
        {
            // A piece submessage with zero length creates a default SentencePiece
            // (piece="", score=0, type=Normal).
            ProtobufWriter model = new();
            model.WriteMessageField(1, Array.Empty<byte>()); // zero-length piece
            model.WriteMessageField(2, MakeBpeTrainerSpec());

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model.ToArray());
            Assert.NotNull(tokenizer);
        }

        [Fact]
        public void Synthetic_BpeModel_MultipleTrainerSpecLastWins()
        {
            // When a non-repeated message field appears multiple times,
            // our parser uses the last occurrence (not protobuf merge semantics).
            ProtobufWriter ts1 = new();
            ts1.WriteInt32Field(3, 2);            // model_type = BPE
            ts1.WriteBoolField(35, false);         // byte_fallback = false
            ts1.WriteStringField(45, "first");     // unk_piece = "first"

            ProtobufWriter ts2 = new();
            ts2.WriteInt32Field(3, 2);            // model_type = BPE
            ts2.WriteBoolField(35, true);          // byte_fallback = true
            ts2.WriteStringField(45, "second");    // unk_piece = "second"

            ProtobufWriter model = new();
            model.WriteMessageField(2, ts1.ToArray());
            model.WriteMessageField(2, ts2.ToArray());

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model.ToArray());
            Assert.True(tokenizer.ByteFallback);
            Assert.Equal("second", tokenizer.UnknownToken);
        }

        [Fact]
        public void Synthetic_BpeModel_MultipleTrainerSpecFieldsNotMerged()
        {
            // Unlike the standard protobuf library which merges repeated occurrences of
            // non-repeated message fields, our parser replaces entirely (last wins).
            // Fields from the first occurrence that aren't in the second are lost.
            ProtobufWriter ts1 = new();
            ts1.WriteInt32Field(3, 2);            // model_type = BPE
            ts1.WriteBoolField(35, true);          // byte_fallback = true (ONLY in first)

            ProtobufWriter ts2 = new();
            ts2.WriteInt32Field(3, 2);            // model_type = BPE
            ts2.WriteStringField(45, "[UNK]");     // unk_piece (ONLY in second; no byte_fallback)

            ProtobufWriter model = new();
            model.WriteMessageField(2, ts1.ToArray());
            model.WriteMessageField(2, ts2.ToArray());

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model.ToArray());
            // With protobuf merge: ByteFallback=true (from first), UnknownToken="[UNK]" (from second).
            // With our last-wins: ByteFallback=false (default, since second has no byte_fallback field).
            Assert.False(tokenizer.ByteFallback);
            Assert.Equal("[UNK]", tokenizer.UnknownToken);
        }

        [Fact]
        public void Synthetic_BpeModel_PieceWithPartialFieldsUsesDefaults()
        {
            // A piece with only the string field; score and type use defaults (0f, Normal).
            ProtobufWriter piece = new();
            piece.WriteStringField(1, "hello"); // only piece name, no score or type

            byte[] model = MakeModelProto(
                MakeBpeTrainerSpec(),
                null,
                piece.ToArray());

            SentencePieceTokenizer tokenizer = CreateFromSyntheticModel(model);
            Assert.NotNull(tokenizer);
        }

        [Theory]
        [InlineData(3)] // Word
        [InlineData(4)] // Char
        [InlineData(99)] // completely unknown
        public void Synthetic_UnsupportedModelTypeThrows(int modelType)
        {
            // Only BPE (2) and Unigram (1) are supported. Other model types
            // should throw ArgumentException from the SentencePieceTokenizer constructor.
            ProtobufWriter ts = new();
            ts.WriteInt32Field(3, modelType);

            byte[] model = MakeModelProto(ts.ToArray());
            using MemoryStream ms = new MemoryStream();
            ms.Write(model, 0, model.Length);
            ms.Position = 0;
            Assert.Throws<ArgumentException>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void Synthetic_BpeModel_MemoryStreamAtNonZeroPosition()
        {
            // When a MemoryStream has been partially consumed (Position > 0),
            // the parser's TryGetBuffer fast path must start at the current position,
            // not at the beginning of the buffer.
            byte[] modelBytes = MakeModelProto(
                MakeBpeTrainerSpec(),
                null,
                MakePiece("test", -1f));
            byte[] prefix = new byte[137]; // arbitrary prefix junk

            using MemoryStream ms = new MemoryStream();
            ms.Write(prefix, 0, prefix.Length);
            ms.Write(modelBytes, 0, modelBytes.Length);
            ms.Position = prefix.Length; // skip past prefix

            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(ms);
            Assert.NotNull(tokenizer);
        }

        // =================================================================
        // Helper infrastructure
        // =================================================================

        private static SentencePieceTokenizer CreateFromSyntheticModel(
            byte[] modelProtoBytes, bool addBos = true, bool addEos = false)
        {
            // Default MemoryStream constructor + Write so TryGetBuffer returns true,
            // exercising the fast path in ModelProtoParser.ParseFrom.
            using MemoryStream ms = new MemoryStream();
            ms.Write(modelProtoBytes, 0, modelProtoBytes.Length);
            ms.Position = 0;
            return SentencePieceTokenizer.Create(ms, addBos, addEos);
        }

        private static byte[] MakePiece(string text, float score = 0f, int type = 1)
        {
            ProtobufWriter w = new();
            w.WriteStringField(1, text);    // piece
            w.WriteFloatField(2, score);    // score
            w.WriteInt32Field(3, type);     // type enum
            return w.ToArray();
        }

        private static byte[] MakeBpeTrainerSpec()
        {
            ProtobufWriter w = new();
            w.WriteInt32Field(3, 2); // model_type = BPE
            return w.ToArray();
        }

        private static byte[] MakeModelProto(
            byte[] trainerSpec, byte[]? normalizerSpec = null, params byte[][] pieces)
        {
            ProtobufWriter w = new();
            foreach (byte[] piece in pieces)
            {
                w.WriteMessageField(1, piece); // repeated SentencePiece
            }
            w.WriteMessageField(2, trainerSpec); // TrainerSpec
            if (normalizerSpec != null)
            {
                w.WriteMessageField(3, normalizerSpec); // NormalizerSpec
            }
            return w.ToArray();
        }

        /// <summary>Minimal protobuf writer for constructing synthetic test data.</summary>
        private sealed class ProtobufWriter
        {
            private readonly MemoryStream _ms = new MemoryStream();

            public byte[] ToArray() => _ms.ToArray();

            public void WriteVarint(ulong value)
            {
                while (value > 0x7F)
                {
                    _ms.WriteByte((byte)(value | 0x80));
                    value >>= 7;
                }
                _ms.WriteByte((byte)value);
            }

            public void WriteTag(int fieldNumber, int wireType) =>
                WriteVarint((ulong)((fieldNumber << 3) | wireType));

            public void WriteInt32Field(int fieldNumber, int value)
            {
                WriteTag(fieldNumber, 0);
                WriteVarint((ulong)(long)value); // sign-extend for negative values
            }

            public void WriteBoolField(int fieldNumber, bool value)
            {
                WriteTag(fieldNumber, 0);
                WriteVarint(value ? 1UL : 0UL);
            }

            public void WriteStringField(int fieldNumber, string value)
            {
                byte[] bytes = Encoding.UTF8.GetBytes(value);
                WriteTag(fieldNumber, 2);
                WriteVarint((ulong)bytes.Length);
                _ms.Write(bytes, 0, bytes.Length);
            }

            public void WriteBytesField(int fieldNumber, byte[] value)
            {
                WriteTag(fieldNumber, 2);
                WriteVarint((ulong)value.Length);
                _ms.Write(value, 0, value.Length);
            }

            public void WriteFloatField(int fieldNumber, float value)
            {
                WriteTag(fieldNumber, 5);
                byte[] bytes = BitConverter.GetBytes(value);
                if (!BitConverter.IsLittleEndian)
                {
                    Array.Reverse(bytes);
                }
                _ms.Write(bytes, 0, 4);
            }

            public void WriteMessageField(int fieldNumber, byte[] submessage)
            {
                WriteTag(fieldNumber, 2);
                WriteVarint((ulong)submessage.Length);
                _ms.Write(submessage, 0, submessage.Length);
            }

            public void WriteFixed64Field(int fieldNumber, ulong value)
            {
                WriteTag(fieldNumber, 1);
                byte[] bytes = BitConverter.GetBytes(value);
                if (!BitConverter.IsLittleEndian)
                {
                    Array.Reverse(bytes);
                }
                _ms.Write(bytes, 0, 8);
            }

            public void WriteFixed32Field(int fieldNumber, uint value)
            {
                WriteTag(fieldNumber, 5);
                byte[] bytes = BitConverter.GetBytes(value);
                if (!BitConverter.IsLittleEndian)
                {
                    Array.Reverse(bytes);
                }
                _ms.Write(bytes, 0, 4);
            }

            public void WriteRaw(params byte[] bytes) =>
                _ms.Write(bytes, 0, bytes.Length);
        }

        /// <summary>Stream wrapper that hides seekability, forcing the fallback parse path.</summary>
        private sealed class NonSeekableStream : Stream
        {
            private readonly MemoryStream _inner;
            public NonSeekableStream(byte[] data) => _inner = new MemoryStream(data);

            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => throw new NotSupportedException();
            public override long Position
            {
                get => throw new NotSupportedException();
                set => throw new NotSupportedException();
            }
            public override void Flush() { }
            public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            protected override void Dispose(bool disposing)
            {
                if (disposing) _inner.Dispose();
                base.Dispose(disposing);
            }
        }
    }
}
