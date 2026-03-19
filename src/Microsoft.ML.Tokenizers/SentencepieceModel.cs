// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Minimal protobuf parser for SentencePiece model files (sentencepiece_model.proto).
// Replaces a full Google.Protobuf dependency with just enough wire-format reading
// to parse the fields the tokenizer implementation actually consumes.
// SentencePiece is under the Apache License 2.0 https://github.com/google/sentencepiece/blob/master/LICENSE

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Sentencepiece
{
    /// <summary>Low-level protobuf wire-format primitives (read-only, forward-only).</summary>
    internal static class SentencePieceProtobufReader
    {
        internal static int ReadRawVarint32(byte[] data, int end, ref int pos)
        {
            if (pos >= end)
            {
                throw new InvalidDataException("Unexpected end of data while reading varint.");
            }

            byte b = data[pos++];
            int result = b & 0x7F;
            if ((b & 0x80) == 0)
            {
                return result;
            }

            for (int shift = 7; shift < 32; shift += 7)
            {
                if (pos >= end)
                {
                    throw new InvalidDataException("Unexpected end of data while reading varint.");
                }

                b = data[pos++];
                result |= (b & 0x7F) << shift;
                if ((b & 0x80) == 0)
                {
                    return result;
                }
            }

            // Negative int32 values are sign-extended to 10-byte varints; consume remaining bytes.
            for (int i = 0; i < 5; i++)
            {
                if (pos >= end)
                {
                    throw new InvalidDataException("Unexpected end of data while reading varint.");
                }

                if ((data[pos++] & 0x80) == 0)
                {
                    return result;
                }
            }

            throw new InvalidDataException("Malformed varint.");
        }

        internal static int ReadLengthPrefix(byte[] data, int end, ref int pos)
        {
            int length = ReadRawVarint32(data, end, ref pos);
            if ((uint)length > (uint)(end - pos))
            {
                throw new InvalidDataException("Invalid length-delimited field size.");
            }

            return length;
        }

        internal static string ReadString(byte[] data, int end, ref int pos)
        {
            int length = ReadLengthPrefix(data, end, ref pos);
            string result = Encoding.UTF8.GetString(data, pos, length);
            pos += length;
            return result;
        }

        internal static float ReadFloat(byte[] data, int end, ref int pos)
        {
            if (pos > end - 4)
            {
                throw new InvalidDataException("Unexpected end of data while reading float.");
            }

            float value;
            if (BitConverter.IsLittleEndian)
            {
                value = BitConverter.ToSingle(data, pos);
            }
            else
            {
                // Protobuf fixed32 is always little-endian; reverse bytes on big-endian platforms.
                byte[] buffer = new byte[4];
                buffer[0] = data[pos + 3];
                buffer[1] = data[pos + 2];
                buffer[2] = data[pos + 1];
                buffer[3] = data[pos];
                value = BitConverter.ToSingle(buffer, 0);
            }

            pos += 4;
            return value;
        }

        internal static void SkipField(byte[] data, int end, int wireType, ref int pos)
        {
            switch (wireType)
            {
                case 0: // varint (max 10 bytes per protobuf spec)
                    for (int i = 0; i < 10; i++)
                    {
                        if (pos >= end)
                        {
                            throw new InvalidDataException("Unexpected end of data while skipping varint.");
                        }

                        if ((data[pos++] & 0x80) == 0)
                        {
                            break;
                        }

                        if (i == 9)
                        {
                            throw new InvalidDataException("Malformed varint.");
                        }
                    }
                    break;

                case 1: // 64-bit fixed
                    if (pos > end - 8)
                    {
                        throw new InvalidDataException("Unexpected end of data while skipping fixed64.");
                    }
                    pos += 8;
                    break;

                case 2: // length-delimited
                    int skipLength = ReadLengthPrefix(data, end, ref pos);
                    pos += skipLength;
                    break;

                case 5: // 32-bit fixed
                    if (pos > end - 4)
                    {
                        throw new InvalidDataException("Unexpected end of data while skipping fixed32.");
                    }
                    pos += 4;
                    break;

                default:
                    throw new InvalidDataException($"Unknown or unsupported protobuf wire type {wireType}.");
            }
        }
    }

    /// <summary>Lightweight replacement for Google.Protobuf.ByteString with a Span property.</summary>
    internal readonly struct SentencePieceByteString(byte[] data, int offset, int length)
    {
        internal ReadOnlySpan<byte> Span => data is null ? ReadOnlySpan<byte>.Empty : data.AsSpan(offset, length);
    }

    /// <summary>ModelProto  (top-level message; field numbers match sentencepiece_model.proto)</summary>
    internal sealed class ModelProto
    {
        internal static readonly ModelProtoParser Parser = new();

        internal List<Types.SentencePiece> Pieces { get; } = new();
        internal TrainerSpec TrainerSpec { get; private set; } = new();
        internal NormalizerSpec NormalizerSpec { get; private set; } = new();

        internal static ModelProto Parse(byte[] data, int start, int end)
        {
            ModelProto result = new();
            int pos = start;
            int length;
            while (pos < end)
            {
                int tag = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                int fieldNumber = tag >> 3;
                int wireType = tag & 7;

                // The 'when wireType == 2' guards serve double duty: they match the expected wire
                // type for these message fields AND provide forward-compatibility — if a future proto
                // version changes a field's type, or if extension fields reuse these numbers with a
                // different wire type, the mismatch falls through to the default skip case.
                //
                // For non-repeated message fields (TrainerSpec, NormalizerSpec), seeing the same field
                // number twice replaces the prior value (last wins). This differs from the standard
                // protobuf library which merges repeated occurrences of non-repeated message fields.
                // SentencePiece model files contain each field at most once, so the difference is moot
                // in practice.
                switch (fieldNumber)
                {
                    case 1 when wireType == 2: // repeated SentencePiece pieces = 1
                        length = SentencePieceProtobufReader.ReadLengthPrefix(data, end, ref pos);
                        result.Pieces.Add(Types.SentencePiece.Parse(data, pos, pos + length));
                        pos += length;
                        break;

                    case 2 when wireType == 2: // TrainerSpec trainer_spec = 2
                        length = SentencePieceProtobufReader.ReadLengthPrefix(data, end, ref pos);
                        result.TrainerSpec = TrainerSpec.Parse(data, pos, pos + length);
                        pos += length;
                        break;

                    case 3 when wireType == 2: // NormalizerSpec normalizer_spec = 3
                        length = SentencePieceProtobufReader.ReadLengthPrefix(data, end, ref pos);
                        result.NormalizerSpec = NormalizerSpec.Parse(data, pos, pos + length);
                        pos += length;
                        break;

                    default:
                        SentencePieceProtobufReader.SkipField(data, end, wireType, ref pos);
                        break;
                }
            }

            return result;
        }

        internal static class Types
        {
            internal sealed class SentencePiece
            {
                internal string Piece { get; set; } = "";
                internal float Score { get; set; }
                internal Types.Type Type { get; set; } = Types.Type.Normal;

                internal static SentencePiece Parse(byte[] data, int start, int end)
                {
                    SentencePiece result = new();
                    int pos = start;
                    while (pos < end)
                    {
                        int tag = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                        int fieldNumber = tag >> 3;
                        int wireType = tag & 7;

                        switch (fieldNumber)
                        {
                            case 1 when wireType == 2: // string piece = 1
                                result.Piece = SentencePieceProtobufReader.ReadString(data, end, ref pos);
                                break;

                            case 2 when wireType == 5: // float score = 2
                                result.Score = SentencePieceProtobufReader.ReadFloat(data, end, ref pos);
                                break;

                            case 3 when wireType == 0: // Type type = 3
                                result.Type = (Types.Type)SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                                break;

                            default:
                                SentencePieceProtobufReader.SkipField(data, end, wireType, ref pos);
                                break;
                        }
                    }

                    return result;
                }

                internal static class Types
                {
                    internal enum Type
                    {
                        Normal = 1,
                        Unknown = 2,
                        Control = 3,
                        UserDefined = 4,
                        Unused = 5,
                        Byte = 6,
                    }
                }
            }
        }
    }

    /// <summary>ModelProtoParser  (entry point: ModelProto.Parser.ParseFrom(Stream))</summary>
    internal sealed class ModelProtoParser
    {
        internal ModelProto ParseFrom(Stream stream)
        {
            if (stream is null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            // Fast-path: if the input is already a MemoryStream with an accessible buffer,
            // parse directly from its underlying array without copying.
            if (stream is MemoryStream memoryStream &&
                memoryStream.TryGetBuffer(out ArraySegment<byte> segment))
            {
                int start = segment.Offset + (int)memoryStream.Position;
                int end = segment.Offset + (int)memoryStream.Length;
                return ModelProto.Parse(segment.Array!, start, end);
            }

            // Fallback: copy remaining data into a new MemoryStream, pre-sizing when possible.
            MemoryStream ms;
            if (stream.CanSeek)
            {
                long remaining = stream.Length - stream.Position;
                ms = remaining > 0 && remaining <= int.MaxValue ? new MemoryStream((int)remaining) : new MemoryStream();
            }
            else
            {
                ms = new MemoryStream();
            }

            stream.CopyTo(ms);
            return ModelProto.Parse(ms.GetBuffer(), 0, (int)ms.Length);
        }
    }

    /// <summary>TrainerSpec  (defaults match sentencepiece_model.proto)</summary>
    internal sealed class TrainerSpec
    {
        internal Types.ModelType ModelType { get; private set; } = Types.ModelType.Unigram;
        internal bool TreatWhitespaceAsSuffix { get; private set; }
        internal bool ByteFallback { get; private set; }
        internal int UnkId { get; private set; }
        internal int BosId { get; private set; } = 1;
        internal int EosId { get; private set; } = 2;
        internal int PadId { get; private set; } = -1;
        internal string UnkPiece { get; private set; } = "<unk>";
        internal string BosPiece { get; private set; } = "<s>";
        internal string EosPiece { get; private set; } = "</s>";
        internal string PadPiece { get; private set; } = "<pad>";

        internal static TrainerSpec Parse(byte[] data, int start, int end)
        {
            TrainerSpec result = new();
            int pos = start;
            while (pos < end)
            {
                int tag = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                int fieldNumber = tag >> 3;
                int wireType = tag & 7;

                switch (fieldNumber)
                {
                    case 3 when wireType == 0:  // ModelType model_type = 3
                        result.ModelType = (Types.ModelType)SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                        break;

                    case 24 when wireType == 0: // bool treat_whitespace_as_suffix = 24
                        result.TreatWhitespaceAsSuffix = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos) != 0;
                        break;

                    case 35 when wireType == 0: // bool byte_fallback = 35
                        result.ByteFallback = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos) != 0;
                        break;

                    case 40 when wireType == 0: // int32 unk_id = 40
                        result.UnkId = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                        break;

                    case 41 when wireType == 0: // int32 bos_id = 41
                        result.BosId = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                        break;

                    case 42 when wireType == 0: // int32 eos_id = 42
                        result.EosId = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                        break;

                    case 43 when wireType == 0: // int32 pad_id = 43
                        result.PadId = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                        break;

                    case 45 when wireType == 2: // string unk_piece = 45
                        result.UnkPiece = SentencePieceProtobufReader.ReadString(data, end, ref pos);
                        break;

                    case 46 when wireType == 2: // string bos_piece = 46
                        result.BosPiece = SentencePieceProtobufReader.ReadString(data, end, ref pos);
                        break;

                    case 47 when wireType == 2: // string eos_piece = 47
                        result.EosPiece = SentencePieceProtobufReader.ReadString(data, end, ref pos);
                        break;

                    case 48 when wireType == 2: // string pad_piece = 48
                        result.PadPiece = SentencePieceProtobufReader.ReadString(data, end, ref pos);
                        break;

                    default:
                        SentencePieceProtobufReader.SkipField(data, end, wireType, ref pos);
                        break;
                }
            }

            return result;
        }

        internal static class Types
        {
            internal enum ModelType
            {
                Unigram = 1,
                Bpe = 2,
                Word = 3,
                Char = 4,
            }
        }
    }

    /// <summary>NormalizerSpec  (defaults match sentencepiece_model.proto)</summary>
    internal sealed class NormalizerSpec
    {
        internal string Name { get; private set; } = "";
        internal SentencePieceByteString PrecompiledCharsmap { get; private set; }
        internal bool AddDummyPrefix { get; private set; } = true;
        internal bool RemoveExtraWhitespaces { get; private set; } = true;
        internal bool EscapeWhitespaces { get; private set; } = true;

        internal static NormalizerSpec Parse(byte[] data, int start, int end)
        {
            NormalizerSpec result = new();
            int pos = start;
            while (pos < end)
            {
                int tag = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos);
                int fieldNumber = tag >> 3;
                int wireType = tag & 7;

                switch (fieldNumber)
                {
                    case 1 when wireType == 2: // string name = 1
                        result.Name = SentencePieceProtobufReader.ReadString(data, end, ref pos);
                        break;

                    case 2 when wireType == 2: // bytes precompiled_charsmap = 2
                        int length = SentencePieceProtobufReader.ReadLengthPrefix(data, end, ref pos);
                        result.PrecompiledCharsmap = new SentencePieceByteString(data, pos, length);
                        pos += length;
                        break;

                    case 3 when wireType == 0: // bool add_dummy_prefix = 3
                        result.AddDummyPrefix = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos) != 0;
                        break;

                    case 4 when wireType == 0: // bool remove_extra_whitespaces = 4
                        result.RemoveExtraWhitespaces = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos) != 0;
                        break;

                    case 5 when wireType == 0: // bool escape_whitespaces = 5
                        result.EscapeWhitespaces = SentencePieceProtobufReader.ReadRawVarint32(data, end, ref pos) != 0;
                        break;

                    default:
                        SentencePieceProtobufReader.SkipField(data, end, wireType, ref pos);
                        break;
                }
            }

            return result;
        }
    }
}
