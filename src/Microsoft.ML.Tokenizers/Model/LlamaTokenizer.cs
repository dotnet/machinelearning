// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Tokenizers
{
    // SentencePiece is under the Apache License 2.0 https://github.com/google/sentencepiece/blob/master/LICENSE

    /// <summary>
    /// LlamaTokenizer is SentencePieceTokenizer which is implemented based on https://github.com/google/sentencepiece.
    /// </summary>
    public sealed class LlamaTokenizer : SentencePieceTokenizer
    {
        internal LlamaTokenizer(ModelProto modelProto, bool addBos, bool addEos, IReadOnlyDictionary<string, int>? addedTokens = null) : base(modelProto, addBos, addEos, addedTokens)
        {
        }

        /// <summary>
        /// Create from the given model stream a LlamaTokenizer which is based on SentencePieceTokenizer. The model stream should contain the SentencePiece Bpe model according to
        /// https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto specification.
        /// </summary>
        /// <param name="modelStream">The stream containing the SentencePiece Bpe model.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="specialTokens">The additional tokens to add to the vocabulary.</param>
        public static LlamaTokenizer Create(
            Stream modelStream,
            bool addBeginOfSentence = true,
            bool addEndOfSentence = false,
            IReadOnlyDictionary<string, int>? specialTokens = null)
        {
            ModelProto modelProto = ModelProto.Parser.ParseFrom(modelStream);

            if (modelProto is null)
            {
                throw new ArgumentNullException(nameof(modelProto));
            }

            if (modelProto.TrainerSpec.ModelType != TrainerSpec.Types.ModelType.Bpe)
            {
                throw new ArgumentException("The model type is not Bpe.", nameof(modelProto));
            }

            if (modelProto.NormalizerSpec.Name != "identity" && !string.IsNullOrEmpty(modelProto.NormalizerSpec.Name))
            {
                throw new ArgumentException($"Normalization '{modelProto.NormalizerSpec.Name}' is not supported.", nameof(modelProto));
            }

            SentencePieceNormalizer normalizer = new(
                                    modelProto.NormalizerSpec.RemoveExtraWhitespaces,
                                    modelProto.NormalizerSpec.AddDummyPrefix,
                                    modelProto.NormalizerSpec.EscapeWhitespaces,
                                    modelProto.TrainerSpec.TreatWhitespaceAsSuffix,
                                    specialTokens);

            return new LlamaTokenizer(modelProto, addBeginOfSentence, addEndOfSentence, specialTokens);
        }
    }
}
