// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.TorchSharp.Utils;

namespace Microsoft.ML.TorchSharp.Extensions
{
    internal static class TokenizerExtensions
    {
        private static Tokenizer _instance;

        internal static Tokenizer GetInstance(IChannel ch)
        {
            if (_instance is null)
            {
                // encoder.json, vocab.bpe, and dict.txt are picked up from the following source:
                // "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
                // "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"
                // "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt"
                Assembly assembly = typeof(TokenizerExtensions).Assembly;

                EnglishRoberta model = new EnglishRoberta(
                                            assembly.GetManifestResourceStream("encoder.json"),
                                            assembly.GetManifestResourceStream("vocab.bpe"),
                                            assembly.GetManifestResourceStream("dict.txt"));
                model.AddMaskSymbol();
                _instance = new Tokenizer(model, new RobertaPreTokenizer());
            }

            return _instance;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static EnglishRoberta RobertaModel(this Tokenizer tokenizer)
        {
            EnglishRoberta model = tokenizer.Model as EnglishRoberta;
            if (model is null)
            {
                throw new InvalidOperationException($"The input tokenizer is not using the EnglishRoberta model.");
            }

            return model;
        }

        internal static IReadOnlyList<int> EncodeToConverted(this Tokenizer tokenizer, string sentence)
        {
            TokenizerResult encoding = tokenizer.Encode(sentence);
            return tokenizer.RobertaModel().IdsToOccurrenceRanks(encoding.Ids);
        }
    }
}