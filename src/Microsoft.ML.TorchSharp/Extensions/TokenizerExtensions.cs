// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.TorchSharp.Utils;

namespace Microsoft.ML.TorchSharp.Extensions
{
    internal static class TokenizerExtensions
    {
        private const string EncoderJsonName = "encoder.json";
        private const string MergeName = "vocab.bpe";
        private const string DictName = "dict.txt";

        private static readonly Uri _encoderJsonUrl = new Uri("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json");
        private static readonly Uri _mergeUrl = new Uri("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe");
        private static readonly Uri _dictUrl = new Uri("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt");

        private static Tokenizer _instance;

        internal static Tokenizer GetInstance(IChannel ch)
        {
            if (_instance is null)
            {
                FileUtils.LoadFromFileOrDownloadFromWeb(string.Empty, EncoderJsonName, _encoderJsonUrl, ch);
                FileUtils.LoadFromFileOrDownloadFromWeb(string.Empty, MergeName, _mergeUrl, ch);
                FileUtils.LoadFromFileOrDownloadFromWeb(string.Empty, DictName, _dictUrl, ch);

                EnglishRoberta model = new EnglishRoberta(EncoderJsonName, MergeName, DictName);
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