// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.TorchSharp.Extensions;
using TorchSharp;
using TorchSharp.Modules;
using Microsoft.ML.TorchSharp.NasBert.Models;
using TransformerEncoder = Microsoft.ML.TorchSharp.NasBert.Models.TransformerEncoder;
using Microsoft.ML.TorchSharp.Utils;

namespace Microsoft.ML.TorchSharp.Roberta.Models
{
    /// <summary>
    /// Base Roberta model without output heads.
    /// </summary>
    internal abstract class RobertaModel : BaseModel
    {
        private bool _disposedValue;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
        private readonly int[] Positions;
        private readonly int[] Zeros;
        private readonly int[] Ones;
        private readonly int[] NegBillionPad;

#pragma warning disable CS0649
        protected readonly LayerNorm LayerNorm;

        protected readonly RobertaEncoder Encoder;

        private const int PadIndex = 1;
        private const int EosIndex = 2;
        public override TransformerEncoder GetEncoder() => Encoder;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

        protected RobertaModel(QATrainer.Options options)
            : base(options)
        {
            var negBillion = (int)-1e9;
            Positions = Enumerable.Range(0, options.MaxSequenceLength).ToArray();
            Zeros = Enumerable.Repeat(0, options.MaxSequenceLength).ToArray();
            Ones = Enumerable.Repeat(1, options.MaxSequenceLength).ToArray();
            NegBillionPad = Enumerable.Repeat(negBillion, options.MaxSequenceLength).ToArray();

            Encoder = new RobertaEncoder(
                numLayers: 12,
                numAttentionHeads: 12,
                numEmbeddings: 50265,
                embeddingSize: 768,
                hiddenSize: 768,
                outputSize: 768,
                ffnHiddenSize: 3072,
                maxPositions: 512,
                maxTokenTypes: 2,
                layerNormEps: 1e-12,
                embeddingDropoutRate: 0.1,
                attentionDropoutRate: 0.1,
                attentionOutputDropoutRate: 0.1,
                outputDropoutRate: 0.1);
        }

        protected void InitWeights(torch.nn.Module module)
        {
            using var disposeScope = torch.NewDisposeScope();
            if (module is Linear linearModule)
            {
                linearModule.weight.normal_(mean: 0.0, std: 0.02);
                if (linearModule.bias.IsNotNull())
                {
                    linearModule.bias.zero_();
                }
            }
            else if (module is Embedding embeddingModule)
            {
                embeddingModule.weight.normal_(mean: 0.0, std: 0.02);
                embeddingModule.weight[1].zero_();  // padding_idx
            }
            else if (module is LayerNorm layerNormModule)
            {
                layerNormModule.weight.fill_(1.0);
                layerNormModule.bias.zero_();
            }
        }

        /// <summary>
        /// Run only Encoder and return features.
        /// </summary>
        protected torch.Tensor ExtractFeatures(torch.Tensor srcTokens)
        {
            var (positions, segments, attentions) = GetEmbeddings(srcTokens);
            var encodedVector = Encoder.call(srcTokens, positions, segments, attentions);
            return encodedVector;
        }

        private (torch.Tensor position, torch.Tensor segment, torch.Tensor attentionMask) GetEmbeddings(torch.Tensor srcTokens)
        {
            using var disposeScope = torch.NewDisposeScope();
            var device = srcTokens.device;
            var srcSize = srcTokens.size(0);
            var positions = new torch.Tensor[srcSize];
            var segments = new torch.Tensor[srcSize];
            var attentionMasks = new torch.Tensor[srcSize];

            for (var i = 0; i < srcSize; ++i)
            {
                var srcTokenArray = srcTokens[i].ToArray<int>();

                var size = srcTokenArray.Length;
                var questionSize = srcTokenArray.AsSpan().IndexOf(EosIndex) - 1;

                var allSize = srcTokenArray.Count(token => token != PadIndex);

                var position = torch.tensor(DataUtils.Concat<int>(Positions.AsSpan(0, allSize), Zeros.AsSpan(0, size - allSize)),
                    1, size, dtype: torch.int64, device: device);
                var segment = questionSize == size - 1 ? torch.tensor(Zeros.AsSpan(0, size).ToArray(), 1, size, dtype: torch.int64, device: device) :
                    torch.tensor(DataUtils.Concat<int>(Zeros.AsSpan(0, questionSize + 2), Ones.AsSpan(0, allSize - questionSize - 2), Zeros.AsSpan(0, size - allSize)),
                    1, size, dtype: torch.int64, device: device);
                var attentionMask = torch.tensor(DataUtils.Concat<int>(Zeros.AsSpan(0, allSize), NegBillionPad.AsSpan(0, size - allSize)),
                    new long[] { 1, 1, 1, size }, dtype: torch.float32, device: device);

                positions[i] = position;
                segments[i] = segment;
                attentionMasks[i] = attentionMask;
            }

            return (torch.cat(positions, dim: 0).MoveToOuterDisposeScope(),
                torch.cat(segments, dim: 0).MoveToOuterDisposeScope(),
                torch.cat(attentionMasks, dim: 0).MoveToOuterDisposeScope());
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    LayerNorm?.Dispose();
                    Encoder.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
