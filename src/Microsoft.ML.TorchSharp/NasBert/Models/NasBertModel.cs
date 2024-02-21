﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    internal abstract class NasBertModel : BaseModel
    {
        public override TransformerEncoder GetEncoder() => Encoder;

        protected readonly NasBertEncoder Encoder;
        private bool _disposedValue;

        public NasBertModel(NasBertTrainer.NasBertOptions options, int padIndex, int symbolsCount)
            : base(options)
        {
            Encoder = new NasBertEncoder(
                paddingIdx: padIndex,
                vocabSize: symbolsCount,
                dropout: Options.Dropout,
                attentionDropout: Options.AttentionDropout,
                activationDropout: Options.ActivationDropout,
                activationFn: Options.ActivationFunction,
                dynamicDropout: Options.DynamicDropout,
                maxSeqLen: Options.MaxSequenceLength,
                embedSize: Options.EmbeddingDim,
                arches: Options.Arches?.ToList(),
                numSegments: 0,
                encoderNormalizeBefore: Options.EncoderNormalizeBefore,
                numEncoderLayers: Options.EncoderLayers,
                applyBertInit: true,
                freezeTransfer: Options.FreezeTransfer);
        }

        protected void Initialize()
        {
            if (Options.FreezeEncoder)
            {
                ModelUtils.FreezeModuleParams(Encoder);
            }
        }

        /// <summary>
        /// Run only Encoder and return features.
        /// </summary>
        protected torch.Tensor ExtractFeatures(torch.Tensor srcTokens)
        {
            return Encoder.call(srcTokens, null, null);
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override void train(bool train = true)
        {
            base.train(train);
            if (!Options.LayerNormTraining)
            {
                Encoder.CloseLayerNormTraining();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    Encoder.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }

    }
}
