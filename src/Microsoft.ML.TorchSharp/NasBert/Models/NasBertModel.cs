// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    internal class NasBertModel : BaseModel
    {
        private readonly PredictionHead _predictionHead;

        public override BaseHead GetHead() => _predictionHead;
        public override TransformerEncoder GetEncoder() => Encoder;

        protected readonly TransformerEncoder Encoder;

        public NasBertModel(TextClassificationTrainer.Options options, int padIndex, int symbolsCount, int numClasses)
            : base(options)
        {
            _predictionHead = new PredictionHead(
                inputDim: Options.EncoderOutputDim,
                numClasses: numClasses,
                dropoutRate: Options.PoolerDropout);

            Encoder = new TransformerEncoder(
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

            Initialize();
            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor srcTokens, torch.Tensor mask = null)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x = ExtractFeatures(srcTokens);
            x = _predictionHead.forward(x);
            return x.MoveToOuterDisposeScope();
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
            return Encoder.forward(srcTokens, null, null);
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

    }
}
