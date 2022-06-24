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
    internal abstract class BaseModel : BaseModule
    {
        protected readonly TextClassificationTrainer.Options Options;
        public BertTaskType HeadType => Options.TaskType;

        protected readonly TransformerEncoder Encoder;

#pragma warning disable CA1024 // Use properties where appropriate: Modules should be fields in TorchSharp 
        public TransformerEncoder GetEncoder() => Encoder;

        public abstract BaseHead GetHead();
#pragma warning restore CA1024 // Use properties where appropriate

        protected BaseModel(TextClassificationTrainer.Options options, Vocabulary vocabulary)
            : base(nameof(BaseModel))
        {
            vocabulary = vocabulary ?? throw new ArgumentNullException(nameof(vocabulary));
            Options = options ?? throw new ArgumentNullException(nameof(options));
            Encoder = new TransformerEncoder(
                paddingIdx: vocabulary.PadIndex,
                vocabSize: vocabulary.Count,
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

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public new abstract torch.Tensor forward(torch.Tensor srcTokens, torch.Tensor tokenMask = null);

        /// <summary>
        /// Run only Encoder and return features.
        /// </summary>
        protected torch.Tensor ExtractFeatures(torch.Tensor srcTokens)
        {
            return Encoder.forward(srcTokens, null, null);
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override void train()
        {
            base.train();
            if (!Options.LayerNormTraining)
            {
                Encoder.CloseLayerNormTraining();
            }
        }
    }
}
