// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.TorchSharp.NasBert.Modules.Layers;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal abstract class TransformerCell : BaseModule
    {
        protected TransformerCell(string name) : base(name) { }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public abstract torch.Tensor forward(torch.Tensor x, torch.Tensor selfAttentionMask,
            torch.Tensor selfAttentionPaddingMask, int arch = 0, bool layerNormTraining = false);

        public virtual void CloseLayerNormTraining() { }
    }

    /// <summary>
    /// Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained models.
    /// Non-discrete cells are used when doing NAS search.
    /// </summary>
    internal sealed class TransformerCellNonDiscrete : TransformerCell
    {
        private readonly ActivationFunction _activationFn;

        /// <summary>
        /// The operations in search space.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly ModuleList Operations;

        public TransformerCellNonDiscrete(
            float dropout = 0.1f,
            float attentionDropout = 0.1f,
            float activationDropout = 0.1f,
            string activationFn = "relu",
            bool addBiasKv = false,
            bool addZeroAttention = false,
            bool dynamicDropout = false)
            : base(nameof(TransformerCellNonDiscrete))
        {
            _activationFn = new ActivationFunction(activationFn);
            var operations = Enumerable.Range(0, SearchSpace.NumLayerChoices)
                .Select(i => SearchSpace.GetLayer(i, dropout, attentionDropout, activationDropout, activationFn,
                    addBiasKv, addZeroAttention, dynamicDropout) as torch.nn.Module)
                .ToArray();
            Operations = new ModuleList(operations);

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, torch.Tensor selfAttentionMask,
            torch.Tensor selfAttentionPaddingMask, int arch = 0, bool layerNormTraining = false)
        {
            return (Operations[arch] as Layer)!.forward(x, new Dictionary<string, object>
            {
                {Layer.AttentionMaskKey, selfAttentionMask},
                {Layer.PaddingMaskKey, selfAttentionPaddingMask},
            });
        }

        public override void CloseLayerNormTraining()
        {
            foreach (var operation in Operations)
            {
                (operation as Layer)!.CloseLayerNormTraining();
            }
        }
    }

    /// <summary>
    /// Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained models.
    /// Discrete cells are used for a fixed neural architecture.
    /// </summary>
    internal sealed class TransformerCellDiscrete : TransformerCell
    {
        private readonly ActivationFunction _activationFn;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly Layer Operation;

        public TransformerCellDiscrete(
            int arch,
            double dropout = 0.1,
            double attentionDropout = 0.1,
            double activationDropout = 0.1,
            string activationFn = "relu",
            bool addBiasKv = false,
            bool addZeroAttention = false,
            bool dynamicDropout = false)
            : base(nameof(TransformerCellDiscrete))
        {
            _activationFn = new ActivationFunction(activationFn);
            Operation = SearchSpace.GetLayer(arch, dropout, attentionDropout, activationDropout, activationFn, addBiasKv,
                addZeroAttention, dynamicDropout);

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, torch.Tensor selfAttentionMask,
            torch.Tensor selfAttentionPaddingMask, int arch = 0, bool layerNormTraining = false)
        {
            return Operation.forward(x, new Dictionary<string, object>
            {
                {Layer.AttentionMaskKey, selfAttentionMask},
                {Layer.PaddingMaskKey, selfAttentionPaddingMask},
            });
        }

        public override void CloseLayerNormTraining() => Operation.CloseLayerNormTraining();
    }
}
