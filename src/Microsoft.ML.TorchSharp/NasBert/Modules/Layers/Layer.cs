// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert.Modules.Layers
{
    internal class Layer : BaseModule
    {
        public const string AttentionMaskKey = "selfAttentionMask";
        public const string PaddingMaskKey = "selfAttentionPaddingMask";

        protected Layer(string name) : base(name)
        {
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public virtual torch.Tensor forward(torch.Tensor x, Dictionary<string, object> param = null)
        {
            return x.alias();
        }

        /// <summary>
        ///  Set LayerNorm layers training status. This method should be invoked after Train().
        /// </summary>
        public virtual void CloseLayerNormTraining() { }
    }
}
