// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert.Modules.Layers
{
    internal sealed class IdentityLayer : Layer
    {
        public IdentityLayer() : base(nameof(IdentityLayer))
        {
            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, Dictionary<string, object> param = null)
        {
            return x.alias();
        }
    }
}
