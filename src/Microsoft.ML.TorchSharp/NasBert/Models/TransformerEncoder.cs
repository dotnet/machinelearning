// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.NasBert.Modules;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    public abstract class TransformerEncoder : torch.nn.Module
    {
        protected TransformerEncoder(string name) : base(name)
        {
        }
    }
}
