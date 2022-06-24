// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert
{
    internal abstract class BaseModule : torch.nn.Module
    {
        public int? InstanceId = null;

        protected BaseModule(string name) : base(name)
        {
        }

    }
}
