// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Optimizers
{
    internal sealed class Adam : BaseOptimizer
    {
        public Adam(TextClassificationTrainer.Options options, IEnumerable<Parameter> parameters)
            : base(nameof(Adam), options, parameters)
        {
            Optimizer = torch.optim.Adam(Parameters, options.LearningRate[0],
                options.AdamBetas[0], options.AdamBetas[1], options.AdamEps, options.WeightDecay);
        }
    }
}
