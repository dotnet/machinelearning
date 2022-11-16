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
    internal abstract class BaseModel : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
    {
        protected readonly NasBertTrainer.Options Options;
        public BertTaskType HeadType => Options.TaskType;

        //public ModelType EncoderType => Options.ModelType;

#pragma warning disable CA1024 // Use properties where appropriate: Modules should be fields in TorchSharp
        public abstract TransformerEncoder GetEncoder();

        public abstract BaseHead GetHead();
#pragma warning restore CA1024 // Use properties where appropriate

        protected BaseModel(TextClassificationTrainer.Options options)
            : base(nameof(BaseModel))
        {
            Options = options ?? throw new ArgumentNullException(nameof(options));
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor srcTokens, torch.Tensor tokenMask = null)
            => throw new NotImplementedException();

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override void train(bool train = true)
        {
            base.train(train);
        }
    }
}
