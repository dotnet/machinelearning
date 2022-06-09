// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal abstract class EmbedTransfer : BaseModule
    {
        protected EmbedTransfer(string name) : base(name) { }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public abstract torch.Tensor forward(torch.Tensor x, int hiddenSize);
    }

    internal sealed class EmbedTransferNonDiscrete : EmbedTransfer
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly ModuleList HiddenTransfer;

        public EmbedTransferNonDiscrete() : base(nameof(EmbedTransferNonDiscrete))
        {
            //var hiddenTransfer = SearchSpace.HiddenSizeChoices[Range.EndAt(Index.FromEnd(1))]
            var hiddenTransfer = SearchSpace.HiddenSizeChoices.Where((source, index) => index != SearchSpace.HiddenSizeChoices.Length - 1)
                .Select(hidden => torch.nn.Linear(hidden, SearchSpace.HiddenSizeChoices[SearchSpace.HiddenSizeChoices.Length - 1]) as torch.nn.Module)
                .ToArray();
            HiddenTransfer = new ModuleList(hiddenTransfer);
            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, int hiddenSize)
        {
            if (hiddenSize == SearchSpace.HiddenSizeChoices[SearchSpace.HiddenSizeChoices.Length - 1]) return x;
            var index = SearchSpace.HiddenSizeChoices.ToList().IndexOf(hiddenSize);
            return index == -1
                ? x.alias()
                : HiddenTransfer[index].forward(x);
        }
    }

    internal sealed class EmbedTransferDiscrete : EmbedTransfer
    {
#nullable enable
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly Linear? HiddenTransfer;
#nullable disable

        public EmbedTransferDiscrete(int embedSize, int hiddenSize) : base(nameof(EmbedTransferDiscrete))
        {
            if (embedSize != hiddenSize)
            {
                HiddenTransfer = torch.nn.Linear(embedSize, hiddenSize);

                ModelUtils.InitXavierUniform(HiddenTransfer.weight);
                ModelUtils.InitZeros(HiddenTransfer.bias);
            }

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, int hiddenSize)
        {
            return HiddenTransfer == null
                ? x.alias()
                : HiddenTransfer.forward(x);
        }
    }
}
