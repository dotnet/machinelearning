// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    internal class NerModel : NasBertModel
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "Has to match TorchSharp model.")]
        private readonly SequenceLabelHead NerHead;
        private bool _disposedValue;

        public override BaseHead GetHead() => NerHead;

        public NerModel(NasBertTrainer.NasBertOptions options, int padIndex, int symbolsCount, int numLabels)
            : base(options, padIndex, symbolsCount)
        {
            NerHead = new SequenceLabelHead(
                inputDim: Options.EncoderOutputDim,
                numLabels: numLabels,
                dropoutRate: Options.PoolerDropout);
            Initialize();
            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor srcTokens, torch.Tensor tokenMask = null)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x = ExtractFeatures(srcTokens);
            x = NerHead.call(x);
            return x.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    NerHead.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
