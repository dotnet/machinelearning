// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    internal sealed class TextClassificationModel : BaseModel
    {
        private readonly PredictionHead _predictionHead;

        public override BaseHead GetHead() => _predictionHead;

        public TextClassificationModel(TextClassificationTrainer.Options options, Vocabulary vocabulary, int numClasses)
            : base(options, vocabulary)
        {
            _predictionHead = new PredictionHead(
                inputDim: Options.EncoderOutputDim,
                numClasses: numClasses,
                dropoutRate: Options.PoolerDropout);
            Initialize();
            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor srcTokens, torch.Tensor tokenMask = null)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x = ExtractFeatures(srcTokens);
            x = _predictionHead.forward(x);
            return x.MoveToOuterDisposeScope();
        }
    }
}
