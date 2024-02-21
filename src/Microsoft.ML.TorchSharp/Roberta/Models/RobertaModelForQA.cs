// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TorchSharp.NasBert.Models;
using Microsoft.ML.TorchSharp.Roberta;
using Microsoft.ML.TorchSharp.Roberta.Models;
using TorchSharp;

internal class RobertaModelForQA : RobertaModel
{
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly SequenceLabelHead QAHead;

    private readonly int NumClasses = 2;

    public override BaseHead GetHead() => QAHead;

    private bool _disposedValue;

    public RobertaModelForQA(QATrainer.Options options)
        : base(options)
    {
        QAHead = new SequenceLabelHead(
            inputDim: Options.EncoderOutputDim,
            numLabels: NumClasses,
            dropoutRate: Options.PoolerDropout);
        apply(InitWeights);
        RegisterComponents();
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override torch.Tensor forward(torch.Tensor srcTokens, torch.Tensor tokenMask = null)
    {
        using var disposeScope = torch.NewDisposeScope();
        var x = ExtractFeatures(srcTokens);
        x = QAHead.call(x);
        return x.MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                QAHead.Dispose();
                _disposedValue = true;
            }
        }

        base.Dispose(disposing);
    }
}
