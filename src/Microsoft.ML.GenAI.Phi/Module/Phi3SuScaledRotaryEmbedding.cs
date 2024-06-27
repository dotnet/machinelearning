// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;

internal class Phi3SuScaledRotaryEmbedding : Phi3RotaryEmbedding
{
    private readonly double[] _shortFactor;
    private readonly double[] _longFactor;
    private readonly int _originalMaxPositionEmbeddings;
    private readonly int _maxPositionEmbeddings;
    private readonly double _base;

    public Phi3SuScaledRotaryEmbedding(int dim, Phi3Config config)
        : base(config.RopeTheta, config.MaxPositionEmbeddings, dim)
    {
        JsonElement shortFactorElement = (JsonElement)config.RopeScaling!["short_factor"];
        JsonElement longFactorDocument = (JsonElement)config.RopeScaling!["long_factor"];
        this._shortFactor = shortFactorElement.EnumerateArray().Select(e => e.GetDouble()).ToArray();
        this._longFactor = longFactorDocument.EnumerateArray().Select(e => e.GetDouble()).ToArray();

        this._originalMaxPositionEmbeddings = config.OriginalMaxPositionEmbeddings;
        this._maxPositionEmbeddings = config.MaxPositionEmbeddings;
        this._base = config.RopeTheta;
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Phi3RotaryEmbeddingOutput forward(Phi3RotaryEmbeddingInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        var seqLen = (torch.max(input.PositionIds) + 1).ToInt32();
        var x = input.Input;
        Tensor extFactors;
        if (seqLen > this._originalMaxPositionEmbeddings)
        {
            extFactors = torch.tensor(this._longFactor, dtype: ScalarType.Float32, x.device);
        }
        else
        {
            extFactors = torch.tensor(this._shortFactor, dtype: ScalarType.Float32, x.device);
        }
        var invFreqShape = torch.arange(0, this.Dim, 2, dtype: ScalarType.Int64).to(torch.float32) / this.Dim;
        invFreqShape = invFreqShape.to(x.device);
        var invFreq = 1.0f / (torch.pow(this._base, invFreqShape) * extFactors);

        var invFreqExpanded = invFreq.unsqueeze(0).unsqueeze(-1);
        invFreqExpanded = invFreqExpanded.expand(new long[] { input.PositionIds.shape[0], -1, 1 });
        var positionIdsExpanded = input.PositionIds.unsqueeze(1).to(torch.float32);

        var freqs = invFreqExpanded * positionIdsExpanded;
        freqs = freqs.transpose(1, 2);
        var emb = torch.cat([freqs, freqs], dim: -1);
        var scale = (1.0 * this._maxPositionEmbeddings) / this._originalMaxPositionEmbeddings;
        double scalingFactor;
        if (scale <= 1)
        {
            scalingFactor = 1.0;
        }
        else
        {
            scalingFactor = Math.Sqrt(1 + Math.Log(scale) / Math.Log(this._originalMaxPositionEmbeddings));
        }

        var cos = torch.cos(emb) * scalingFactor;
        var sin = torch.sin(emb) * scalingFactor;

        return new(cos.to_type(x.dtype), sin.to_type(x.dtype));
    }
}
