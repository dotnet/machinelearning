// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.GenAI.Core;
using TorchSharp;
using static TorchSharp.torch;
namespace Microsoft.ML.GenAI;

internal class QuantizedLinear : GenAILinear, IQuantizeModule
{
    public QuantizedLinear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(inFeatures, outFeatures, hasBias, dtype, device)
    {
    }

    public void Int8()
    {
        if (this.weight is null)
        {
            throw new Exception("Weight is not initialized");
        }

        if (this.weight.device_type != DeviceType.META)
        {
            // if weight is not on meta device, this means that weight and bias are already loaded
            // so we can quantize them in memory

            var timer = new System.Diagnostics.Stopwatch();
            timer.Start();
            // scale and zero point on vector-wise
            // scale = 255 / max(weight, axis=1) - min(weight, axis=1)
            var scale = 255 / (torch.max(this.weight, 1).values - torch.min(this.weight, 1).values);

            // zero point = - scale * min(weight, axis=1) - 128
            var zeroPoint = -scale * torch.min(this.weight, 1).values - 128;
            // round zero point to nearest integer
            zeroPoint = torch.round(zeroPoint).to(torch.int8);

            // assert zero point is in range [-128, 127]
            //if (torch.any(this.zeroPoint < -128).item<bool>() || torch.any(this.zeroPoint > 127).item<bool>())
            //{
            //    throw new Exception("Zero point is out of range [-128, 127]");
            //}

            // quantize weight
            var eightBitWeight = torch.round(this.weight * scale.view(-1, 1) + zeroPoint.view(-1, 1)).to(torch.int8);

            // assert weight is in range [-128, 127]
            //if (torch.any(this._8bitWeight < -128).item<bool>() || torch.any(this._8bitWeight > 127).item<bool>())
            //{
            //    throw new Exception("Weight is out of range [-128, 127]");
            //}
            timer.Stop();
            // dispose float32 weight
            this.weight.Dispose();
            this.weight = null;
            this._internal_buffers.Remove("weight");
            this.register_buffer("8bit_weight", eightBitWeight);
            this.register_buffer("zeroPoint", zeroPoint);
            this.register_buffer("scale", scale);
        }
        else
        {
            // if weight is on meta device, then we just need to create the placeholder for 8bit_weight, zeroPoint and scale
            var eightBitWeight = torch.zeros(this.weight.shape, dtype: torch.int8);
            var zeroPoint = torch.zeros(this.weight.shape[0], dtype: torch.int8);
            var scale = torch.zeros(this.weight.shape[0], dtype: torch.float32);

            this._internal_buffers.Remove("weight");
            this.weight = null;
            this.register_buffer("8bit_weight", eightBitWeight);
            this.register_buffer("zeroPoint", zeroPoint);
            this.register_buffer("scale", scale);
        }
    }
#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        if (this._internal_buffers.ContainsKey("weight"))
        {
            return base.forward(input);
        }
        else if (this._internal_buffers.ContainsKey("8bit_weight"))
        {
            // 8bit quantization
            using var dispose = torch.NewDisposeScope();
            var weight = this.get_buffer("8bit_weight").to(ScalarType.Float32);
            var zeroPoint = this.get_buffer("zeroPoint").to(ScalarType.Float32);
            var scale = this.get_buffer("scale").to(ScalarType.Float32);
            var restoreWeight = (weight - zeroPoint.view(-1, 1)) / scale.view(-1, 1);
            // use float32
            var result = torch.matmul(input.to(ScalarType.Float32), restoreWeight.T);

            if (this.bias is not null)
            {
                result = result + this.bias.to_type(ScalarType.Float32);
            }

            //result.Peek("result");
            return result.to_type(input.dtype).MoveToOuterDisposeScope();
        }
        else if (this._internal_buffers.ContainsKey("4bit_weight"))
        {
            using var dispose = torch.NewDisposeScope();
            var weight = this.get_buffer("4bit_weight");
            var weightLower = weight % 16;
            var weightUpper = weight / 16;
            weight = torch.cat([weightUpper, weightLower], 0).to(ScalarType.Float32);
            weight = weight.view(this._outFeatures, this._inFeatures);
            weight -= 8;
            var zeroPoint = this.get_buffer("zeroPoint");
            var zeroPointLower = zeroPoint % 16;
            var zeroPointUpper = zeroPoint / 16;
            zeroPoint = torch.cat([zeroPointUpper, zeroPointLower], 0).to(ScalarType.Float32);
            zeroPoint -= 8;
            var scale = this.get_buffer("scale").to(ScalarType.Float32);
            var restoreWeight = (weight - zeroPoint.view(-1, 1)) / scale.view(-1, 1);
            // use float32
            var result = torch.matmul(input.to(ScalarType.Float32), restoreWeight.T);

            if (this.bias is not null)
            {
                result = result + this.bias.to_type(ScalarType.Float32);
            }

            //result.Peek("result");
            return result.to_type(input.dtype).MoveToOuterDisposeScope();
        }
        else
        {
            throw new Exception("Quantization is not done yet");
        }
    }

    public void Int4()
    {
        if (this.weight is null)
        {
            throw new Exception("Weight is not initialized");
        }
        var placeHolderDim = this._outFeatures / 2 + this._outFeatures % 2;
        var fourBitWeightDim = this.weight.size(0) * this.weight.size(1);
        var fourBitWeightPlaceHolderDim = Convert.ToInt32(fourBitWeightDim / 2 + fourBitWeightDim % 2);
        if (this.weight.device_type != DeviceType.META)
        {
            using var scope = NewDisposeScope();
            var timer = new System.Diagnostics.Stopwatch();
            timer.Start();
            // scale and zero point on vector-wise
            // scale = 15 / max(weight, axis=1) - min(weight, axis=1)
            var scale = 15 / (torch.max(this.weight, 1).values - torch.min(this.weight, 1).values);

            // zero point = - scale * min(weight, axis=1) - 8
            var zeroPoint = -scale * torch.min(this.weight, 1).values - 8;
            // round zero point to nearest integer
            zeroPoint = torch.round(zeroPoint);
            var fourBitWeight = torch.round(this.weight * scale.view(-1, 1) + zeroPoint.view(-1, 1)).to(torch.int8);

            zeroPoint = (zeroPoint + 8).to(torch.uint8);
            fourBitWeight = (fourBitWeight + 8).view(-1).to(torch.uint8);

            // torch doesn't provide int4, so we use int8 as placeholder
            // and foreach int8, we save two int4, e.g. 0b1010 -> 0b10, 0b10
            var zpPlaceHolder = zeroPoint[..placeHolderDim];
            zpPlaceHolder = zpPlaceHolder * 16 + zeroPoint[placeHolderDim..];

            // assert zero point is in range [-128, 127]
            //if (torch.any(this.zeroPoint < -128).item<bool>() || torch.any(this.zeroPoint > 127).item<bool>())
            //{
            //    throw new Exception("Zero point is out of range [-128, 127]");
            //}

            // quantize weight
            var fourBitWeightPlaceHolder = fourBitWeight[..fourBitWeightPlaceHolderDim];
            fourBitWeightPlaceHolder = fourBitWeightPlaceHolder * 16 + fourBitWeight[fourBitWeightPlaceHolderDim..];

            // assert weight is in range [-128, 127]
            //if (torch.any(this._8bitWeight < -128).item<bool>() || torch.any(this._8bitWeight > 127).item<bool>())
            //{
            //    throw new Exception("Weight is out of range [-128, 127]");
            //}

            // dispose float32 weight
            this.weight.Dispose();

            this._internal_buffers.Remove("weight");
            this.register_buffer("4bit_weight", fourBitWeightPlaceHolder.MoveToOuterDisposeScope());
            this.register_buffer("zeroPoint", zpPlaceHolder.MoveToOuterDisposeScope());
            this.register_buffer("scale", scale.MoveToOuterDisposeScope());
            timer.Stop();
        }
        else
        {
            // if weight is on meta device, then we just need to create the placeholder for 8bit_weight, zeroPoint and scale
            var fourBitWeight = torch.zeros(fourBitWeightPlaceHolderDim, dtype: torch.int8);
            var zeroPoint = torch.zeros(placeHolderDim, dtype: torch.int8);
            var scale = torch.zeros(this.weight.shape[0], dtype: torch.float32);

            this._internal_buffers.Remove("weight");
            this.weight = null;
            this.register_buffer("4bit_weight", fourBitWeight);
            this.register_buffer("zeroPoint", zeroPoint);
            this.register_buffer("scale", scale);
        }
    }
}
