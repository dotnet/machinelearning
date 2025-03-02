// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.GenAI.Core;
using TorchSharp;
using TorchSharp.BitsAndBytes;
using TorchSharp.Modules;
using static TorchSharp.torch;
namespace Microsoft.ML.GenAI.Core;

internal class QuantizedLinear : GenAILinear, IQuantizeModule
{
    private Tensor? _quantizedTensor = null;
    private Tensor? _absMax = null;
    private int _blockSize;
    private int _n;
    private string? _quantizedDType = null;
    private readonly long[] _weightShape;

    public QuantizedLinear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(inFeatures, outFeatures, hasBias, dtype, device)
    {
        _weightShape = [outFeatures, inFeatures];
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
        var inputShape = input.shape;
        if (this._internal_buffers.ContainsKey("weight"))
        {
            return base.forward(input);
        }
        else if (this._internal_buffers.ContainsKey("8bit_weight"))
        {
            // 8bit quantization
            using var dispose = torch.NewDisposeScope();
            var weight = this.get_buffer("8bit_weight")!.to(ScalarType.Float32);
            var zeroPoint = this.get_buffer("zeroPoint")!.to(ScalarType.Float32);
            var scale = this.get_buffer("scale")!.to(ScalarType.Float32);
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
        else if ((_quantizedDType == "fp4" || _quantizedDType == "nf4") && _quantizedTensor is not null && _absMax is not null)
        {
            using var dispose = torch.NewDisposeScope();
            if (input.shape.Length >= 3 && input.shape[1] != 1)
            {
                // dequantize quantizedWeight to float32 and use torch.matmul
                var dequantizedWeight = BitsAndByteUtils.Dequantize4Bit(
                    tensor: this._quantizedTensor,
                    originalDType: input.dtype,
                    originalShape: this._weightShape,
                    blockSize: _blockSize,
                    n: this._n,
                    absMax: this._absMax!,
                    quantizedDType: _quantizedDType);

                var output = torch.matmul(input, dequantizedWeight.T);

                if (this.bias is not null)
                {
                    output = output.add_(this.bias.to_type(output.dtype));
                }

                return output.MoveToOuterDisposeScope();
            }
            else
            {
                var output = BitsAndByteUtils.Gemv4Bit(
                input: input,
                quantizedWeight: this._quantizedTensor,
                originalWeightShape: _weightShape,
                absMax: this._absMax!,
                quantizedDType: _quantizedDType,
                blockSize: _blockSize);

                if (this.bias is not null)
                {
                    output = output.add_(this.bias.to_type(output.dtype));
                }

                return output.MoveToOuterDisposeScope();
            }
        }
        else
        {
            throw new Exception("Quantization is not done yet");
        }
    }

    public void FP4()
    {
        if (this.weight is null)
        {
            throw new Exception("Weight is not initialized");
        }

        if (this.weight.device_type == DeviceType.META)
        {
            return;
        }
        using var dispose = torch.NewDisposeScope();

        _quantizedDType = "fp4"; // Available options: "fp4", "nf4"
        _blockSize = 64; // can be [64, 128, 256, 512, 1024]

        // Quantize to 4Bit
        (_quantizedTensor, _absMax, _blockSize, _n) = BitsAndByteUtils.Quantize4Bit(this.weight.cuda(), _quantizedDType, _blockSize);

        this.weight.Dispose();
        this.weight = null;
        this._internal_buffers.Remove("weight");
        _quantizedTensor.MoveToOuterDisposeScope();
        _absMax.MoveToOuterDisposeScope();
    }
}
