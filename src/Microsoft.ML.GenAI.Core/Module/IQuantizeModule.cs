// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.GenAI.Core;

public interface IQuantizeModule
{
    public void Int8();

    /// <summary>
    /// Quantize using BitsAndBytes.FP4
    /// </summary>
    /// <param name="config"><see cref="Quantize4BitConfig"/></param>
    public void Quantize4Bit(Quantize4BitConfig config);
}

/// <summary>
/// Quantize configuration for 4-bit quantization.
/// </summary>
public record Quantize4BitConfig
{
    public Quantize4BitConfig(string quantizedDType = "fp4", int blockSize = 64)
    {
        QuantizedDType = quantizedDType;
        BlockSize = blockSize;
    }

    /// <summary>
    /// Quantized data type, can be "fp4" or "nf4".
    /// </summary>
    public string QuantizedDType { get; init; }

    /// <summary>
    /// Block size for quantization, can be [64, 128, 256, 512, 1024]. The larger the size, the faster the speed and the lower the precision.
    /// </summary>
    public int BlockSize { get; init; }
}
