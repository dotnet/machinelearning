// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using FluentAssertions;
using Microsoft.ML.GenAI.Core.Extension;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core.Tests;

public class QuantizedLinearTests
{
    [Fact]
    public void Int4QuantizeSizeTests()
    {
        // meta is critical for the test
        // as the size of the model to test is 372 GB
        // and can't be loaded in real device like cpu or cuda
        var device = "meta";
        var model = new QuantizedLinear(100000, 100, device: device);

        var sizeInBytes = model.GetSizeInBytes();

        var sizeInGigaBytes = sizeInBytes / 1024 / 1024;
        sizeInGigaBytes.Should().Be(38);

        // to int4
        model.Int4();
        var sizeInBytesAfterInt8 = model.GetSizeInBytes();
        var sizeInGigaBytesAfterInt8 = sizeInBytesAfterInt8 / 1024 / 1024;
        sizeInGigaBytesAfterInt8.Should().Be(4); // 38 // 8 = 4
    }

    [Fact]
    public void Int8QuantizeSizeTests()
    {
        // meta is critical for the test
        // as the size of the model to test is 372 GB
        // and can't be loaded in real device like cpu or cuda
        var device = "meta";
        var model = new QuantizedLinear(100000, 100, device: device);

        var sizeInBytes = model.GetSizeInBytes();

        var sizeInGigaBytes = sizeInBytes / 1024 / 1024;
        sizeInGigaBytes.Should().Be(38);

        // to int8
        model.Int8();
        var sizeInBytesAfterInt8 = model.GetSizeInBytes();
        var sizeInGigaBytesAfterInt8 = sizeInBytesAfterInt8 / 1024 / 1024;
        sizeInGigaBytesAfterInt8.Should().Be(9); // 38 // 4 = 9
    }

    [Fact]
    public void Int4QuantizeForwardTest()
    {
        var device = "cpu";
        var model = new QuantizedLinear(123, 10, device: device);

        // set both weight and bias to rand int8 values
        // and compare the result before and after ToInt8
        var input = torch.ones([10, 2200, 123], device: device);
        var weight = torch.ones([10, 123], device: device, dtype: ScalarType.Int64) * -1;
        var bias = torch.ones([10], device: device) * 2;

        var weightStr = weight.Peek("weight").ToString();

        weight = (weight + 8).view(-1).to(torch.uint8);
        var weightPlaceHolderDim = (int)weight.size(0);
        weightPlaceHolderDim = weightPlaceHolderDim / 2 + weightPlaceHolderDim % 2;
        var weightPlaceHolder = weight[..weightPlaceHolderDim];
        weightPlaceHolder = weightPlaceHolder * 16 + weight[weightPlaceHolderDim..];

        var high4Bit = weightPlaceHolder / 16;
        var low4Bit = weightPlaceHolder % 16;
        weight = torch.cat(new Tensor[] { high4Bit, low4Bit }).view(10, 123);
        weight = weight.to(torch.int64);
        weight -= 8;
        weight.Peek("weight").Should().Be(weightStr);

        model.load_state_dict(new Dictionary<string, Tensor>
        {
            ["weight"] = weight,
            ["bias"] = bias
        });

        var resultBeforeInt4 = model.forward(input);

        model.ToInt4QuantizeModule();

        var resultAfterInt4 = model.forward(input);

        // compare the result
        resultBeforeInt4.Peek("result").Should().Be(resultAfterInt4.Peek("result"));
    }

    [Fact]
    public void Int8QuantizeForwardTest()
    {
        var device = "cpu";
        var model = new QuantizedLinear(123, 10, device: device);

        // set both weight and bias to rand int8 values
        // and compare the result before and after ToInt8
        var input = torch.ones([10, 2200, 123], device: device);
        var weight = torch.ones([10, 123], device: device) * -1;
        var bias = torch.ones([10], device: device) * 2;

        model.load_state_dict(new Dictionary<string, Tensor>
        {
            ["weight"] = weight,
            ["bias"] = bias
        });

        var resultBeforeInt8 = model.forward(input);

        model.ToInt8QuantizeModule();

        var resultAfterInt8 = model.forward(input);

        resultBeforeInt8.Peek("result").Should().Be("result: sum: 312.6933  dType: Float32 shape: [10,2200,10]");
        resultAfterInt8.Peek("result").Should().Be("result: sum: 312.6933  dType: Float32 shape: [10,2200,10]");
    }
}
