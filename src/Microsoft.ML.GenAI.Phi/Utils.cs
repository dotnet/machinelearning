// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace Microsoft.ML.GenAI.Phi;

internal static class Utils
{
    public static string GetEmbeddedResource(string resourceName)
    {
        // read file content from embedded resource
        var assembly = Assembly.GetExecutingAssembly();
        var resourceStream = assembly.GetManifestResourceStream(resourceName);

        if (resourceStream == null)
        {
            throw new ArgumentException("Resource not found", nameof(resourceName));
        }

        using var reader = new System.IO.StreamReader(resourceStream);
        return reader.ReadToEnd();
    }

    public static Tensor ApplyRotaryEmbeddings(Tensor input, Tensor freqsComplex)
    {
        // Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
        // Two consecutive values will become a single complex number
        // (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
        var inputComplex = input.to_type(ScalarType.Float32).reshape(input.shape[0], input.shape[1], input.shape[2], -1, 2).view_as_complex();
        freqsComplex = freqsComplex.to(input.device);

        // Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
        // (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
        var freqsComplexReshaped = freqsComplex.unsqueeze(0).unsqueeze(2);

        // Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
        // Which results in the rotation of the complex number as shown in the Figure 1 of the paper
        // (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
        var rotatedComplex = inputComplex * freqsComplexReshaped;
        // Console.WriteLine(rotated_complex.mean().ToSingle());

        // Convert the complex number back to the real number
        // (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
        var rotated = rotatedComplex.view_as_real();

        // (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
        var rotatedReshaped = rotated.reshape(rotated.shape[0], rotated.shape[1], rotated.shape[2], -1);

        return rotatedReshaped.type_as(input);
    }

    public static Tensor PrecomputeThetaPosFrequencies(int headDim, int seqLen, string device, float theta = 10000.0f)
    {
        // As written in the paragraph 3.2.2 of the paper
        // >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
        if (headDim % 2 != 0)
        {
            throw new ArgumentException("Dimension must be divisible by 2", nameof(headDim));
        }

        // Build the theta parameter
        // According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        // Shape: (Head_Dim / 2)
        var thetaNumerator = torch.arange(0, headDim, 2).to(torch.float32).to(device);
        // Shape: (Head_Dim / 2)
        var thetaInput = torch.pow(theta, -1.0f * (thetaNumerator / headDim)).to(device); // (Dim / 2)
        // Construct the positions (the "m" parameter)
        // Shape: (Seq_Len)
        var m = torch.arange(seqLen, device: device);
        // Multiply each theta by each position using the outer product.
        // Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqs = torch.outer(m, thetaInput).to(torch.float32).to(device);

        // We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        // (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqsComplex = torch.polar(torch.ones_like(freqs), freqs);

        return freqsComplex;
    }

    // python
    // def rotate_half(x):
    // """Rotates half the hidden dims of the input."""
    // x1 = x[..., : x.shape[-1] // 2]
    // x2 = x[..., x.shape[-1] // 2 :]
    // return torch.cat((-x2, x1), dim=-1)
    public static Tensor RotateHalf(Tensor x)
    {
        var x1 = x[.., .., .., ..(int)(x.shape[^1] / 2)];
        var x2 = x[.., .., .., (int)(x.shape[^1] / 2)..];
        // (x1 * x1 * x2).Peek("x1 * x1 * x2");
        return torch.cat([-x2, x1], dim: -1);
    }

    public static (Tensor, Tensor) ApplyRotaryPosEmb(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? positionIds = null, int unsqueezeDim = 1)
    {
        // The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
        // sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
        // that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
        // k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
        // cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
        // the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

        if (positionIds is not null)
        {
            cos = cos[positionIds!].unsqueeze(unsqueezeDim);
            sin = sin[positionIds!].unsqueeze(unsqueezeDim);
        }
        else
        {
            cos = cos.unsqueeze(unsqueezeDim);
            sin = sin.unsqueeze(unsqueezeDim);
        }
        var qEmbed = q * cos;
        qEmbed += RotateHalf(q) * sin;

        var kEmbed = k * cos;
        kEmbed += RotateHalf(k) * sin;
        // var kEmbed = (k * cos) + (RotateHalf(k) * sin);
        return (qEmbed, kEmbed);
    }



    public static Tensor Phi2RepeatKV(Tensor x, int nRep)
    {
        var batchSize = x.shape[0];
        var seqLen = x.shape[1];
        var nKVHeads = x.shape[2];
        var headDim = x.shape[3];
        if (nRep == 1)
        {
            return x;
        }

        return x.unsqueeze(3)
                .expand(batchSize, seqLen, nKVHeads, nRep, headDim)
                .view(batchSize, seqLen, nKVHeads * nRep, headDim);
    }

    public static Tensor Phi3RepeatKV(Tensor x, int nRep)
    {
        var batchSize = x.shape[0];
        var nKVHeads = x.shape[1];
        var seqLen = x.shape[2];
        var headDim = x.shape[3];
        if (nRep == 1)
        {
            return x;
        }

        return x.unsqueeze(3)
                .expand(batchSize, nKVHeads, nRep, seqLen, headDim)
                .view(batchSize, nKVHeads * nRep, seqLen, headDim);
    }

}
