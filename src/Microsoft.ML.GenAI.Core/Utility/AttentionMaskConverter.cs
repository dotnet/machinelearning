// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.GenAI.Core;

public class AttentionMaskConverter
{
    private readonly bool _isCausal;
    private readonly int? _slidingWindow;

    public AttentionMaskConverter(bool isCausal, int? slidingWindow)
    {
        this._isCausal = isCausal;
        this._slidingWindow = slidingWindow;
    }

    /// <summary>
    /// Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
    /// key_value_length) shape and by adding a large negative bias to not-attended positions.If attention_mask is
    /// causal, a causal mask will be added.
    /// </summary>
    /// <param name="attentionMask2d"></param>
    /// <param name="queryLength"></param>
    /// <param name="dType"></param>
    /// <param name="keyValueLength"></param>
    /// <returns></returns>
    public Tensor To4D(
        Tensor attentionMask2d,
        int queryLength,
        ScalarType dType,
        int? keyValueLength = null)
    {
        long[] inputShape = [attentionMask2d.shape[0], queryLength];

        // create causal mask
        // [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        Tensor? causal4dMask = null;
        if ((inputShape[^1] > 1 || this._slidingWindow is not null) && this._isCausal)
        {
            if (keyValueLength is null)
            {
                throw new ArgumentException("key_value_length should be provided when attention_mask is causal");
            }

            var pastKeyValuesLength = keyValueLength.Value - queryLength;
            causal4dMask = MakeCausalMask(inputShape, dType, attentionMask2d.device, pastKeyValuesLength, this._slidingWindow);
        }
        else if (this._slidingWindow is not null)
        {
            throw new NotImplementedException("Sliding window is not supported for non-causal masks");
        }

        var expandedAttnMask = ExpandMask(attentionMask2d, dType, queryLength).to(attentionMask2d.device);
        if (causal4dMask is not null)
        {
            var min = torch.finfo(dType).min;
            expandedAttnMask = causal4dMask.masked_fill(expandedAttnMask.to(ScalarType.Bool), min);
        }

        return expandedAttnMask;
    }

    public Tensor? ToCausal4D(
        int batchSize,
        int queryLength,
        int keyValueLength,
        ScalarType dType,
        Device device)
    {
        if (!_isCausal)
        {
            throw new ArgumentException("This is not a causal mask");
        }

        long[] inputShape = [batchSize, queryLength];
        var pastKeyValueLength = keyValueLength - queryLength;

        // create causal mask
        // [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        Tensor? causal4DMask = null;
        if (queryLength > 1 || this._slidingWindow is int)
        {
            causal4DMask = MakeCausalMask(inputShape, dType, device, pastKeyValueLength, this._slidingWindow);
        }

        return causal4DMask;
    }

    public static Tensor MakeCausalMask(
        long[] inputIdsShape,
        ScalarType dType,
        Device device,
        int pastKeyValuesLength = 0,
        int? slidingWindow = null)
    {
        // Make causal mask used for bi-directional self-attention.
        var bsz = inputIdsShape[0];
        var tgtLen = inputIdsShape[1];
        var min = torch.finfo(dType).min;
        var mask = torch.full([tgtLen, tgtLen], min, dtype: dType, device: device);
        var maskCondition = torch.arange(tgtLen, device: device);
        mask.masked_fill_(maskCondition < (maskCondition + 1).view(tgtLen, 1), 0);
        mask = mask.to(dType);


        if (pastKeyValuesLength > 0)
        {
            mask = torch.cat([torch.zeros([tgtLen, pastKeyValuesLength], dtype: dType, device: device), mask], dim: -1);
        }

        if (slidingWindow is int window)
        {
            var diagonal = pastKeyValuesLength - window - 1;
            var contextMask = torch.tril(torch.ones([tgtLen, tgtLen], dtype: ScalarType.Bool, device: device), diagonal: diagonal);
            mask = mask.masked_fill(contextMask, min);
        }

        // return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, tgtLen, tgtLen + pastKeyValuesLength);
    }

    /// <summary>
    /// Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`
    /// </summary>
    /// <param name="attentionMask">The attention mask should be 2D.</param>
    /// <param name="device">The device to place the mask tensor.</param>
    /// <param name="dType">The data type of the mask tensor.</param>
    /// <param name="pastKeyValuesLength">The length of past key values in cache.</param>
    /// <param name="slidingWindow">The sliding window size.</param>
    /// <param name="inputShape">The input shape should be a tuple that defines `(batch_size, query_length)`.</param>
    public static Tensor? Create4DCausalAttentionMask(
        Tensor? attentionMask,
        long[] inputShape,
        ScalarType dType,
        Device device,
        int pastKeyValuesLength = 0,
        int? slidingWindow = null)
    {
        var converter = new AttentionMaskConverter(isCausal: true, slidingWindow: slidingWindow);
        var batchSize = (int)inputShape[0];
        var queryLength = (int)inputShape[1];
        var keyValueLength = pastKeyValuesLength + queryLength;
        if (attentionMask is not null)
        {
            if (attentionMask.ndim != 2)
            {
                throw new ArgumentException("Attention mask should be 2D");
            }
            return converter.To4D(attentionMask, (int)inputShape[1], dType, keyValueLength);
        }

        return converter.ToCausal4D(batchSize, queryLength, keyValueLength, dType, device);
    }

    public static Tensor ExpandMask(
        Tensor mask,
        ScalarType dType,
        int? tgtLen = null)
    {
        var bsz = (int)mask.shape[0];
        var srcLen = (int)mask.shape[1];
        tgtLen ??= srcLen;

        var expandedMask = mask.unsqueeze(1).unsqueeze(1).expand(bsz, 1, tgtLen.Value, srcLen).to(dType);
        var invertedMask = 1.0 - expandedMask;
        var min = torch.finfo(dType).min;

        return invertedMask.masked_fill(invertedMask.to(ScalarType.Bool), min);
    }
}
