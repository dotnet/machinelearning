// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public interface IKVCache : IDictionary<int, (Tensor, Tensor)>, IDisposable
{
    public (Tensor, Tensor) UpdateKVCache(Tensor key, Tensor value, int layerIndex);

    public int GetSeqLen(int layerIndex = 0);

    public int? GetMaxLength();

    public int GetUsableLength(int newSeqLen, int layerIndex = 0);
}

public class DynamicKVCache : Dictionary<int, (Tensor, Tensor)>, IKVCache
{
    private readonly DisposeScope _disposeScope = NewDisposeScope();
    public DynamicKVCache()
    {
    }

    public (Tensor, Tensor) UpdateKVCache(Tensor key, Tensor value, int layerIndex)
    {
        if (this.ContainsKey(layerIndex))
        {
            var (oldKey, oldValue) = this[layerIndex];
            oldKey.DetachFromDisposeScope();
            oldValue.DetachFromDisposeScope();

            var newKey = torch.cat([oldKey, key], -2).MoveToOtherDisposeScope(this._disposeScope);
            var newValue = torch.cat([oldValue, value], -2).MoveToOtherDisposeScope(this._disposeScope);

            oldKey.Dispose();
            oldValue.Dispose();

            this[layerIndex] = (newKey, newValue);
        }
        else
        {
            this.Add(layerIndex, (key.MoveToOtherDisposeScope(this._disposeScope), value.MoveToOtherDisposeScope(this._disposeScope)));
        }

        return this[layerIndex];
    }

    public int GetSeqLen(int layerIndex = 0)
    {
        if (this.TryGetValue(layerIndex, out var kv))
        {
            return kv.Item1.IntShape()[^2];
        }

        return 0;
    }

    public int? GetMaxLength()
    {
        return null;
    }

    public int GetUsableLength(int newSeqLen, int layerIndex = 0)
    {
        var maxLength = this.GetMaxLength();
        var previousSeqLen = this.GetSeqLen(layerIndex);

        if (maxLength.HasValue && previousSeqLen + newSeqLen > maxLength.Value)
        {
            return maxLength.Value - previousSeqLen;
        }

        return previousSeqLen;
    }

    public void Dispose()
    {
        this._disposeScope.Dispose();
    }
}
