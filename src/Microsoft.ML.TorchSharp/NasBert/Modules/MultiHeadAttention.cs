// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal sealed class MultiHeadAttention : BaseModule
    {
        private const string PrevKeyKey = "prevKey";
        private const string PrevValueKey = "prevValue";
        private const string AttentionStateKey = "attentionState";

        private readonly int _embeddingDim;
        private readonly int _kDim;
        private readonly int _vDim;
        private readonly bool _qkvSameDim;
        private readonly bool _addBiasProj;
        private readonly bool _addBiasKv;

        private readonly int _numHeads;
        private readonly double _dropout;
        private readonly int _headDim;
        private readonly double _scaling;

        private readonly bool _selfAttention;
        private readonly bool _encoderDecoderAttention;
        private readonly bool _addZeroAttention;


#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format Has to match TorchSharp.
        private readonly Linear QProjection;
        private readonly Linear KProjection;
        private readonly Linear VProjection;

        private readonly Parameter KBias;
        private readonly Parameter VBias;

        private readonly Linear OutProjLinear;
        private readonly Dropout DropoutLayer;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format


        public MultiHeadAttention(
            int embeddingDim,
            int numHeads,
            int? kDim = null,
            int? vDim = null,
            double dropout = 0.0,
            bool bias = true,
            bool addBiasKv = false,
            bool addZeroAttention = false,
            bool selfAttention = false,
            bool encoderDecoderAttention = false)
            : base(nameof(MultiHeadAttention))
        {
            _embeddingDim = embeddingDim;
            _kDim = kDim ?? embeddingDim;
            _vDim = vDim ?? embeddingDim;
            _qkvSameDim = (_kDim == _embeddingDim) && (_vDim == _embeddingDim);

            _numHeads = numHeads;
            _dropout = dropout;
            _headDim = _embeddingDim / _numHeads;
            _scaling = Math.Pow(_headDim, -0.5);
            if (_headDim * _numHeads != _embeddingDim)
            {
                throw new ArgumentException("EmbeddingDim must be divisible by NumHeads");
            }

            _selfAttention = selfAttention;
            _encoderDecoderAttention = encoderDecoderAttention;
            if (_selfAttention && !_qkvSameDim)
            {
                throw new ArgumentException("Self-attention requires query, key and value to be of the same size");
            }

            _addBiasProj = bias;
            _addBiasKv = addBiasKv;
            _addZeroAttention = addZeroAttention;

            QProjection = torch.nn.Linear(_embeddingDim, _embeddingDim, _addBiasProj);
            KProjection = torch.nn.Linear(_kDim, _embeddingDim, _addBiasProj);
            VProjection = torch.nn.Linear(_vDim, _embeddingDim, _addBiasProj);

            if (_addBiasKv)
            {
                KBias = torch.zeros(1, 1, _embeddingDim).AsParameter();
                VBias = torch.zeros(1, 1, _embeddingDim).AsParameter();
            }

            OutProjLinear = torch.nn.Linear(_embeddingDim, _embeddingDim, _addBiasProj);
            DropoutLayer = torch.nn.Dropout(_dropout);

            Initialize();
            RegisterComponents();
        }

        public void Initialize()
        {
            if (_qkvSameDim)
            {
                ModelUtils.InitXavierUniform(QProjection.weight, 1.0 / Math.Sqrt(2.0));
                ModelUtils.InitXavierUniform(KProjection.weight, 1.0 / Math.Sqrt(2.0));
                ModelUtils.InitXavierUniform(VProjection.weight, 1.0 / Math.Sqrt(2.0));
            }
            else
            {
                ModelUtils.InitXavierUniform(QProjection.weight);
                ModelUtils.InitXavierUniform(KProjection.weight);
                ModelUtils.InitXavierUniform(VProjection.weight);
            }

            ModelUtils.InitXavierUniform(OutProjLinear.weight);

            if (_addBiasProj)
            {
                ModelUtils.InitConstant(QProjection.bias, 0);
                ModelUtils.InitConstant(KProjection.bias, 0);
                ModelUtils.InitConstant(VProjection.bias, 0);
                ModelUtils.InitConstant(OutProjLinear.bias, 0);
            }

            if (_addBiasKv)
            {
                ModelUtils.InitXavierUniform(KBias);
                ModelUtils.InitXavierUniform(VBias);
            }
        }

        /// <summary>
        /// Input shape: seqLen x batch x channel
        /// Time-steps can be masked by supplying a T x T mask in the <paramref name="attentionMask"/> argument.
        /// Padding elements can be excluded from the key by passing a binary ByteTensor(<paramref name="keyPaddingMask"/>)
        ///     with shape: batch x srcLen, where padding elements are indicated by 1s.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public torch.Tensor forward(
            torch.Tensor query,
            torch.Tensor key,
            torch.Tensor value,
            out torch.Tensor outAttentionWeights,
            torch.Tensor keyPaddingMask = null,
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState = null,
            bool needWeights = true,
            bool staticKv = false,
            torch.Tensor attentionMask = null)
        {
            outAttentionWeights = null;

            if (query.IsNull() || query.size().Length != 3 || query.size(2) != _embeddingDim)
            {
                throw new ArgumentException("query must NOT be null and must be 3D in multi-head attention;" +
                                            "the last dimension should be the same as embedding dimension.");
            }

            using var disposeScope = torch.NewDisposeScope();

            var qSize = query.size();
            var tgtLen = qSize[0];
            var batchSize = qSize[1];
            var embedDim = qSize[2];

            // Get saved state from incrementalState
            Dictionary<string, torch.Tensor> savedState = null;
            if (incrementalState != null)
            {
                savedState = GetInputBuffer(incrementalState);

                // previous time steps are cached - no need to recompute key and value if they are static.
                if (savedState.ContainsKey(PrevKeyKey) && savedState.ContainsKey(PrevValueKey) && staticKv)
                {
                    if (_selfAttention || !_encoderDecoderAttention)
                    {
                        throw new ArgumentException(
                            "prevKey and prevValue are only valid in encoder-decoder attention.");
                    }

                    key = value = null;
                }
            }

            // Calculate current qkv projection
            var (q, k, v) = QkvProjection(query, key, value);

            // Simulate using-statement by try-finally
            torch.Tensor attentionMaskPad = attentionMask?.alias();
            torch.Tensor keyPaddingMaskPad = keyPaddingMask?.alias();
            q.mul_(_scaling);

            if (_addBiasKv)
            {
                var kRepeat = KBias.repeat(1, batchSize, 1);
                var vRepeat = VBias.repeat(1, batchSize, 1);
                k = torch.cat(new List<torch.Tensor> { k, kRepeat }, dimension: 0);
                v = torch.cat(new List<torch.Tensor> { v, vRepeat }, dimension: 0);
                attentionMaskPad = PadMask(attentionMaskPad);
                keyPaddingMaskPad = PadMask(keyPaddingMaskPad);
            }

            q = q.view(tgtLen, batchSize * _numHeads, _headDim).transpose_(0, 1);
            k = k?.view(-1, batchSize * _numHeads, _headDim).transpose_(0, 1);
            v = v?.view(-1, batchSize * _numHeads, _headDim).transpose_(0, 1);

            if (savedState != null)
            {
                // saved states are stored with shape (batchSize, NumHeads, seqLen, HeadDim)
                if (savedState.ContainsKey(PrevKeyKey))
                {
                    var prevKey = savedState[PrevKeyKey].view(batchSize * _numHeads, -1, _headDim);
                    k = staticKv
                        ? prevKey
                        : torch.cat(new List<torch.Tensor> { prevKey, k }, dimension: 1);
                }

                if (savedState.ContainsKey(PrevValueKey))
                {
                    var prevValue = savedState[PrevValueKey].view(batchSize * _numHeads, -1, _headDim);
                    v = staticKv
                        ? prevValue
                        : torch.cat(new List<torch.Tensor> { prevValue, v }, dimension: 1);
                }

                savedState[PrevKeyKey].Dispose();
                savedState[PrevKeyKey] = k?.view(batchSize, _numHeads, -1, _headDim);
                savedState[PrevValueKey].Dispose();
                savedState[PrevValueKey] = v?.view(batchSize, _numHeads, -1, _headDim);

                SetInputBuffer(incrementalState, savedState);
            }

            Debug.Assert(k.IsNotNull() && v.IsNotNull());
            var srcLen = k!.size(1);

            // This is part of a workaround to get around fork/join parallelism not supporting Optional types.
            if (keyPaddingMaskPad?.shape.Length == 0) keyPaddingMaskPad = null;
            Debug.Assert(keyPaddingMaskPad.IsNull() ||
                            (keyPaddingMaskPad.size(0) == batchSize && keyPaddingMaskPad.size(1) == srcLen));

            if (_addZeroAttention)
            {
                srcLen += 1;
                var zeroPadSize = k.size();
                zeroPadSize[1] = 1;
                var kZeros = k.new_zeros(zeroPadSize);
                var vZeros = v!.new_zeros(zeroPadSize);
                k = torch.cat(new List<torch.Tensor> { k, kZeros }, dimension: 1);
                v = torch.cat(new List<torch.Tensor> { v, vZeros }, dimension: 1);
                attentionMaskPad = PadMask(attentionMaskPad);
                keyPaddingMaskPad = PadMask(keyPaddingMaskPad);
            }

            var attentionWeights = torch.matmul(q, k.transpose(1, 2));
            Debug.Assert(attentionWeights.size().SequenceEqual(new[] { batchSize * _numHeads, tgtLen, srcLen }));

            if (attentionMaskPad.IsNotNull())
            {
                attentionWeights.add_(attentionMaskPad.unsqueeze(0));
            }

            if (keyPaddingMaskPad.IsNotNull())
            {
                // Don't attend to pad symbols
                keyPaddingMaskPad = keyPaddingMaskPad.unsqueeze(1).unsqueeze(2);

                attentionWeights = attentionWeights
                    .view(batchSize, _numHeads, tgtLen, srcLen)
                    .masked_fill(keyPaddingMaskPad, float.NegativeInfinity)
                    .view(batchSize * _numHeads, tgtLen, srcLen);
            }

            attentionWeights = torch.nn.functional.softmax(attentionWeights, dim: -1);
            attentionWeights = DropoutLayer.forward(attentionWeights);

            if (needWeights)
            {
                // Average attention weights over heads
                var weightsView = attentionWeights.view(batchSize, _numHeads, tgtLen, srcLen);
                outAttentionWeights = weightsView.sum(dim: 1).div_(_numHeads);
            }

            var attention = torch.matmul(attentionWeights, v);
            Debug.Assert(attention.size().SequenceEqual(new[] { batchSize * _numHeads, tgtLen, _headDim }));
            attention = attention.transpose(0, 1).contiguous().view(tgtLen, batchSize, embedDim);
            var attentionOutput = OutProjLinear.forward(attention);

            outAttentionWeights?.MoveToOuterDisposeScope();
            return attentionOutput.MoveToOuterDisposeScope();
        }

        private static torch.Tensor PadMask(torch.Tensor tensor)
        {
            if (tensor.IsNull())
            {
                return null;
            }

            using var zeros = tensor.new_zeros(tensor.size(0), 1);
            return torch.cat(new List<torch.Tensor> { tensor, zeros }, dimension: 1);
        }

        private Dictionary<string, torch.Tensor> GetInputBuffer(
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState)
        {
            return ModelUtils.GetIncrementalState(this, incrementalState, AttentionStateKey) ?? new Dictionary<string, torch.Tensor>();
        }

        private void SetInputBuffer(
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState,
            Dictionary<string, torch.Tensor> buffer)
        {
            ModelUtils.SetIncrementalState(this, incrementalState, AttentionStateKey, buffer);
        }

        private (torch.Tensor, torch.Tensor, torch.Tensor) QkvProjection(
            torch.Tensor query, torch.Tensor key, torch.Tensor value)
        {
            using var disposeScope = torch.NewDisposeScope();

            torch.Tensor q = null;
            torch.Tensor k = null;
            torch.Tensor v = null;
            if (_selfAttention)
            {
                q = QProjection.forward(query);
                k = KProjection.forward(query);
                v = VProjection.forward(query);
            }
            else if (_encoderDecoderAttention)
            {
                q = QProjection.forward(query);
                if (key.IsNull())
                {
                    k = v = null;
                }
                else
                {
                    k = KProjection.forward(key);
                    v = VProjection.forward(key);
                }
            }
            else
            {
                q = QProjection.forward(query);
                k = KProjection.forward(key);
                v = VProjection.forward(value);
            }

            return (q.MoveToOuterDisposeScope(), k.MoveToOuterDisposeScope(), v.MoveToOuterDisposeScope());
        }
    }
}
