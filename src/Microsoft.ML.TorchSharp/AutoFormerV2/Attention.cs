// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    /// <summary>
    /// The Attention layer.
    /// </summary>
    public class Attention : Module<Tensor, Tensor, Tensor>
    {
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly int numHeads;
        private readonly double scale;
        private readonly int keyChannels;
        private readonly int nHkD;
        private readonly int d;
        private readonly int dh;
        private readonly double attnRatio;

        private readonly LayerNorm norm;
        private readonly Linear qkv;
        private readonly Linear proj;
        private readonly Parameter attention_biases;
        private readonly TensorIndex attention_bias_idxs;
        private readonly Softmax softmax;
        private bool _disposedValue;
#pragma warning restore MSML_PrivateFieldName


        /// <summary>
        /// Initializes a new instance of the <see cref="Attention"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="keyChannels">The key channels.</param>
        /// <param name="numHeads">The number of blocks.</param>
        /// <param name="attnRatio">The ratio of attention.</param>
        /// <param name="windowResolution">The resolution of window.</param>
        public Attention(int inChannels, int keyChannels, int numHeads = 8, int attnRatio = 4, List<int> windowResolution = null)
            : base(nameof(Attention))
        {
            windowResolution ??= new List<int>() { 14, 14 };
            this.numHeads = numHeads;
            this.scale = System.Math.Pow(keyChannels, -0.5);
            this.keyChannels = keyChannels;
            this.nHkD = numHeads * keyChannels;
            this.d = attnRatio * keyChannels;
            this.dh = this.d * numHeads;
            this.attnRatio = attnRatio;
            int h = this.dh + (this.nHkD * 2);

            this.norm = nn.LayerNorm(new long[] { inChannels });
            this.qkv = nn.Linear(inChannels, h);
            this.proj = nn.Linear(this.dh, inChannels);

            var points = new List<List<int>>();
            for (int i = 0; i < windowResolution[0]; i++)
            {
                for (int j = 0; j < windowResolution[1]; j++)
                {
                    points.Add(new List<int>() { i, j });
                }
            }

            int n = points.Count;
            var attentionOffsets = new Dictionary<Tuple<int, int>, int>();
            var idxs = new List<int>();
            var idxsTensor = torch.zeros(new long[] { n, n }, dtype: torch.int64);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var offset = new Tuple<int, int>(Math.Abs(points[i][0] - points[j][0]), Math.Abs(points[i][1] - points[j][1]));
                    if (!attentionOffsets.ContainsKey(offset))
                    {
                        attentionOffsets.Add(offset, attentionOffsets.Count);
                    }

                    idxs.Add(attentionOffsets[offset]);
                    idxsTensor[i][j] = attentionOffsets[offset];
                }
            }

            this.attention_biases = nn.Parameter(torch.zeros(numHeads, attentionOffsets.Count));
            this.attention_bias_idxs = TensorIndex.Tensor(idxsTensor);
            this.softmax = nn.Softmax(dim: -1);
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor x, Tensor mask)
        {
            using (var scope = torch.NewDisposeScope())
            {
                long b = x.shape[0];
                long n = x.shape[1];
                long c = x.shape[2];
                x = this.norm.forward(x);
                var qkv = this.qkv.forward(x);
                qkv = qkv.view(b, n, this.numHeads, -1);
                var tmp = qkv.split(new long[] { this.keyChannels, this.keyChannels, this.d }, dim: 3);
                var q = tmp[0];
                var k = tmp[1];
                var v = tmp[2];
                q = q.permute(0, 2, 1, 3);
                k = k.permute(0, 2, 1, 3);
                v = v.permute(0, 2, 1, 3);

                var attn = (torch.matmul(q, k.transpose(-2, -1)) * this.scale) + this.attention_biases[RangeUtil.ToTensorIndex(..), this.attention_bias_idxs];
                if (!(mask is null))
                {
                    long nW = mask.shape[0];
                    attn = attn.view(-1, nW, this.numHeads, n, n) + mask.unsqueeze(1).unsqueeze(0);
                    attn = attn.view(-1, this.numHeads, n, n);
                    attn = this.softmax.forward(attn);
                }
                else
                {
                    attn = this.softmax.forward(attn);
                }

                x = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, this.dh);
                x = this.proj.forward(x);

                return x.MoveToOuterDisposeScope();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    norm.Dispose();
                    qkv.Dispose();
                    proj.Dispose();
                    attention_biases.Dispose();
                    softmax.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
