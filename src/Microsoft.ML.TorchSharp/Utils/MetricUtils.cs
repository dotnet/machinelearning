// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.TorchSharp.Extensions;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.Utils
{
    internal class MetricUtils
    {
        /// <summary>
        /// Get top k answer spans for QA task.
        /// </summary>
        /// <param name="logits">Model output logits</param>
        /// <param name="k">Number of highest scores</param>
        /// <param name="input0Len"></param>
        /// <param name="input1Len"></param>
        public static IList<(int start, int end, float score)> ComputeTopKSpansWithScore(torch.Tensor logits, int k, int input0Len, int input1Len)
        {
            var splitLogits = logits.split(1, dim: -1);
            var startLogits = splitLogits[0].squeeze(-1).contiguous();  //[maxseqlen]
            var endLogits = splitLogits[1].squeeze(-1).contiguous();  //[maxseqlen]

            var (predictStartScores, predictStarts) = startLogits.topk(k);
            var (predictEndScores, predictEnds) = endLogits.topk(k);

            var startScores = predictStartScores.ToArray<float>();
            var endScores = predictEndScores.ToArray<float>();
            var starts = predictStarts.ToArray<long>();
            var ends = predictEnds.ToArray<long>();
            var topK = new List<(int start, int end, float score)>();
            for (var i = 0; i < starts.Length; ++i)
            {
                for (var j = 0; j < ends.Length; ++j)
                {
                    if (starts[i] <= input0Len + 1 || starts[i] >= input0Len + input1Len + 2 || ends[j] <= input0Len + 1 || ends[j] >= input0Len + input1Len + 2 || starts[i] > ends[j])
                    {
                        continue;
                    }
                    topK.Add(((int)starts[i], (int)ends[j], startScores[i] + endScores[j]));
                }
            }
            topK = topK.OrderByDescending(tuple => tuple.score).Take(k).ToList();
            return topK;
        }
    }
}
