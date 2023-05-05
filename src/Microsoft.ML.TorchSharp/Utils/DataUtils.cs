// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.Utils
{
    internal class DataUtils
    {
        public static torch.Tensor CollateTokens(IList<torch.Tensor> values, int padIndex, int? eosIndex = null,
            bool leftPad = false, bool moveEosToBeginning = false, torch.Device device = null)
        {
            Contracts.AssertNonEmpty(values, "Can't collate 0 values");
            Contracts.Assert(values.All(v => v.dim() == 1), "All tensors should be 1D to collate.");

            var size = values.Select(v => v.size(0)).Max();
            var res = values[0].new_full(values.Count, size, padIndex, device: device);

            for (var i = 0; i < values.Count; ++i)
            {
                var v = values[i];
                CopyTensor(
                    v,
                    leftPad
                        ? res[torch.TensorIndex.Single(i), torch.TensorIndex.Slice(start: size - v.size(0))]
                        : res[torch.TensorIndex.Single(i), torch.TensorIndex.Slice(stop: v.size(0))],
                    moveEosToBeginning,
                    eosIndex);
            }

            return res;
        }

        /// <summary>
        /// Copy <paramref name="src"/> tensor to <paramref name="dst"/> tensor.
        /// If <paramref name="moveEosToBeginning"/> is true, an EOS token will be added to the beginning
        /// of <paramref name="dst"/> tensor, and the last token of <paramref name="src"/> will be dropped.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dst"></param>
        /// <param name="moveEosToBeginning"></param>
        /// <param name="eosIndex"></param>
        /// <exception cref="ArgumentException"></exception>
        private static void CopyTensor(torch.Tensor src, torch.Tensor dst,
            bool moveEosToBeginning = false, int? eosIndex = null)
        {
            if (src.numel() != dst.numel())
            {
                throw new ArgumentException(
                    $"Inconsistent capacity when copying tensor, got {src.numel()} and {dst.numel()}.");
            }

            if (moveEosToBeginning && (eosIndex == null || eosIndex < 0))
            {
                throw new ArgumentException(
                    $"{nameof(eosIndex)} must not be null or negative when {nameof(moveEosToBeginning)} is true.");
            }

            if (moveEosToBeginning && src[-1][0].ToInt32() == eosIndex)
            {
                dst[0] = torch.tensor((int)eosIndex);
                dst[torch.TensorIndex.Slice(start: 1)] = src[torch.TensorIndex.Slice(stop: -1)];
            }
            else
            {
                dst.copy_(src);
            }
        }
    }
}
