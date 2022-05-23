// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.Utils
{
    internal static class TorchUtils
    {
        public static void DisposeDictionaryWithTensor<TKey, TResult>(Dictionary<TKey, TResult> dictionary)
        {
            if (dictionary == null)
            {
                return;
            }

            foreach (var value in dictionary.Values)
            {
                if (value is torch.Tensor tensor)
                {
                    tensor.Dispose();
                }
                else if (value is Dictionary<dynamic, dynamic> subDictionary)
                {
                    DisposeDictionaryWithTensor(subDictionary);
                }
            }

            foreach (var key in dictionary.Keys)
            {
                if (key is torch.Tensor tensor)
                {
                    tensor.Dispose();
                }
                else if (key is Dictionary<dynamic, dynamic> subDictionary)
                {
                    DisposeDictionaryWithTensor(subDictionary);
                }
            }
        }
    }
}
