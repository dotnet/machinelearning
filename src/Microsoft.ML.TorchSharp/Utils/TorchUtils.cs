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
                return;

            foreach (var kvp in dictionary)
            {
                if (kvp.Value is torch.Tensor tensor)
                    tensor.Dispose();

                else if (kvp.Value is Dictionary<dynamic, dynamic> subDictionary)
                    DisposeDictionaryWithTensor(subDictionary);

                if (kvp.Key is torch.Tensor keyTensor)
                    keyTensor.Dispose();

                else if (kvp.Key is Dictionary<dynamic, dynamic> subDictionary)
                    DisposeDictionaryWithTensor(subDictionary);
            }
        }
    }
}
