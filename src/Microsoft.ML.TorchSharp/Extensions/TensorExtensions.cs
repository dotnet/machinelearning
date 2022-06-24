// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.Extensions
{
    internal static class TensorExtensions
    {

#nullable enable
        public static bool IsNull(this torch.Tensor? tensor)
        {
            return tensor is null || tensor.IsInvalid;
        }

        public static bool IsNotNull(this torch.Tensor? tensor)
        {
            return !tensor.IsNull();
        }
#nullable disable

        public static T[] ToArray<T>(this torch.Tensor tensor) where T : unmanaged
        {
            if (tensor.IsNull())
            {
                return Array.Empty<T>();
            }

            using var cpu = tensor.cpu();
            return cpu.data<T>().ToArray();
        }
    }
}
