// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.TorchSharp.NasBert.Models;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.Utils
{
    internal static class ModelUtils
    {
        public static void InitXavierUniform(torch.Tensor tensor, double gain = 1)
        {
            using var xavier = torch.nn.init.xavier_uniform_(tensor, gain);
        }

        public static void InitConstant(torch.Tensor tensor, Scalar val)
        {
            using var cons = torch.nn.init.constant_(tensor, val);
        }

        public static void InitNormal(torch.Tensor tensor, double mean = 0, double std = 1)
        {
            using var norm = torch.nn.init.normal_(tensor, mean, std);
        }

        public static void InitZeros(torch.Tensor tensor)
        {
            using var zeros = torch.nn.init.zeros_(tensor);
        }

        public static void FreezeModuleParams(ModuleList<torch.nn.Module> modules)
        {
            foreach (var module in modules)
            {
                FreezeModuleParams(module);
            }
        }

        public static void FreezeModuleParams(torch.nn.Module module)
        {
            if (module is null) return;
            foreach (var param in module.parameters())
            {
                param.requires_grad = false;
            }
        }
    }
}
