// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.TorchSharp.NasBert.Models;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.Utils
{
    internal static class ModelUtils
    {
        private static readonly DefaultDictionary<string, int> _incrementalStateInstanceId =
            new DefaultDictionary<string, int>(() => 0);

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

        /// <summary>
        /// Helper for getting incremental state for a torch.nn.Module.
        /// </summary>
        public static Dictionary<string, torch.Tensor> GetIncrementalState(
            BaseModule module,
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState,
            string key)
        {
            var fullKey = GetFullIncrementalStateKey(module, key);
            return incrementalState.ContainsKey(fullKey) ? incrementalState[fullKey] : null;
        }

        /// <summary>
        /// Helper for setting incremental state for a torch.nn.Module.
        /// </summary>
        public static void SetIncrementalState(
            BaseModule module,
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState,
            string key,
            Dictionary<string, torch.Tensor> value)
        {
            var fullKey = GetFullIncrementalStateKey(module, key);
            if (incrementalState?.TryGetValue(fullKey, out var oldState) == true)
            {
                TorchUtils.DisposeDictionaryWithTensor(oldState);
            }
            incrementalState[fullKey] = value;
        }

        public static string GetFullIncrementalStateKey(BaseModule module, string key)
        {
            var moduleName = module.GetName();

            // Assign a unique ID to each module instance, so that incremental state is not shared across module instances.
            if (module.InstanceId == null)
            {
                _incrementalStateInstanceId[moduleName] += 1;
                module.InstanceId = _incrementalStateInstanceId[moduleName];
            }

            return $"{moduleName}.{module.InstanceId}.{key}";
        }

        //public static void FreezeModuleParams(ModuleList modules)
        //{
        //    foreach (var module in modules)
        //    {
        //        FreezeModuleParams(module);
        //    }
        //}

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
