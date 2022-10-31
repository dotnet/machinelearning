// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TorchSharp.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    /// <summary>
    /// Incremental state for incremental generation.
    /// Refer to https://github.com/facebookresearch/fairseq/blob/main/fairseq/incremental_decoding_utils.py.
    /// </summary>
    public interface IIncrementalState
    {
        public void InitIncrementalState();

        public Dictionary<string, torch.Tensor> GetIncrementalState(
            torch.nn.Module module,
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState,
            string key);

        public void SetIncrementalState(
            torch.nn.Module module,
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState,
            string key,
            Dictionary<string, torch.Tensor> value);
    }

    /// <summary>
    /// Incremental state for incremental generation.
    /// Refer to https://github.com/facebookresearch/fairseq/blob/main/fairseq/incremental_decoding_utils.py.
    /// </summary>
    public class IncrementalState : IIncrementalState
    {
        /// <summary>
        /// To separate different modules sharing the same name.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:This name should be CamelCased", Justification = "Need to match TorchSharp.")]
        private static int _global_incremental_state_id;

        /// <summary>
        /// To separate different modules sharing the same name.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:This name should be CamelCased", Justification = "Need to match TorchSharp.")]
        private int _incremental_state_id;

        private static Dictionary<string, torch.Tensor> EmptyIncrementalState => new Dictionary<string, torch.Tensor>();

        public IncrementalState()
        {
            InitIncrementalState();
        }

        public void InitIncrementalState()
        {
            _incremental_state_id = _global_incremental_state_id;
            _global_incremental_state_id++;
        }

        public Dictionary<string, torch.Tensor> GetIncrementalState(
            torch.nn.Module module,
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState,
            string key)
        {
            var fullKey = GetFullIncrementalStateKey(GetModuleName(module), key);
            ++_incremental_state_id;
            return incrementalState?.ContainsKey(fullKey) == true ? incrementalState[fullKey] : EmptyIncrementalState;
        }

        public void SetIncrementalState(
            torch.nn.Module module,
            Dictionary<string, Dictionary<string, torch.Tensor>> incrementalState,
            string key,
            Dictionary<string, torch.Tensor> value)
        {
            incrementalState = incrementalState ?? throw new ArgumentNullException(nameof(incrementalState));

            var fullKey = GetFullIncrementalStateKey(GetModuleName(module), key);
            if (incrementalState.TryGetValue(fullKey, out var oldState))
            {
                TorchUtils.DisposeDictionaryWithTensor(oldState);
            }
            incrementalState[fullKey] = value;
        }

        private static string GetModuleName(torch.nn.Module module)
        {
            return module?.GetName() ?? "<Empty>";
        }

        private string GetFullIncrementalStateKey(string moduleName, string key)
        {
            return $"{moduleName}.{_incremental_state_id}.{key}";
        }
    }
}
