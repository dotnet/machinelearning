// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Torch
{
    public static class TorchUtils
    {
        /// <summary>
        /// Load Torch model into memory.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">The file path of the model to load.</param>
        internal static TorchModel LoadTorchModel(IHostEnvironment env, string modelPath)
        {
            var module = GetModule(env, modelPath);
            return new TorchModel(env, module, modelPath);
        }

        internal static TorchModuleWrapper GetModule(IHostEnvironment env, string modelPath)
        {
            Contracts.Assert(CheckModel(env, modelPath));
            return new TorchJitModuleWrapper(TorchSharp.JIT.Module.Load(modelPath));
        }

        internal static bool CheckModel(IHostEnvironment env, string modelPath)
        {
            Contracts.CheckValue(env, nameof(env));
            if (IsTorchScriptModel(env, modelPath))
            {
                return true;
            }

            return false;
        }

        // A PyTorch TorchScript model is a single file. Given a modelPath, this utility method
        // determines if we should treat it as a TorchScript model or not.
        internal static bool IsTorchScriptModel(IHostEnvironment env, string modelPath)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(modelPath, nameof(modelPath));
            env.CheckUserArg(File.Exists(modelPath), nameof(modelPath));
            FileAttributes attr = File.GetAttributes(modelPath);
            return attr.HasFlag(FileAttributes.Archive);
        }

        /// <summary>
        /// Creates a folder at a given path. Do nothing if folder already exists.
        /// </summary>
        internal static void CreateFolder(IHostEnvironment env, string folder)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckNonWhiteSpace(folder, nameof(folder));

            // If directory exists, do nothing.
            if (Directory.Exists(folder))
                return;

            // REVIEW: should we do something similar to tensorflow transform and use CreateTempDirectoryWithAcl?
            try
            {
                Directory.CreateDirectory(folder);
            }
            catch (Exception exc)
            {
                throw Contracts.ExceptParam(nameof(folder), $"Failed to create folder for the provided path: {folder}. \nException: {exc.Message}");
            }
        }
    }
}
