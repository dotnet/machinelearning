// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Tools;
namespace Microsoft.ML.Benchmarks
{
    internal static class ExecuteMaml
    {
        public static void ExecuteMamlCommand(this string command, MLContext environment)
        {
            if (Maml.MainCore(environment, command, alwaysPrintStacktrace: false) < 0)
                throw new Exception($"Command {command} returned negative error code");
        }
    }
}
