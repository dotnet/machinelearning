// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Transforms.TensorFlow;
using System;

namespace Microsoft.ML.DnnAnalyzer
{
    public static class DnnAnalyzer
    {
        public static void Main(string[] args)
        {
            if (args == null || args.Length != 1)
            {
                Console.Error.WriteLine("Usage: dotnet DnnAnalyzer.dll <model_location>");
                return;
            }

            foreach (var (name, opType, type, inputs) in TensorFlowUtils.GetModelNodes(args[0]))
            {
                var inputsString = inputs.Length == 0 ? "" : $", input nodes: {string.Join(", ", inputs)}";
                Console.WriteLine($"Graph node: '{name}', operation type: '{opType}', output type: '{type}'{inputsString}");
            }
        }
    }
}
