// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms.TensorFlow;
using System.Linq;

namespace Microsoft.ML.DnnAnalyzer
{
    public static class DnnAnalyzer
    {
        public static void Main(string[] args)
        {
            using (var env = new TlcEnvironment())
            using (var ch = env.Start("DnnAnalyzer"))
            {
                if (Utils.Size(args) != 1)
                {
                    ch.Error("Usage: dotnet DnnAnalyzer.dll <model_location>");
                    ch.Done();
                    return;
                }

                foreach (var (name, opType, type, inputs) in TensorFlowUtils.GetModelNodes(args[0]))
                {
                    var inputsString = inputs.Length == 0 ? "" : $", input nodes: {string.Join(", ", inputs)}";
                    ch.Info($"Graph node: '{name}', operation type: '{opType}', output type: '{type}'{inputsString}");
                }

                ch.Done();
            }
        }
    }
}
