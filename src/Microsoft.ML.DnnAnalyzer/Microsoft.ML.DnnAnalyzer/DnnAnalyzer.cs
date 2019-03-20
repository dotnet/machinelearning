// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

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

            foreach (var (name, opType, type, inputs) in GetModelNodes(args[0]))
            {
                var inputsString = inputs.Length == 0 ? "" : $", input nodes: {string.Join(", ", inputs)}";
                Console.WriteLine($"Graph node: '{name}', operation type: '{opType}', output type: '{type}'{inputsString}");
            }
        }

        private static IEnumerable<(string, string, DataViewType, string[])> GetModelNodes(string modelPath)
        {
            var mlContext = new MLContext();
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(modelPath);
            var schema = tensorFlowModel.GetModelSchema();

            for (int i = 0; i < schema.Count; i++)
            {
                var name = schema[i].Name;
                var type = schema[i].Type;

                var metadataType = schema[i].Annotations.Schema.GetColumnOrNull("TensorflowOperatorType")?.Type;
                ReadOnlyMemory<char> opType = default;
                schema[i].Annotations.GetValue("TensorflowOperatorType", ref opType);
                metadataType = schema[i].Annotations.Schema.GetColumnOrNull("TensorflowUpstreamOperators")?.Type;
                VBuffer <ReadOnlyMemory<char>> inputOps = default;
                if (metadataType != null)
                {
                    schema[i].Annotations.GetValue("TensorflowUpstreamOperators", ref inputOps);
                }

                string[] inputOpsResult = inputOps.DenseValues()
                    .Select(input => input.ToString())
                    .ToArray();

                yield return (name, opType.ToString(), type, inputOpsResult);
            }
        }
    }
}
