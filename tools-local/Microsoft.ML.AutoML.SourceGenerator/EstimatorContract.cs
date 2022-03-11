// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json.Serialization;

namespace Microsoft.ML.AutoML.SourceGenerator
{
    internal class EstimatorContract
    {
        [JsonPropertyName("functionName")]
        public string FunctionName { get; set; }

        [JsonPropertyName("estimatorTypes")]
        public List<string> EstimatorTypes { get; set; }

        [JsonPropertyName("nugetDependencies")]
        public List<string> NugetDependencies { get; set; }

        [JsonPropertyName("usingStatements")]
        public List<string> UsingStatements { get; set; }

        [JsonPropertyName("arguments")]
        public List<Argument> ArgumentsList { get; set; }

        [JsonPropertyName("advanceOption")]
        public string AdvanceOption { get; set; }

        [JsonPropertyName("searchOption")]
        public string SearchOption { get; set; }
    }

    internal class Argument
    {
        [JsonPropertyName("argumentName")]
        public string ArgumentName { get; set; }

        [JsonPropertyName("argumentType")]
        public string ArgumentType { get; set; }
    }

    internal class EstimatorsContract
    {
        [JsonPropertyName("estimators")]
        public EstimatorContract[] Estimators { get; set; }
    }
}
