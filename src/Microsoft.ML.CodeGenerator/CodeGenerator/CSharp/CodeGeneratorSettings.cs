// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Microsoft.ML.CodeGenerator.CSharp
{
    internal class CodeGeneratorSettings
    {
        public CodeGeneratorSettings()
        {
            // Set default value
            Target = GenerateTarget.Cli;
        }

        public string LabelName { get; set; }

        public string ModelPath { get; set; }

        public string OnnxModelPath { get; set; }

        public string OutputName { get; set; }

        public string OutputBaseDir { get; set; }

        public string TrainDataset { get; set; }

        public string TestDataset { get; set; }

        public GenerateTarget Target { get; set; }

        public string StablePackageVersion { get; set; }

        public string UnstablePackageVersion { get; set; }

        public bool IsAzureAttach { get; set; }

        public bool IsImage { get; set; }

        /// <summary>
        /// classification label
        /// for Azure image only
        /// </summary>
#pragma warning disable MSML_NoInstanceInitializers // No initializers on instance fields or properties
        public string[] ClassificationLabel { get; set; } = new string[] { };

        public IDictionary<string, ColumnMapping> OnnxInputMapping { get; set; } = new Dictionary<string, ColumnMapping>();
#pragma warning restore MSML_NoInstanceInitializers // No initializers on instance fields or properties

        internal TaskKind MlTask { get; set; }

        /// <summary>
        /// For onnx model only
        /// </summary>
        public struct ColumnMapping
        {
            /// <summary>
            /// Mapping Column Name
            /// </summary>
            public string ColumnName;

            /// <summary>
            /// Mapping Column Type
            /// </summary>
            public DataKind ColumnType;
        }
    }

    internal enum GenerateTarget
    {
        ModelBuilder = 0,
        Cli = 1,
    }
}
