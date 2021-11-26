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
            OnnxInputMapping = new Dictionary<string, ColumnMapping>();
            ClassificationLabel = new string[] { };
            ObjectLabel = new string[] { };
        }

        public string LabelName { get; set; }

        /// <summary>
        /// mlnet model name
        /// </summary>
        public string ModelName { get; set; }

        /// <summary>
        /// onnx model name
        /// </summary>
        public string OnnxModelName { get; set; }

        /// <summary>
        /// classification label
        /// for Azure image only
        /// </summary>
        public string[] ClassificationLabel { get; set; }

        public string[] ObjectLabel { get; set; }

        public string OutputName { get; set; }

        public string OutputBaseDir { get; set; }

        public string TrainDataset { get; set; }

        public string TestDataset { get; set; }

        public GenerateTarget Target { get; set; }

        public string StablePackageVersion { get; set; }

        public string UnstablePackageVersion { get; set; }

        public string OnnxRuntimePackageVersion { get; set; }

        public bool IsAzureAttach { get; set; }

        public bool IsImage { get; set; }

        public bool IsObjectDetection { get; set; }

        public IDictionary<string, ColumnMapping> OnnxInputMapping { get; set; }

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
