// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.AutoML;

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

        public string OutputName { get; set; }

        public string OutputBaseDir { get; set; }

        public string TrainDataset { get; set; }

        public string TestDataset { get; set; }

        public GenerateTarget Target { get; set; }

        internal TaskKind MlTask { get; set; }

    }

    internal enum GenerateTarget
    {
        ModelBuilder = 0,
        Cli = 1,
    }
}
