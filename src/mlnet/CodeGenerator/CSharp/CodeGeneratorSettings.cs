using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    internal class CodeGeneratorSettings
    {
        internal string LabelName { get; set; }

        internal string ModelPath { get; set; }

        internal string OutputName { get; set; }

        internal string OutputBaseDir { get; set; }

        internal string TrainDataset { get; set; }

        internal string TestDataset { get; set; }

        internal TaskKind MlTask { get; set; }

    }
}
