using Microsoft.ML.Auto;
using Microsoft.ML.CLI.Utilities.File;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    internal class CodeGeneratorSettings
    {
        internal string LabelName { get; set; }

        internal IFileInfo ModelPath { get; set; }

        internal string OutputName { get; set; }

        internal string OutputBaseDir { get; set; }

        internal IFileInfo TrainDataset { get; set; }

        internal IFileInfo TestDataset { get; set; }

        internal TaskKind MlTask { get; set; }

    }
}
