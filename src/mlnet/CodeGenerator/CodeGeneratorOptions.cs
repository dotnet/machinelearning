using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI
{
    internal class CodeGeneratorOptions
    {
        internal FileInfo TrainDataset { get; set; }

        internal FileInfo TestDataset { get; set; }

        internal TaskKind MlTask { get; set; }

    }
}
