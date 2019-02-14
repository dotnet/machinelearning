// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI
{
    internal class Options
    {
        internal string OutputName { get; set; }

        internal string Name { get; set; }

        internal FileInfo Dataset { get; set; }

        internal FileInfo ValidationDataset { get; set; }

        internal FileInfo TrainDataset { get; set; }

        internal FileInfo TestDataset { get; set; }

        internal string LabelName { get; set; }

        internal uint LabelIndex { get; set; }

        internal TaskKind MlTask { get; set; }

        internal uint Timeout { get; set; }

        internal string OutputBaseDir { get; set; }

    }
}
