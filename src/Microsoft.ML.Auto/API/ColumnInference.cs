// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.ObjectModel;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class ColumnInferenceResults
    {
        public TextLoader.Options TextLoaderOptions { get; internal set; } = new TextLoader.Options();
        public ColumnInformation ColumnInformation { get; internal set; } = new ColumnInformation();
    }

    public sealed class ColumnInformation
    {
        public string LabelColumnName { get; set; } = DefaultColumnNames.Label;
        public string ExampleWeightColumnName { get; set; }
        public string SamplingKeyColumnName { get; set; }
        public ICollection<string> CategoricalColumnNames { get; } = new Collection<string>();
        public ICollection<string> NumericColumnNames { get; } = new Collection<string>();
        public ICollection<string> TextColumnNames { get; } = new Collection<string>();
        public ICollection<string> IgnoredColumnNames { get; } = new Collection<string>();
    }
}