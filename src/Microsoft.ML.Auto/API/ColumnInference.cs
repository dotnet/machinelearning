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
        public string LabelColumn { get; set; } = DefaultColumnNames.Label;
        public string WeightColumn { get; set; }
        public ICollection<string> CategoricalColumns { get; } = new Collection<string>();
        public ICollection<string> NumericColumns { get; } = new Collection<string>();
        public ICollection<string> TextColumns { get; } = new Collection<string>();
        public ICollection<string> IgnoredColumns { get; } = new Collection<string>();
    }
}