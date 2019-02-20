// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public class ColumnInferenceResults
    {
        public TextLoader.Arguments TextLoaderArgs { get; set; }
        public ColumnInformation ColumnInformation { get; set; }
    }

    public class ColumnInformation
    {
        public string LabelColumn = DefaultColumnNames.Label;
        public string NameColumn = DefaultColumnNames.Name;
        public string GroupIdColumn = DefaultColumnNames.GroupId;
        public string WeightColumn = DefaultColumnNames.Weight;
        public IEnumerable<string> CategoricalColumns { get; set; }
        public IEnumerable<string> NumericColumns { get; set; }
        public IEnumerable<string> TextColumns { get; set; }
        public IEnumerable<string> IgnoredColumns { get; set; }
    }
}
