// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    internal class ColumnDimensions
    {
        public readonly int? Cardinality;
        public readonly int? MissingValueCount;
        public readonly SummaryStatistics SummaryStatistics;

        public ColumnDimensions(int? cardinality, int? missingValueCount,
            SummaryStatistics summaryStatistics = null)
        {
            Cardinality = cardinality;
            MissingValueCount = missingValueCount;
            SummaryStatistics = summaryStatistics;
        }

        public bool HasMissingValues()
        {
            return MissingValueCount != null && MissingValueCount.Value > 0;
        }
    }
}
