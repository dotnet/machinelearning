// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    internal class ColumnDimensions
    {
        public int? Cardinality;
        public bool? HasMissing;

        public ColumnDimensions(int? cardinality, bool? hasMissing)
        {
            Cardinality = cardinality;
            HasMissing = hasMissing;
        }
    }
}
