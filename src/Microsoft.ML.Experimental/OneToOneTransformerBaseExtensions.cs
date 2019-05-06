// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Experimental
{
    public static class OneToOneTransformerBaseExtensions
    {
        /// <summary>
        /// Returns the names of the input-output column pairs on which the transformation is applied.
        /// </summary>
        public static IReadOnlyList<InputOutputColumnPair> GetColumnPairs(this OneToOneTransformerBase transformer) =>
            InputOutputColumnPair.ConvertFromValueTuples(transformer.ColumnPairs);
    }
}
