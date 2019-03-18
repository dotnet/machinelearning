// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Opaque IDataView implementation to provide a barrier for data pipe optimizations.
    /// Used in cross validatation to generate the train/test pipelines for each fold.
    /// </summary>
    [BestFriend]
    internal sealed class OpaqueDataView : IDataView
    {
        private readonly IDataView _source;
        public bool CanShuffle => _source.CanShuffle;
        public DataViewSchema Schema => _source.Schema;

        public OpaqueDataView(IDataView source)
        {
            _source = source;
        }

        public long? GetRowCount() => _source.GetRowCount();

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            =>_source.GetRowCursor(columnsNeeded, rand);

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            => _source.GetRowCursorSet(columnsNeeded, n, rand);
    }
}
