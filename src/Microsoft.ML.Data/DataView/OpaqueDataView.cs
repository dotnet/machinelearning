// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using System;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Opaque IDataView implementation to provide a barrier for data pipe optimizations.
    /// Used in cross validatation to generate the train/test pipelines for each fold.
    /// </summary>
    public sealed class OpaqueDataView : IDataView
    {
        private readonly IDataView _source;
        public bool CanShuffle => _source.CanShuffle;
        public Schema Schema => _source.Schema;

        public OpaqueDataView(IDataView source)
        {
            _source = source;
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        public RowCursor GetRowCursor(Func<int, bool> predicate, Random rand = null)
        {
            return _source.GetRowCursor(predicate, rand);
        }

        public RowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, Random rand = null)
        {
            return _source.GetRowCursorSet(out consolidator, predicate, n, rand);
        }
    }
}
