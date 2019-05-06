// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Base class for creating a cursor of rows that filters out some input rows.
    /// </summary>
    [BestFriend]
    internal abstract class LinkedRowFilterCursorBase : LinkedRowRootCursorBase
    {
        public override long Batch => Input.Batch;

        protected LinkedRowFilterCursorBase(IChannelProvider provider, DataViewRowCursor input, DataViewSchema schema, bool[] active)
            : base(provider, input, schema, active)
        {
        }

        public override ValueGetter<DataViewRowId> GetIdGetter()
        {
            return Input.GetIdGetter();
        }

        protected override bool MoveNextCore()
        {
            while (Root.MoveNext())
            {
                if (Accept())
                    return true;
            }

            return false;
        }

        /// <summary>
        /// Return whether the current input row should be returned by this cursor.
        /// </summary>
        protected abstract bool Accept();
    }
}
