// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public partial class DateTimeOffsetDataFrameColumn : PrimitiveDataFrameColumn<DateTimeOffset>
    {
        public DateTimeOffsetDataFrameColumn(string name, IEnumerable<DateTimeOffset?> values) : base(name, values) { }

        public DateTimeOffsetDataFrameColumn(string name, IEnumerable<DateTimeOffset> values) : base(name, values) { }

        public DateTimeOffsetDataFrameColumn(string name, long length = 0) : base(name, length) { }

        public DateTimeOffsetDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal DateTimeOffsetDataFrameColumn(string name, PrimitiveColumnContainer<DateTimeOffset> values) : base(name, values) { }

        protected override PrimitiveDataFrameColumn<DateTimeOffset> CreateNewColumn(string name, long length = 0)
        {
            return new DateTimeOffsetDataFrameColumn(name, length);
        }

        internal override PrimitiveDataFrameColumn<DateTimeOffset> CreateNewColumn(string name, PrimitiveColumnContainer<DateTimeOffset> container)
        {
            return new DateTimeOffsetDataFrameColumn(name, container);
        }
    }
}
