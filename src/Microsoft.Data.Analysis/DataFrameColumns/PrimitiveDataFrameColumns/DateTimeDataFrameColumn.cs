// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public partial class DateTimeDataFrameColumn : PrimitiveDataFrameColumn<DateTime>
    {
        public DateTimeDataFrameColumn(string name, IEnumerable<DateTime?> values) : base(name, values) { }

        public DateTimeDataFrameColumn(string name, IEnumerable<DateTime> values) : base(name, values) { }

        public DateTimeDataFrameColumn(string name, long length = 0) : base(name, length) { }

        public DateTimeDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal DateTimeDataFrameColumn(string name, PrimitiveColumnContainer<DateTime> values) : base(name, values) { }

        protected override PrimitiveDataFrameColumn<DateTime> CreateNewColumn(string name, long length = 0)
        {
            return new DateTimeDataFrameColumn(name, length);
        }

        internal override PrimitiveDataFrameColumn<DateTime> CreateNewColumn(string name, PrimitiveColumnContainer<DateTime> container)
        {
            return new DateTimeDataFrameColumn(name, container);
        }
    }
}
