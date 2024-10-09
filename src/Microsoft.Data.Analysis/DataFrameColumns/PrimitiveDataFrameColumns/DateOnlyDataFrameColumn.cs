// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public partial class DateOnlyDataFrameColumn : PrimitiveDataFrameColumn<DateOnly>
    {
        public DateOnlyDataFrameColumn(string name, IEnumerable<DateOnly?> values) : base(name, values) { }

        public DateOnlyDataFrameColumn(string name, IEnumerable<DateOnly> values) : base(name, values) { }

        public DateOnlyDataFrameColumn(string name, long length = 0) : base(name, length) { }

        public DateOnlyDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal DateOnlyDataFrameColumn(string name, PrimitiveColumnContainer<DateOnly> values) : base(name, values) { }

        protected override PrimitiveDataFrameColumn<DateOnly> CreateNewColumn(string name, long length = 0)
        {
            return new DateOnlyDataFrameColumn(name, length);
        }

        internal override PrimitiveDataFrameColumn<DateOnly> CreateNewColumn(string name, PrimitiveColumnContainer<DateOnly> container)
        {
            return new DateOnlyDataFrameColumn(name, container);
        }
    }
}
#endif
