// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class DecimalDataFrameColumn : PrimitiveDataFrameColumn<decimal>
    {
        public DecimalDataFrameColumn(string name, IEnumerable<decimal?> values) : base(name, values) { }

        public DecimalDataFrameColumn(string name, IEnumerable<decimal> values) : base(name, values) { }

        public DecimalDataFrameColumn(string name, long length = 0) : base(name, length) { }

        public DecimalDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal DecimalDataFrameColumn(string name, PrimitiveColumnContainer<decimal> values) : base(name, values) { }
    }
}
