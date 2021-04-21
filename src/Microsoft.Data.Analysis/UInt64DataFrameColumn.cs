// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class UInt64DataFrameColumn : PrimitiveDataFrameColumn<ulong>
    {
        public UInt64DataFrameColumn(string name, IEnumerable<ulong?> values) : base(name, values) { }

        public UInt64DataFrameColumn(string name, IEnumerable<ulong> values) : base(name, values) { }

        public UInt64DataFrameColumn(string name, long length = 0) : base(name, length) { }

        public UInt64DataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal UInt64DataFrameColumn(string name, PrimitiveColumnContainer<ulong> values) : base(name, values) { }
    }
}
