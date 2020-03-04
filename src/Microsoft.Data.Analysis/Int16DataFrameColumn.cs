// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class Int16DataFrameColumn : PrimitiveDataFrameColumn<short>
    {
        public Int16DataFrameColumn(string name, IEnumerable<short?> values) : base(name, values) { }

        public Int16DataFrameColumn(string name, IEnumerable<short> values) : base(name, values) { }

        public Int16DataFrameColumn(string name, long length = 0) : base(name, length) { }

        public Int16DataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal Int16DataFrameColumn(string name, PrimitiveColumnContainer<short> values) : base(name, values) { }
    }
}
