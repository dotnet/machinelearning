// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class UInt16DataFrameColumn : PrimitiveDataFrameColumn<ushort>
    {
        public UInt16DataFrameColumn(string name, IEnumerable<ushort?> values) : base(name, values) { }

        public UInt16DataFrameColumn(string name, IEnumerable<ushort> values) : base(name, values) { }

        public UInt16DataFrameColumn(string name, long length = 0) : base(name, length) { }

        public UInt16DataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal UInt16DataFrameColumn(string name, PrimitiveColumnContainer<ushort> values) : base(name, values) { }

        protected override PrimitiveDataFrameColumn<ushort> CreateNewColumn(string name, long length = 0)
        {
            return new UInt16DataFrameColumn(name, length);
        }

        internal override PrimitiveDataFrameColumn<ushort> CreateNewColumn(string name, PrimitiveColumnContainer<ushort> container)
        {
            return new UInt16DataFrameColumn(name, container);
        }
    }
}
