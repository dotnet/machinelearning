// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class ByteDataFrameColumn : PrimitiveDataFrameColumn<byte>
    {
        public ByteDataFrameColumn(string name, IEnumerable<byte?> values) : base(name, values) { }

        public ByteDataFrameColumn(string name, IEnumerable<byte> values) : base(name, values) { }

        public ByteDataFrameColumn(string name, long length = 0) : base(name, length) { }

        public ByteDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal ByteDataFrameColumn(string name, PrimitiveColumnContainer<byte> values) : base(name, values) { }
    }
}
