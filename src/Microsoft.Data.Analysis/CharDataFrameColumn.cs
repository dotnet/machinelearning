// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class CharDataFrameColumn : PrimitiveDataFrameColumn<char>
    {
        public CharDataFrameColumn(string name, IEnumerable<char?> values) : base(name, values) { }

        public CharDataFrameColumn(string name, IEnumerable<char> values) : base(name, values) { }

        public CharDataFrameColumn(string name, long length = 0) : base(name, length) { }

        public CharDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal CharDataFrameColumn(string name, PrimitiveColumnContainer<char> values) : base(name, values) { }
    }
}
