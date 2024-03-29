﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class DoubleDataFrameColumn : PrimitiveDataFrameColumn<double>
    {
        public DoubleDataFrameColumn(string name, IEnumerable<double?> values) : base(name, values) { }

        public DoubleDataFrameColumn(string name, IEnumerable<double> values) : base(name, values) { }

        public DoubleDataFrameColumn(string name, long length = 0) : base(name, length) { }

        public DoubleDataFrameColumn(string name, ReadOnlyMemory<byte> buffer, ReadOnlyMemory<byte> nullBitMap, int length = 0, int nullCount = 0) : base(name, buffer, nullBitMap, length, nullCount) { }

        internal DoubleDataFrameColumn(string name, PrimitiveColumnContainer<double> values) : base(name, values) { }

        protected override PrimitiveDataFrameColumn<double> CreateNewColumn(string name, long length = 0)
        {
            return new DoubleDataFrameColumn(name, length);
        }

        internal override PrimitiveDataFrameColumn<double> CreateNewColumn(string name, PrimitiveColumnContainer<double> container)
        {
            return new DoubleDataFrameColumn(name, container);
        }
    }
}
