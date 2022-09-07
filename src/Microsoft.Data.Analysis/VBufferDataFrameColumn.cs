// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    internal class VBufferDataFrameColumn : DataFrameColumn, IEnumerable<VBuffer<object>>
    {
        public VBufferDataFrameColumn(string name, long length, Type type) : base(name, length, type)
        {
        }

        public override long NullCount => throw new NotImplementedException();

        public IEnumerator<VBuffer<object>> GetEnumerator()
        {
            throw new NotImplementedException();
        }

        public override Dictionary<long, ICollection<long>> GetGroupedOccurrences(DataFrameColumn other, out HashSet<long> otherColumnNullIndices)
        {
            throw new NotImplementedException();
        }

        protected override IEnumerator GetEnumeratorCore()
        {
            throw new NotImplementedException();
        }

        protected override object GetValue(long rowIndex)
        {
            throw new NotImplementedException();
        }

        protected override IReadOnlyList<object> GetValues(long startIndex, int length)
        {
            throw new NotImplementedException();
        }

        protected override void SetValue(long rowIndex, object value)
        {
            if (value == null || value is VBuffer<object>)
            {
                int bufferIndex = GetBufferIndexContainingRowIndex(ref rowIndex);
                var oldValue = this[rowIndex];
                _stringBuffers[bufferIndex][(int)rowIndex] = (string)value;
                if (oldValue != (string)value)
                {
                    if (value == null)
                        _nullCount++;
                    if (oldValue == null && _nullCount > 0)
                        _nullCount--;
                }
            }
            else
            {
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(VBuffer<object>)), nameof(value));
            }
        }
    }
}
