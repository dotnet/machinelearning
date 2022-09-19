// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Apache.Arrow;
using Apache.Arrow.Types;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    public class VBufferDataFrameColumn<T> : DataFrameColumn, IEnumerable<VBuffer<T>>
    {
        private readonly List<List<VBuffer<T>>> _vBuffers = new List<List<VBuffer<T>>>(); // To store more than intMax number of strings

        public VBufferDataFrameColumn(string name, long length, Type type) : base(name, length, type)
        {
            int numberOfBuffersRequired = Math.Max((int)(length / int.MaxValue), 1);
            for (int i = 0; i < numberOfBuffersRequired; i++)
            {
                long bufferLen = length - _vBuffers.Count * int.MaxValue;
                List<VBuffer<T>> buffer = new List<VBuffer<T>>((int)Math.Min(int.MaxValue, bufferLen));
                _vBuffers.Add(buffer);
                for (int j = 0; j < bufferLen; j++)
                {
                    buffer.Add(default);
                }
            }
        }

        public VBufferDataFrameColumn(string name, IEnumerable<VBuffer<T>> values) : base(name, 0, typeof(VBuffer<T>))
        {
            values = values ?? throw new ArgumentNullException(nameof(values));
            if (_vBuffers.Count == 0)
            {
                _vBuffers.Add(new List<VBuffer<T>>());
            }
            foreach (var value in values)
            {
                Append(value);
            }
        }

        public void Append(VBuffer<T> value)
        {
            List<VBuffer<T>> lastBuffer = _vBuffers[_vBuffers.Count - 1];
            if (lastBuffer.Count == int.MaxValue)
            {
                lastBuffer = new List<VBuffer<T>>();
                _vBuffers.Add(lastBuffer);
            }
            lastBuffer.Add(value);
            Length++;
        }

        public override long NullCount => throw new NotImplementedException();

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
            if (value == null || value is VBuffer<T>)
            {
                int bufferIndex = GetBufferIndexContainingRowIndex(ref rowIndex);
                _vBuffers[bufferIndex][(int)rowIndex] = (VBuffer<T>)value;
            }
            else
            {
                throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(VBuffer<object>)), nameof(value));
            }
        }

        private int GetBufferIndexContainingRowIndex(ref long rowIndex)
        {
            if (rowIndex > Length)
            {
                throw new ArgumentOutOfRangeException(Strings.ColumnIndexOutOfRange, nameof(rowIndex));
            }
            return (int)(rowIndex / int.MaxValue);
        }

        IEnumerator<VBuffer<T>> IEnumerable<VBuffer<T>>.GetEnumerator()
        {
            throw new NotImplementedException();
        }
    }
}
