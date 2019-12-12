// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class StringDataFrameColumn : DataFrameColumn
    {
        public override DataFrameColumn Add(DataFrameColumn column, bool inPlace = false)
        {
            if (Length != column.Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            StringDataFrameColumn ret = inPlace ? this : Clone();
            for (long i = 0; i < Length; i++)
            {
                ret[i] += column[i].ToString();
            }
            return ret;
        }

        public static StringDataFrameColumn operator+(StringDataFrameColumn column, string value)
        {
            return column.Add(value);
        }

        public static StringDataFrameColumn operator+(string value, StringDataFrameColumn column)
        {
            return Add(value, column);
        }

        public static StringDataFrameColumn Add(string value, StringDataFrameColumn right)
        {
            StringDataFrameColumn ret = right.Clone();
            for (int i = 0; i < ret._stringBuffers.Count; i++)
            {
                IList<string> buffer = ret._stringBuffers[i];
                int bufferLen = buffer.Count;
                for (int j = 0; j < bufferLen; j++)
                {
                    buffer[j] = value + buffer[j];
                }
            }
            return ret;
        }

        public StringDataFrameColumn Add(string value, bool inPlace = false)
        {
            StringDataFrameColumn ret = inPlace ? this : Clone();
            for (int i = 0; i < ret._stringBuffers.Count; i++)
            {
                IList<string> buffer = ret._stringBuffers[i];
                int bufferLen = buffer.Count;
                for (int j = 0; j < bufferLen; j++)
                {
                    buffer[j] += value;
                }
            }
            return ret;
        }

        public override DataFrameColumn Add<T>(T value, bool inPlace = false)
        {
            return Add(value.ToString(), inPlace);
        }

        internal static PrimitiveDataFrameColumn<bool> ElementwiseEqualsImplementation(DataFrameColumn left, DataFrameColumn right)
        {
            if (left.Length != right.Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(right));
            }
            PrimitiveDataFrameColumn<bool> ret = new PrimitiveDataFrameColumn<bool>(left.Name, left.Length);
            for (long i = 0; i < left.Length; i++)
            {
                ret[i] = (string)left[i] == right[i]?.ToString();
            }
            return ret;
            
        }

        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals(DataFrameColumn column)
        {
            return ElementwiseEqualsImplementation(this, column);
        }

        public PrimitiveDataFrameColumn<bool> ElementwiseEquals(string value)
        {
            PrimitiveDataFrameColumn<bool> ret = new PrimitiveDataFrameColumn<bool>(Name, Length);
            for (long i = 0; i < Length; i++)
            {
                ret[i] = this[i] == value;
            }
            return ret;
        }

        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals<T>(T value)
        {
            if (value is DataFrameColumn column)
            {
                return ElementwiseEquals(column);
            }
            return ElementwiseEquals(value.ToString());
        }

        internal static PrimitiveDataFrameColumn<bool> ElementwiseNotEqualsImplementation(DataFrameColumn left, DataFrameColumn column)
        {
            if (left.Length != column.Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            PrimitiveDataFrameColumn<bool> ret = new PrimitiveDataFrameColumn<bool>(left.Name, left.Length);
            for (long i = 0; i < left.Length; i++)
            {
                ret[i] = (string)left[i] != column[i].ToString();
            }
            return ret;
        }

        public PrimitiveDataFrameColumn<bool> ElementwiseNotEquals(string value)
        {
            PrimitiveDataFrameColumn<bool> ret = new PrimitiveDataFrameColumn<bool>(Name, Length);
            for (long i = 0; i < Length; i++)
            {
                ret[i] = this[i] != value;
            }
            return ret;
        }

        public override PrimitiveDataFrameColumn<bool> ElementwiseNotEquals(DataFrameColumn column)
        {
            return ElementwiseNotEqualsImplementation(this, column);
        }

        public override PrimitiveDataFrameColumn<bool> ElementwiseNotEquals<T>(T value)
        {
            if (value is DataFrameColumn column)
            {
                return ElementwiseNotEquals(column);
            }
            return ElementwiseNotEquals(value.ToString());
        }
    }
}
