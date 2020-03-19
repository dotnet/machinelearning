

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameColumn.BinaryOperationImplementations.ExplodedColumns.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn AddImplementation(DecimalDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Add(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn AddImplementation(DoubleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Add(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn AddImplementation(SingleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Add(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn AddImplementation(Int32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Add(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn AddImplementation(Int64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Add(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn AddImplementation(UInt32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Add(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn AddImplementation(UInt64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Add(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn AddImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Add(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn AddImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Add(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn AddImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Add(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn AddImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Add(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn AddImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Add(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn AddImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Add(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn AddImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Add(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn ReverseAddImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.ReverseAdd(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn ReverseAddImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.ReverseAdd(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn ReverseAddImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.ReverseAdd(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn ReverseAddImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.ReverseAdd(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn ReverseAddImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.ReverseAdd(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn ReverseAddImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.ReverseAdd(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn ReverseAddImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.ReverseAdd(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn SubtractImplementation(DecimalDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Subtract(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn SubtractImplementation(DoubleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Subtract(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn SubtractImplementation(SingleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Subtract(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn SubtractImplementation(Int32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Subtract(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn SubtractImplementation(Int64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Subtract(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn SubtractImplementation(UInt32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Subtract(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn SubtractImplementation(UInt64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Subtract(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn SubtractImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Subtract(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn SubtractImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Subtract(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn SubtractImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Subtract(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn SubtractImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Subtract(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn SubtractImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Subtract(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn SubtractImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Subtract(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn SubtractImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Subtract(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn ReverseSubtractImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.ReverseSubtract(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn ReverseSubtractImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.ReverseSubtract(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn ReverseSubtractImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.ReverseSubtract(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn ReverseSubtractImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.ReverseSubtract(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn ReverseSubtractImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.ReverseSubtract(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn ReverseSubtractImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.ReverseSubtract(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn ReverseSubtractImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.ReverseSubtract(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn MultiplyImplementation(DecimalDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Multiply(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn MultiplyImplementation(DoubleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Multiply(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn MultiplyImplementation(SingleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Multiply(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn MultiplyImplementation(Int32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Multiply(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn MultiplyImplementation(Int64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Multiply(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn MultiplyImplementation(UInt32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Multiply(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn MultiplyImplementation(UInt64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Multiply(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn MultiplyImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Multiply(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn MultiplyImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Multiply(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn MultiplyImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Multiply(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn MultiplyImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Multiply(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn MultiplyImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Multiply(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn MultiplyImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Multiply(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn MultiplyImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Multiply(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn ReverseMultiplyImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.ReverseMultiply(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn ReverseMultiplyImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.ReverseMultiply(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn ReverseMultiplyImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.ReverseMultiply(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn ReverseMultiplyImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.ReverseMultiply(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn ReverseMultiplyImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.ReverseMultiply(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn ReverseMultiplyImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.ReverseMultiply(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn ReverseMultiplyImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.ReverseMultiply(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn DivideImplementation(DecimalDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Divide(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn DivideImplementation(DoubleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Divide(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn DivideImplementation(SingleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Divide(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn DivideImplementation(Int32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Divide(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn DivideImplementation(Int64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Divide(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn DivideImplementation(UInt32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Divide(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn DivideImplementation(UInt64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Divide(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn DivideImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Divide(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn DivideImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Divide(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn DivideImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Divide(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn DivideImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Divide(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn DivideImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Divide(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn DivideImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Divide(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn DivideImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Divide(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn ReverseDivideImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.ReverseDivide(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn ReverseDivideImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.ReverseDivide(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn ReverseDivideImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.ReverseDivide(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn ReverseDivideImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.ReverseDivide(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn ReverseDivideImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.ReverseDivide(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn ReverseDivideImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.ReverseDivide(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn ReverseDivideImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.ReverseDivide(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn ModuloImplementation(DecimalDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Modulo(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn ModuloImplementation(DoubleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Modulo(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn ModuloImplementation(SingleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Modulo(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn ModuloImplementation(Int32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Modulo(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn ModuloImplementation(Int64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Modulo(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn ModuloImplementation(UInt32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Modulo(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn ModuloImplementation(UInt64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Modulo(column.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn ModuloImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.Modulo(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn ModuloImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.Modulo(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn ModuloImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.Modulo(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn ModuloImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.Modulo(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn ModuloImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.Modulo(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn ModuloImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.Modulo(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn ModuloImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.Modulo(value);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal DecimalDataFrameColumn ReverseModuloImplementation(decimal value, bool inPlace = false)
        {
            DecimalDataFrameColumn newColumn = inPlace ? this : CloneAsDecimalColumn();
            newColumn.ColumnContainer.ReverseModulo(value);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal DoubleDataFrameColumn ReverseModuloImplementation(double value, bool inPlace = false)
        {
            DoubleDataFrameColumn newColumn = inPlace ? this : CloneAsDoubleColumn();
            newColumn.ColumnContainer.ReverseModulo(value);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal SingleDataFrameColumn ReverseModuloImplementation(float value, bool inPlace = false)
        {
            SingleDataFrameColumn newColumn = inPlace ? this : CloneAsSingleColumn();
            newColumn.ColumnContainer.ReverseModulo(value);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal Int32DataFrameColumn ReverseModuloImplementation(int value, bool inPlace = false)
        {
            Int32DataFrameColumn newColumn = inPlace ? this : CloneAsInt32Column();
            newColumn.ColumnContainer.ReverseModulo(value);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal Int64DataFrameColumn ReverseModuloImplementation(long value, bool inPlace = false)
        {
            Int64DataFrameColumn newColumn = inPlace ? this : CloneAsInt64Column();
            newColumn.ColumnContainer.ReverseModulo(value);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal UInt32DataFrameColumn ReverseModuloImplementation(uint value, bool inPlace = false)
        {
            UInt32DataFrameColumn newColumn = inPlace ? this : CloneAsUInt32Column();
            newColumn.ColumnContainer.ReverseModulo(value);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal UInt64DataFrameColumn ReverseModuloImplementation(ulong value, bool inPlace = false)
        {
            UInt64DataFrameColumn newColumn = inPlace ? this : CloneAsUInt64Column();
            newColumn.ColumnContainer.ReverseModulo(value);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(BooleanDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(ByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(DecimalDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(DoubleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(SingleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(Int32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(Int64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(SByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(Int16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(UInt32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(UInt64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(UInt16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(bool value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(byte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(decimal value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(double value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(float value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(int value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(long value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(sbyte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(short value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(uint value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(ulong value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(ushort value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(BooleanDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(ByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(DecimalDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(DoubleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(SingleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(Int32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(Int64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(SByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(Int16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(UInt32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(UInt64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(UInt16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(bool value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(byte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(decimal value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(double value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(float value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(int value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(long value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(sbyte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(short value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(uint value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(ulong value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(ushort value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseNotEquals(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(BooleanDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(ByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(DecimalDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(DoubleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(SingleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(Int32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(Int64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(SByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(Int16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(UInt32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(UInt64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(UInt16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(bool value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(byte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(decimal value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(double value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(float value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(int value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(long value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(sbyte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(short value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(uint value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(ulong value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(ushort value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(BooleanDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(ByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(DecimalDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(DoubleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(SingleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(Int32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(Int64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(SByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(Int16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(UInt32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(UInt64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(UInt16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(bool value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(byte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(decimal value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(double value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(float value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(int value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(long value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(sbyte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(short value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(uint value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(ulong value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(ushort value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThanOrEqual(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(BooleanDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(ByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(DecimalDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(DoubleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(SingleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(Int32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(Int64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(SByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(Int16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(UInt32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(UInt64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(UInt16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(bool value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(byte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(decimal value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(double value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(float value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(int value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(long value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(sbyte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(short value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(uint value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(ulong value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(ushort value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseGreaterThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(BooleanDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(ByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(DecimalDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(DoubleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(SingleDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(Int32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(Int64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(SByteDataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(Int16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(UInt32DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(UInt64DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(UInt16DataFrameColumn column)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(column.ColumnContainer, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(bool value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(byte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(decimal value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(double value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(float value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(int value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(long value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(sbyte value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(short value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(uint value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(ulong value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(ushort value)
        {
            BooleanDataFrameColumn newColumn = CloneAsBooleanColumn();
            ColumnContainer.ElementwiseLessThan(value, newColumn.ColumnContainer);
            return newColumn;
        }
    }
}
