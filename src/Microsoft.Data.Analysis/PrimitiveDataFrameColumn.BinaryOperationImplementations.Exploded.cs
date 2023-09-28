

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
        //Binary Operations
        internal DecimalDataFrameColumn HandleOperationImplementation(BinaryOperation operation, DecimalDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            var newColumn = inPlace ? this : (DecimalDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, column.ColumnContainer);
            return newColumn;
        }

        //Binary Scalar Operations
        internal DecimalDataFrameColumn HandleOperationImplementation(BinaryScalarOperation operation, decimal right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (DecimalDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, right);
            return newColumn;
        }

        internal DecimalDataFrameColumn HandleReverseOperationImplementation(BinaryScalarOperation operation, decimal right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (DecimalDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleReverseOperation(operation, right);
            return newColumn;
        }
    }

    public partial class DoubleDataFrameColumn
    {
        //Binary Operations
        internal DoubleDataFrameColumn HandleOperationImplementation(BinaryOperation operation, DoubleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            var newColumn = inPlace ? this : (DoubleDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, column.ColumnContainer);
            return newColumn;
        }

        //Binary Scalar Operations
        internal DoubleDataFrameColumn HandleOperationImplementation(BinaryScalarOperation operation, double right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (DoubleDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, right);
            return newColumn;
        }

        internal DoubleDataFrameColumn HandleReverseOperationImplementation(BinaryScalarOperation operation, double right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (DoubleDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleReverseOperation(operation, right);
            return newColumn;
        }
    }

    public partial class SingleDataFrameColumn
    {
        //Binary Operations
        internal SingleDataFrameColumn HandleOperationImplementation(BinaryOperation operation, SingleDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            var newColumn = inPlace ? this : (SingleDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, column.ColumnContainer);
            return newColumn;
        }

        //Binary Scalar Operations
        internal SingleDataFrameColumn HandleOperationImplementation(BinaryScalarOperation operation, float right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (SingleDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, right);
            return newColumn;
        }

        internal SingleDataFrameColumn HandleReverseOperationImplementation(BinaryScalarOperation operation, float right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (SingleDataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleReverseOperation(operation, right);
            return newColumn;
        }
    }

    public partial class Int32DataFrameColumn
    {
        //Binary Operations
        internal Int32DataFrameColumn HandleOperationImplementation(BinaryOperation operation, Int32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            var newColumn = inPlace ? this : (Int32DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, column.ColumnContainer);
            return newColumn;
        }

        //Binary Scalar Operations
        internal Int32DataFrameColumn HandleOperationImplementation(BinaryScalarOperation operation, int right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (Int32DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, right);
            return newColumn;
        }

        internal Int32DataFrameColumn HandleReverseOperationImplementation(BinaryScalarOperation operation, int right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (Int32DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleReverseOperation(operation, right);
            return newColumn;
        }
    }

    public partial class Int64DataFrameColumn
    {
        //Binary Operations
        internal Int64DataFrameColumn HandleOperationImplementation(BinaryOperation operation, Int64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            var newColumn = inPlace ? this : (Int64DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, column.ColumnContainer);
            return newColumn;
        }

        //Binary Scalar Operations
        internal Int64DataFrameColumn HandleOperationImplementation(BinaryScalarOperation operation, long right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (Int64DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, right);
            return newColumn;
        }

        internal Int64DataFrameColumn HandleReverseOperationImplementation(BinaryScalarOperation operation, long right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (Int64DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleReverseOperation(operation, right);
            return newColumn;
        }
    }

    public partial class UInt32DataFrameColumn
    {
        //Binary Operations
        internal UInt32DataFrameColumn HandleOperationImplementation(BinaryOperation operation, UInt32DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            var newColumn = inPlace ? this : (UInt32DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, column.ColumnContainer);
            return newColumn;
        }

        //Binary Scalar Operations
        internal UInt32DataFrameColumn HandleOperationImplementation(BinaryScalarOperation operation, uint right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (UInt32DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, right);
            return newColumn;
        }

        internal UInt32DataFrameColumn HandleReverseOperationImplementation(BinaryScalarOperation operation, uint right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (UInt32DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleReverseOperation(operation, right);
            return newColumn;
        }
    }

    public partial class UInt64DataFrameColumn
    {
        //Binary Operations
        internal UInt64DataFrameColumn HandleOperationImplementation(BinaryOperation operation, UInt64DataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            var newColumn = inPlace ? this : (UInt64DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, column.ColumnContainer);
            return newColumn;
        }

        //Binary Scalar Operations
        internal UInt64DataFrameColumn HandleOperationImplementation(BinaryScalarOperation operation, ulong right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (UInt64DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleOperation(operation, right);
            return newColumn;
        }

        internal UInt64DataFrameColumn HandleReverseOperationImplementation(BinaryScalarOperation operation, ulong right, bool inPlace = false)
        {
            var newColumn = inPlace ? this : (UInt64DataFrameColumn)Clone();
            newColumn.ColumnContainer.HandleReverseOperation(operation, right);
            return newColumn;
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(BooleanDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(ByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(DecimalDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(DoubleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(SingleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(Int32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(Int64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(SByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(Int16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(UInt32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(UInt64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(UInt16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(DateTimeDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(column.ColumnContainer));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(bool value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(byte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(decimal value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(double value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(float value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(int value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(long value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(sbyte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(short value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(uint value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(ulong value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(ushort value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseEqualsImplementation(DateTime value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseEquals(value));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(BooleanDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(ByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(DecimalDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(DoubleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(SingleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(Int32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(Int64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(SByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(Int16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(UInt32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(UInt64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(UInt16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(DateTimeDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(column.ColumnContainer));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(bool value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(byte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(decimal value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(double value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(float value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(int value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(long value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(sbyte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(short value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(uint value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(ulong value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(ushort value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseNotEqualsImplementation(DateTime value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseNotEquals(value));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(BooleanDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(ByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(DecimalDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(DoubleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(SingleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(Int32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(Int64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(SByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(Int16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(UInt32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(UInt64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(UInt16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(DateTimeDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(bool value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(byte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(decimal value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(double value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(float value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(int value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(long value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(sbyte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(short value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(uint value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(ulong value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(ushort value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanOrEqualImplementation(DateTime value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThanOrEqual(value));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(BooleanDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(ByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(DecimalDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(DoubleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(SingleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(Int32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(Int64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(SByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(Int16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(UInt32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(UInt64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(UInt16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(DateTimeDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(column.ColumnContainer));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(bool value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(byte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(decimal value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(double value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(float value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(int value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(long value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(sbyte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(short value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(uint value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(ulong value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(ushort value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanOrEqualImplementation(DateTime value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThanOrEqual(value));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(BooleanDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(ByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(DecimalDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(DoubleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(SingleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(Int32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(Int64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(SByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(Int16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(UInt32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(UInt64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(UInt16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(DateTimeDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(column.ColumnContainer));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(bool value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(byte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(decimal value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(double value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(float value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(int value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(long value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(sbyte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(short value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(uint value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(ulong value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(ushort value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseGreaterThanImplementation(DateTime value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseGreaterThan(value));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(BooleanDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(ByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(DecimalDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(DoubleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(SingleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(Int32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(Int64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(SByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(Int16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(UInt32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(UInt64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(UInt16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(DateTimeDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(column.ColumnContainer));
        }
    }

    public partial class BooleanDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(bool value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(byte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(decimal value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(double value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(float value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(int value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(long value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(sbyte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(short value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(uint value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(ulong value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(ushort value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn ElementwiseLessThanImplementation(DateTime value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.ElementwiseLessThan(value));
        }
    }
}
