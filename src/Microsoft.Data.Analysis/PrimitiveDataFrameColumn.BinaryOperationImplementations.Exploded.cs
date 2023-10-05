

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
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, BooleanDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, bool value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class ByteDataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, ByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, byte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class DecimalDataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, DecimalDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, decimal value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class DoubleDataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, DoubleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, double value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class SingleDataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, SingleDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, float value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class Int32DataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, Int32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, int value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class Int64DataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, Int64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, long value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class SByteDataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, SByteDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, sbyte value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class Int16DataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, Int16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, short value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class UInt32DataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, UInt32DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, uint value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class UInt64DataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, UInt64DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, ulong value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class UInt16DataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, UInt16DataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, ushort value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonOperation operation, DateTimeDataFrameColumn column)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, column.ColumnContainer));
        }

        internal BooleanDataFrameColumn HandleOperationImplementation(ComparisonScalarOperation operation, DateTime value)
        {
            return new BooleanDataFrameColumn(Name, ColumnContainer.HandleOperation(operation, value));
        }
    }
}
