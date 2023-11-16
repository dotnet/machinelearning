

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameColumn.BinaryOperationAPIs.ExplodedColumns.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{

    public partial class ByteDataFrameColumn
    {
        public Int32DataFrameColumn Add(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public UInt32DataFrameColumn Add(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public UInt32DataFrameColumn Subtract(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public UInt32DataFrameColumn Multiply(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public UInt32DataFrameColumn Divide(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public UInt32DataFrameColumn Modulo(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public UInt64DataFrameColumn Add(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public UInt64DataFrameColumn Subtract(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public UInt64DataFrameColumn Multiply(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public UInt64DataFrameColumn Divide(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public UInt64DataFrameColumn Modulo(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
    }

    public partial class DecimalDataFrameColumn
    {
        public DecimalDataFrameColumn Add(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, column, inPlace);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, column, inPlace);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace);
        }
        public DecimalDataFrameColumn Add(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Subtract(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Multiply(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Divide(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdecimalColumn, inPlace);
        }
        public DecimalDataFrameColumn Modulo(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdecimalColumn, inPlace);
        }
    }

    public partial class DoubleDataFrameColumn
    {
        public DoubleDataFrameColumn Add(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, column, inPlace);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, column, inPlace);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace);
        }
        public DoubleDataFrameColumn Add(SingleDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(SingleDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(SingleDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(SingleDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(SingleDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Subtract(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Multiply(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Divide(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherdoubleColumn, inPlace);
        }
        public DoubleDataFrameColumn Modulo(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherdoubleColumn, inPlace);
        }
    }

    public partial class SingleDataFrameColumn
    {
        public SingleDataFrameColumn Add(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, column, inPlace);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, column, inPlace);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace);
        }
        public SingleDataFrameColumn Add(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Add(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(Int64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Add(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Add(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Add(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Add(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(UInt64DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Add(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Subtract(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Multiply(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Divide(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace);
        }
        public SingleDataFrameColumn Modulo(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace);
        }
    }

    public partial class Int32DataFrameColumn
    {
        public Int32DataFrameColumn Add(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Subtract(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Multiply(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Divide(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Modulo(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, column, inPlace);
        }
        public Int32DataFrameColumn Subtract(Int32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace);
        }
        public Int32DataFrameColumn Multiply(Int32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace);
        }
        public Int32DataFrameColumn Divide(Int32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, column, inPlace);
        }
        public Int32DataFrameColumn Modulo(Int32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace);
        }
        public Int64DataFrameColumn Add(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Subtract(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Multiply(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Divide(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Modulo(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Add(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Subtract(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Multiply(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Divide(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Modulo(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace);
        }
        public Int64DataFrameColumn Add(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Divide(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace: true);
        }
        public SingleDataFrameColumn Add(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Divide(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace: true);
        }
        public Int32DataFrameColumn Add(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Subtract(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Multiply(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Divide(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace);
        }
        public Int32DataFrameColumn Modulo(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace);
        }
    }

    public partial class Int64DataFrameColumn
    {
        public Int64DataFrameColumn Add(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Subtract(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Multiply(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Divide(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Modulo(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Subtract(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Multiply(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Divide(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Modulo(Int32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Add(Int64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, column, inPlace);
        }
        public Int64DataFrameColumn Subtract(Int64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace);
        }
        public Int64DataFrameColumn Multiply(Int64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace);
        }
        public Int64DataFrameColumn Divide(Int64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, column, inPlace);
        }
        public Int64DataFrameColumn Modulo(Int64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace);
        }
        public Int64DataFrameColumn Add(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Subtract(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Multiply(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Divide(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Modulo(SByteDataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Add(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Subtract(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Multiply(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Divide(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Modulo(Int16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Add(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Subtract(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Multiply(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Divide(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Modulo(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Add(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Subtract(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Multiply(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Divide(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace);
        }
        public Int64DataFrameColumn Modulo(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace);
        }
    }

    public partial class SByteDataFrameColumn
    {
        public Int32DataFrameColumn Add(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public Int64DataFrameColumn Add(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Divide(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace: true);
        }
        public SingleDataFrameColumn Add(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Divide(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace: true);
        }
        public Int32DataFrameColumn Add(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
    }

    public partial class Int16DataFrameColumn
    {
        public Int32DataFrameColumn Add(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public Int64DataFrameColumn Add(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Divide(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(UInt32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace: true);
        }
        public SingleDataFrameColumn Add(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Divide(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(UInt64DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace: true);
        }
        public Int32DataFrameColumn Add(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
    }

    public partial class UInt32DataFrameColumn
    {
        public UInt32DataFrameColumn Add(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Add, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Subtract(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Multiply(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Divide(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Modulo(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otheruintColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int32DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int64DataFrameColumn Add(SByteDataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(SByteDataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(SByteDataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Divide(SByteDataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(SByteDataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int16DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int16DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int16DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int16DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, otherlongColumn, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int16DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            var otherlongColumn = column.CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherlongColumn, inPlace: true);
        }
        public UInt32DataFrameColumn Add(UInt32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, column, inPlace);
        }
        public UInt32DataFrameColumn Subtract(UInt32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace);
        }
        public UInt32DataFrameColumn Multiply(UInt32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace);
        }
        public UInt32DataFrameColumn Divide(UInt32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, column, inPlace);
        }
        public UInt32DataFrameColumn Modulo(UInt32DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace);
        }
        public UInt64DataFrameColumn Add(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public UInt64DataFrameColumn Subtract(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public UInt64DataFrameColumn Multiply(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public UInt64DataFrameColumn Divide(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public UInt64DataFrameColumn Modulo(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public UInt32DataFrameColumn Add(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Add, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Subtract(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Multiply(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Divide(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otheruintColumn, inPlace);
        }
        public UInt32DataFrameColumn Modulo(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otheruintColumn, inPlace);
        }
    }

    public partial class UInt64DataFrameColumn
    {
        public UInt64DataFrameColumn Add(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Subtract(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Multiply(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Divide(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Modulo(ByteDataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherulongColumn, inPlace);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(Int32DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(Int32DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(Int32DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Divide(Int32DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(Int32DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Add(SByteDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SByteDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SByteDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SByteDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SByteDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Add(Int16DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(Int16DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(Int16DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Divide(Int16DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, otherfloatColumn, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(Int16DataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            var otherfloatColumn = column.CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherfloatColumn, inPlace: true);
        }
        public UInt64DataFrameColumn Add(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Subtract(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Multiply(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Divide(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Modulo(UInt32DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Add(UInt64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, column, inPlace);
        }
        public UInt64DataFrameColumn Subtract(UInt64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace);
        }
        public UInt64DataFrameColumn Multiply(UInt64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace);
        }
        public UInt64DataFrameColumn Divide(UInt64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, column, inPlace);
        }
        public UInt64DataFrameColumn Modulo(UInt64DataFrameColumn column, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace);
        }
        public UInt64DataFrameColumn Add(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Add, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Subtract(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Subtract, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Multiply(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Multiply, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Divide(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Divide, otherulongColumn, inPlace);
        }
        public UInt64DataFrameColumn Modulo(UInt16DataFrameColumn column, bool inPlace = false)
        {
            var otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(BinaryOperation.Modulo, otherulongColumn, inPlace);
        }
    }

    public partial class UInt16DataFrameColumn
    {
        public Int32DataFrameColumn Add(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public DecimalDataFrameColumn Add(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(DecimalDataFrameColumn column)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public DoubleDataFrameColumn Add(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(DoubleDataFrameColumn column)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public SingleDataFrameColumn Add(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public SingleDataFrameColumn Divide(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(SingleDataFrameColumn column)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int32DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int64DataFrameColumn Add(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public Int64DataFrameColumn Divide(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(Int64DataFrameColumn column)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(SByteDataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Add(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(Int16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
        public UInt32DataFrameColumn Add(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public UInt32DataFrameColumn Subtract(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public UInt32DataFrameColumn Multiply(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public UInt32DataFrameColumn Divide(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public UInt32DataFrameColumn Modulo(UInt32DataFrameColumn column)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public UInt64DataFrameColumn Add(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Add, column, inPlace: true);
        }
        public UInt64DataFrameColumn Subtract(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Subtract, column, inPlace: true);
        }
        public UInt64DataFrameColumn Multiply(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Multiply, column, inPlace: true);
        }
        public UInt64DataFrameColumn Divide(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Divide, column, inPlace: true);
        }
        public UInt64DataFrameColumn Modulo(UInt64DataFrameColumn column)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Modulo, column, inPlace: true);
        }
        public Int32DataFrameColumn Add(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Divide(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, otherintColumn, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(UInt16DataFrameColumn column)
        {
            var intColumn = CloneAsInt32Column();
            var otherintColumn = column.CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, otherintColumn, inPlace: true);
        }
    }

    public partial class ByteDataFrameColumn
    {
        public Int32DataFrameColumn Add(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt32DataFrameColumn Add(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseAdd(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt32DataFrameColumn Subtract(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseSubtract(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt32DataFrameColumn Multiply(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseMultiply(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt32DataFrameColumn Divide(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseDivide(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt32DataFrameColumn Modulo(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseModulo(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt64DataFrameColumn Add(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseAdd(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt64DataFrameColumn Subtract(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseSubtract(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt64DataFrameColumn Multiply(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseMultiply(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt64DataFrameColumn Divide(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseDivide(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt64DataFrameColumn Modulo(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseModulo(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
    }

    public partial class DecimalDataFrameColumn
    {
        public DecimalDataFrameColumn Add(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(byte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(decimal value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public DecimalDataFrameColumn Subtract(decimal value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public DecimalDataFrameColumn Multiply(decimal value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public DecimalDataFrameColumn Divide(decimal value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public DecimalDataFrameColumn Modulo(decimal value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public DecimalDataFrameColumn Add(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(int value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(long value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(short value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(uint value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(ulong value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseAdd(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Subtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseSubtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Multiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseMultiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Divide(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseDivide(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Modulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn ReverseModulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (decimal)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
    }

    public partial class DoubleDataFrameColumn
    {
        public DoubleDataFrameColumn Add(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(byte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(double value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(double value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public DoubleDataFrameColumn Subtract(double value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public DoubleDataFrameColumn Multiply(double value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public DoubleDataFrameColumn Divide(double value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(double value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public DoubleDataFrameColumn Modulo(double value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(double value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public DoubleDataFrameColumn Add(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(float value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(int value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(long value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(short value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(uint value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(ulong value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseAdd(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Subtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseSubtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Multiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseMultiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Divide(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseDivide(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Modulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn ReverseModulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (double)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
    }

    public partial class SingleDataFrameColumn
    {
        public SingleDataFrameColumn Add(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(byte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(float value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public SingleDataFrameColumn Subtract(float value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(float value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public SingleDataFrameColumn Multiply(float value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(float value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public SingleDataFrameColumn Divide(float value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(float value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public SingleDataFrameColumn Modulo(float value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(float value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public SingleDataFrameColumn Add(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(int value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Add(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(long value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Add(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Add(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(short value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Add(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(uint value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Add(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(ulong value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Add(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseAdd(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Subtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseSubtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Multiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseMultiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Divide(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseDivide(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public SingleDataFrameColumn Modulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public SingleDataFrameColumn ReverseModulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (float)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
    }

    public partial class Int32DataFrameColumn
    {
        public Int32DataFrameColumn Add(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseAdd(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Subtract(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseSubtract(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Multiply(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseMultiply(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Divide(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseDivide(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Modulo(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseModulo(byte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(int value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public Int32DataFrameColumn ReverseAdd(int value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public Int32DataFrameColumn Subtract(int value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public Int32DataFrameColumn ReverseSubtract(int value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public Int32DataFrameColumn Multiply(int value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public Int32DataFrameColumn ReverseMultiply(int value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public Int32DataFrameColumn Divide(int value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public Int32DataFrameColumn ReverseDivide(int value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public Int32DataFrameColumn Modulo(int value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public Int32DataFrameColumn ReverseModulo(int value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public Int64DataFrameColumn Add(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseAdd(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Subtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseSubtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Multiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseMultiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Divide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseDivide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Modulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseModulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Add(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseAdd(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Subtract(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseSubtract(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Multiply(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseMultiply(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Divide(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseDivide(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Modulo(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseModulo(short value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Add(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseAdd(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Subtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseSubtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Multiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseMultiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Divide(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseDivide(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int32DataFrameColumn Modulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int32DataFrameColumn ReverseModulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (int)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
    }

    public partial class Int64DataFrameColumn
    {
        public Int64DataFrameColumn Add(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseAdd(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Subtract(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseSubtract(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Multiply(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseMultiply(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Divide(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseDivide(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Modulo(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseModulo(byte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseAdd(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Subtract(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseSubtract(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Multiply(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseMultiply(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Divide(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseDivide(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Modulo(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseModulo(int value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Add(long value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public Int64DataFrameColumn ReverseAdd(long value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public Int64DataFrameColumn Subtract(long value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public Int64DataFrameColumn ReverseSubtract(long value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public Int64DataFrameColumn Multiply(long value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public Int64DataFrameColumn ReverseMultiply(long value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public Int64DataFrameColumn Divide(long value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public Int64DataFrameColumn ReverseDivide(long value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public Int64DataFrameColumn Modulo(long value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public Int64DataFrameColumn ReverseModulo(long value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public Int64DataFrameColumn Add(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseAdd(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Subtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseSubtract(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Multiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseMultiply(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Divide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseDivide(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Modulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseModulo(sbyte value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Add(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseAdd(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Subtract(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseSubtract(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Multiply(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseMultiply(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Divide(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseDivide(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Modulo(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseModulo(short value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Add(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseAdd(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Subtract(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseSubtract(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Multiply(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseMultiply(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Divide(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseDivide(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Modulo(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseModulo(uint value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Add(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseAdd(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Subtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseSubtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Multiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseMultiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Divide(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseDivide(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public Int64DataFrameColumn Modulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public Int64DataFrameColumn ReverseModulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (long)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
    }

    public partial class SByteDataFrameColumn
    {
        public Int32DataFrameColumn Add(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
    }

    public partial class Int16DataFrameColumn
    {
        public Int32DataFrameColumn Add(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(uint value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(ulong value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
    }

    public partial class UInt32DataFrameColumn
    {
        public UInt32DataFrameColumn Add(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseAdd(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Subtract(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseSubtract(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Multiply(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseMultiply(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Divide(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseDivide(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Modulo(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseModulo(byte value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(int value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(sbyte value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(short value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt32DataFrameColumn Add(uint value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public UInt32DataFrameColumn ReverseAdd(uint value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public UInt32DataFrameColumn Subtract(uint value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public UInt32DataFrameColumn ReverseSubtract(uint value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public UInt32DataFrameColumn Multiply(uint value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public UInt32DataFrameColumn ReverseMultiply(uint value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public UInt32DataFrameColumn Divide(uint value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public UInt32DataFrameColumn ReverseDivide(uint value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public UInt32DataFrameColumn Modulo(uint value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public UInt32DataFrameColumn ReverseModulo(uint value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public UInt64DataFrameColumn Add(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseAdd(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt64DataFrameColumn Subtract(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseSubtract(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt64DataFrameColumn Multiply(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseMultiply(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt64DataFrameColumn Divide(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseDivide(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt64DataFrameColumn Modulo(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseModulo(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt32DataFrameColumn Add(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseAdd(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Subtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseSubtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Multiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseMultiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Divide(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseDivide(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn Modulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public UInt32DataFrameColumn ReverseModulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (uint)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
    }

    public partial class UInt64DataFrameColumn
    {
        public UInt64DataFrameColumn Add(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseAdd(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Subtract(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseSubtract(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Multiply(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseMultiply(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Divide(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseDivide(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Modulo(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseModulo(byte value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(int value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(sbyte value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(short value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt64DataFrameColumn Add(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseAdd(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Subtract(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseSubtract(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Multiply(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseMultiply(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Divide(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseDivide(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Modulo(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseModulo(uint value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Add(ulong value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public UInt64DataFrameColumn ReverseAdd(ulong value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }
        public UInt64DataFrameColumn Subtract(ulong value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public UInt64DataFrameColumn ReverseSubtract(ulong value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }
        public UInt64DataFrameColumn Multiply(ulong value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public UInt64DataFrameColumn ReverseMultiply(ulong value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }
        public UInt64DataFrameColumn Divide(ulong value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public UInt64DataFrameColumn ReverseDivide(ulong value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }
        public UInt64DataFrameColumn Modulo(ulong value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public UInt64DataFrameColumn ReverseModulo(ulong value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }
        public UInt64DataFrameColumn Add(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseAdd(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Add, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Subtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseSubtract(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Multiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseMultiply(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Divide(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseDivide(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Divide, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn Modulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
        public UInt64DataFrameColumn ReverseModulo(ushort value, bool inPlace = false)
        {
            var convertedValue = (ulong)value;
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, convertedValue, inPlace);
        }
    }

    public partial class UInt16DataFrameColumn
    {
        public Int32DataFrameColumn Add(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(byte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn Add(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseAdd(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DecimalDataFrameColumn Subtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseSubtract(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DecimalDataFrameColumn Multiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseMultiply(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DecimalDataFrameColumn Divide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseDivide(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DecimalDataFrameColumn Modulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DecimalDataFrameColumn ReverseModulo(decimal value)
        {
            var decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn Add(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseAdd(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public DoubleDataFrameColumn Subtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseSubtract(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public DoubleDataFrameColumn Multiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseMultiply(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public DoubleDataFrameColumn Divide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseDivide(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public DoubleDataFrameColumn Modulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public DoubleDataFrameColumn ReverseModulo(double value)
        {
            var doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn Add(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseAdd(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public SingleDataFrameColumn Subtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseSubtract(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public SingleDataFrameColumn Multiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseMultiply(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public SingleDataFrameColumn Divide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseDivide(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public SingleDataFrameColumn Modulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public SingleDataFrameColumn ReverseModulo(float value)
        {
            var floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(int value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn Add(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseAdd(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int64DataFrameColumn Subtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseSubtract(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int64DataFrameColumn Multiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseMultiply(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int64DataFrameColumn Divide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseDivide(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int64DataFrameColumn Modulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int64DataFrameColumn ReverseModulo(long value)
        {
            var longColumn = CloneAsInt64Column();
            return longColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(sbyte value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(short value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt32DataFrameColumn Add(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseAdd(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt32DataFrameColumn Subtract(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseSubtract(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt32DataFrameColumn Multiply(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseMultiply(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt32DataFrameColumn Divide(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseDivide(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt32DataFrameColumn Modulo(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt32DataFrameColumn ReverseModulo(uint value)
        {
            var uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt64DataFrameColumn Add(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseAdd(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public UInt64DataFrameColumn Subtract(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseSubtract(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public UInt64DataFrameColumn Multiply(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseMultiply(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public UInt64DataFrameColumn Divide(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseDivide(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public UInt64DataFrameColumn Modulo(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public UInt64DataFrameColumn ReverseModulo(ulong value)
        {
            var ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn Add(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseAdd(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace: true);
        }
        public Int32DataFrameColumn Subtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseSubtract(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace: true);
        }
        public Int32DataFrameColumn Multiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseMultiply(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace: true);
        }
        public Int32DataFrameColumn Divide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseDivide(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace: true);
        }
        public Int32DataFrameColumn Modulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
        public Int32DataFrameColumn ReverseModulo(ushort value)
        {
            var intColumn = CloneAsInt32Column();
            return intColumn.HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace: true);
        }
    }

    public partial class BooleanDataFrameColumn
    {
        public BooleanDataFrameColumn And(BooleanDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            BooleanDataFrameColumn retColumn = inPlace ? this : CloneAsBooleanColumn();
            retColumn.ColumnContainer.HandleOperation(BinaryOperation.And, column.ColumnContainer);
            return retColumn;
        }
    }

    public partial class BooleanDataFrameColumn
    {
        public new BooleanDataFrameColumn And(bool value, bool inPlace = false)
        {
            BooleanDataFrameColumn retColumn = inPlace ? this : CloneAsBooleanColumn();
            retColumn.ColumnContainer.HandleOperation(BinaryOperation.And, value);
            return retColumn;
        }
    }

    public partial class BooleanDataFrameColumn
    {
        public BooleanDataFrameColumn Or(BooleanDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            BooleanDataFrameColumn retColumn = inPlace ? this : CloneAsBooleanColumn();
            retColumn.ColumnContainer.HandleOperation(BinaryOperation.Or, column.ColumnContainer);
            return retColumn;
        }
    }

    public partial class BooleanDataFrameColumn
    {
        public new BooleanDataFrameColumn Or(bool value, bool inPlace = false)
        {
            BooleanDataFrameColumn retColumn = inPlace ? this : CloneAsBooleanColumn();
            retColumn.ColumnContainer.HandleOperation(BinaryOperation.Or, value);
            return retColumn;
        }
    }

    public partial class BooleanDataFrameColumn
    {
        public BooleanDataFrameColumn Xor(BooleanDataFrameColumn column, bool inPlace = false)
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            BooleanDataFrameColumn retColumn = inPlace ? this : CloneAsBooleanColumn();
            retColumn.ColumnContainer.HandleOperation(BinaryOperation.Xor, column.ColumnContainer);
            return retColumn;
        }
    }

    public partial class BooleanDataFrameColumn
    {
        public new BooleanDataFrameColumn Xor(bool value, bool inPlace = false)
        {
            BooleanDataFrameColumn retColumn = inPlace ? this : CloneAsBooleanColumn();
            retColumn.ColumnContainer.HandleOperation(BinaryOperation.Xor, value);
            return retColumn;
        }
    }


    public partial class BooleanDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(BooleanDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(BooleanDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(BooleanDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(BooleanDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(BooleanDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(BooleanDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
    }

    public partial class ByteDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            ByteDataFrameColumn otherbyteColumn = column.CloneAsByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            ByteDataFrameColumn otherbyteColumn = column.CloneAsByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            ByteDataFrameColumn otherbyteColumn = column.CloneAsByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            ByteDataFrameColumn otherbyteColumn = column.CloneAsByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            ByteDataFrameColumn otherbyteColumn = column.CloneAsByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            ByteDataFrameColumn otherbyteColumn = column.CloneAsByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
    }

    public partial class DecimalDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            DecimalDataFrameColumn otherdecimalColumn = column.CloneAsDecimalColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalColumn);
        }
    }

    public partial class DoubleDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            DoubleDataFrameColumn otherdoubleColumn = column.CloneAsDoubleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleColumn);
        }
    }

    public partial class SingleDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            SingleDataFrameColumn otherfloatColumn = column.CloneAsSingleColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatColumn);
        }
    }

    public partial class Int32DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            Int32DataFrameColumn otherintColumn = column.CloneAsInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintColumn);
        }
    }

    public partial class Int64DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            Int64DataFrameColumn otherlongColumn = column.CloneAsInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongColumn);
        }
    }

    public partial class SByteDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            SByteDataFrameColumn othersbyteColumn = column.CloneAsSByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othersbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            SByteDataFrameColumn othersbyteColumn = column.CloneAsSByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othersbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            SByteDataFrameColumn othersbyteColumn = column.CloneAsSByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othersbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            SByteDataFrameColumn othersbyteColumn = column.CloneAsSByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othersbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            SByteDataFrameColumn othersbyteColumn = column.CloneAsSByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othersbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            SByteDataFrameColumn othersbyteColumn = column.CloneAsSByteColumn();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othersbyteColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
    }

    public partial class Int16DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othershortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            Int16DataFrameColumn othershortColumn = column.CloneAsInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othershortColumn);
        }
    }

    public partial class UInt32DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            UInt32DataFrameColumn otheruintColumn = column.CloneAsUInt32Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintColumn);
        }
    }

    public partial class UInt64DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            UInt64DataFrameColumn otherulongColumn = column.CloneAsUInt64Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongColumn);
        }
    }

    public partial class UInt16DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(ByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DecimalDataFrameColumn column)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DoubleDataFrameColumn column)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SingleDataFrameColumn column)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int32DataFrameColumn column)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int64DataFrameColumn column)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(SByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(SByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(SByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(SByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(SByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(SByteDataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(Int16DataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(Int16DataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(Int16DataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(Int16DataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(Int16DataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(Int16DataFrameColumn column)
        {
            UInt16DataFrameColumn otherushortColumn = column.CloneAsUInt16Column();
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherushortColumn);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt32DataFrameColumn column)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt64DataFrameColumn column)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
        public BooleanDataFrameColumn ElementwiseEquals(UInt16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(UInt16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(UInt16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(UInt16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(UInt16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(UInt16DataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(DateTimeDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DateTimeDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DateTimeDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DateTimeDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, column);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DateTimeDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, column);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DateTimeDataFrameColumn column)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, column);
        }
    }

    public partial class BooleanDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(bool value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(bool value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(bool value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(bool value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(bool value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(bool value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
    }

    public partial class ByteDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            byte otherbyteValue = (byte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            byte otherbyteValue = (byte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            byte otherbyteValue = (byte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            byte otherbyteValue = (byte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            byte otherbyteValue = (byte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            byte otherbyteValue = (byte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
    }

    public partial class DecimalDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdecimalValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            decimal otherdecimalValue = (decimal)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdecimalValue);
        }
    }

    public partial class DoubleDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherdoubleValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            double otherdoubleValue = (double)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherdoubleValue);
        }
    }

    public partial class SingleDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherfloatValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            float otherfloatValue = (float)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherfloatValue);
        }
    }

    public partial class Int32DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            int otherintValue = (int)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherintValue);
        }
    }

    public partial class Int64DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherlongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            long otherlongValue = (long)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherlongValue);
        }
    }

    public partial class SByteDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            sbyte othersbyteValue = (sbyte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othersbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            sbyte othersbyteValue = (sbyte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othersbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            sbyte othersbyteValue = (sbyte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othersbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            sbyte othersbyteValue = (sbyte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othersbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            sbyte othersbyteValue = (sbyte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othersbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            sbyte othersbyteValue = (sbyte)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othersbyteValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            Int16DataFrameColumn shortColumn = CloneAsInt16Column();
            return shortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            UInt16DataFrameColumn ushortColumn = CloneAsUInt16Column();
            return ushortColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
    }

    public partial class Int16DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, othershortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            short othershortValue = (short)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, othershortValue);
        }
    }

    public partial class UInt32DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otheruintValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            uint otheruintValue = (uint)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otheruintValue);
        }
    }

    public partial class UInt64DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherulongValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            ulong otherulongValue = (ulong)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherulongValue);
        }
    }

    public partial class UInt16DataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(byte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(byte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(byte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(byte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(byte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(byte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(decimal value)
        {
            DecimalDataFrameColumn decimalColumn = CloneAsDecimalColumn();
            return decimalColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(double value)
        {
            DoubleDataFrameColumn doubleColumn = CloneAsDoubleColumn();
            return doubleColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(float value)
        {
            SingleDataFrameColumn floatColumn = CloneAsSingleColumn();
            return floatColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(int value)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            return intColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(long value)
        {
            Int64DataFrameColumn longColumn = CloneAsInt64Column();
            return longColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(sbyte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(sbyte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(sbyte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(sbyte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(sbyte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(sbyte value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(short value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(short value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(short value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(short value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(short value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(short value)
        {
            ushort otherushortValue = (ushort)value;
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, otherushortValue);
        }
        public BooleanDataFrameColumn ElementwiseEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(uint value)
        {
            UInt32DataFrameColumn uintColumn = CloneAsUInt32Column();
            return uintColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ulong value)
        {
            UInt64DataFrameColumn ulongColumn = CloneAsUInt64Column();
            return ulongColumn.HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
        public BooleanDataFrameColumn ElementwiseEquals(ushort value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(ushort value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(ushort value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(ushort value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(ushort value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(ushort value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
    }

    public partial class DateTimeDataFrameColumn
    {
        public BooleanDataFrameColumn ElementwiseEquals(DateTime value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseNotEquals(DateTime value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThanOrEqual(DateTime value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThanOrEqual(DateTime value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }
        public BooleanDataFrameColumn ElementwiseGreaterThan(DateTime value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }
        public BooleanDataFrameColumn ElementwiseLessThan(DateTime value)
        {
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }
    }


    public partial class ByteDataFrameColumn
    {
        public new Int32DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.LeftShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class Int32DataFrameColumn
    {
        public new Int32DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<int>)base.LeftShift(value, inPlace);
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class Int64DataFrameColumn
    {
        public new Int64DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<long>)base.LeftShift(value, inPlace);
            return new Int64DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class SByteDataFrameColumn
    {
        public new Int32DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.LeftShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class Int16DataFrameColumn
    {
        public new Int32DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.LeftShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class UInt32DataFrameColumn
    {
        public new UInt32DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<uint>)base.LeftShift(value, inPlace);
            return new UInt32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class UInt64DataFrameColumn
    {
        public new UInt64DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<ulong>)base.LeftShift(value, inPlace);
            return new UInt64DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class UInt16DataFrameColumn
    {
        public new Int32DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.LeftShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class ByteDataFrameColumn
    {
        public new Int32DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.RightShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class Int32DataFrameColumn
    {
        public new Int32DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<int>)base.RightShift(value, inPlace);
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class Int64DataFrameColumn
    {
        public new Int64DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<long>)base.RightShift(value, inPlace);
            return new Int64DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class SByteDataFrameColumn
    {
        public new Int32DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.RightShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class Int16DataFrameColumn
    {
        public new Int32DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.RightShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class UInt32DataFrameColumn
    {
        public new UInt32DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<uint>)base.RightShift(value, inPlace);
            return new UInt32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class UInt64DataFrameColumn
    {
        public new UInt64DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            var result = (PrimitiveDataFrameColumn<ulong>)base.RightShift(value, inPlace);
            return new UInt64DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }

    public partial class UInt16DataFrameColumn
    {
        public new Int32DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            Int32DataFrameColumn intColumn = CloneAsInt32Column();
            var result = (PrimitiveDataFrameColumn<int>)(intColumn.RightShift(value, inPlace));
            return new Int32DataFrameColumn(result.Name, result.ColumnContainer);
        }
    }
}
