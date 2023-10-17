
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumn.BinaryOperations.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Microsoft.Data.Analysis
{
    public partial class PrimitiveDataFrameColumn<T> : DataFrameColumn
        where T : unmanaged
    {
        /// <inheritdoc/>
        public override DataFrameColumn Add(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.Add, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn Subtract(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.Subtract, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn Multiply(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.Multiply, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn Divide(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.Divide, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn Modulo(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.Modulo, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn And(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.And, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.And, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.And, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.And, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.And, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.And, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.And, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.And, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.And, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.And, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.And, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.And, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.And, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.And, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn Or(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.Or, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn Xor(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(BinaryOperation.Xor, DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }


        /// <inheritdoc/>
        public override DataFrameColumn Add<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return Add(column, inPlace);
            }
            return HandleOperationImplementation(BinaryOperation.Add, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseAdd<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Add, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Subtract<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return Subtract(column, inPlace);
            }
            return HandleOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseSubtract<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Subtract, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Multiply<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return Multiply(column, inPlace);
            }
            return HandleOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseMultiply<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Multiply, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Divide<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return Divide(column, inPlace);
            }
            return HandleOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseDivide<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Divide, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Modulo<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return Modulo(column, inPlace);
            }
            return HandleOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseModulo<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryOperation.Modulo, value, inPlace);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> And(bool value, bool inPlace = false)
        {
            return HandleBitwiseOperationImplementation(BinaryOperation.And, value, inPlace);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> Or(bool value, bool inPlace = false)
        {
            return HandleBitwiseOperationImplementation(BinaryOperation.Or, value, inPlace);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> Xor(bool value, bool inPlace = false)
        {
            return HandleBitwiseOperationImplementation(BinaryOperation.Xor, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryIntOperation.LeftShift, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            return HandleOperationImplementation(BinaryIntOperation.RightShift, value, inPlace);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, DateTimeColumn);
                case null:
                    return ElementwiseIsNull();

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals<U>(U value)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return ElementwiseEquals(column);
            }
            return HandleOperationImplementation(ComparisonOperation.ElementwiseEquals, value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseNotEquals(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, DateTimeColumn);
                case null:
                    return ElementwiseIsNotNull();

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseNotEquals<U>(U value)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return ElementwiseNotEquals(column);
            }
            return HandleOperationImplementation(ComparisonOperation.ElementwiseNotEquals, value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanOrEqual(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, DateTimeColumn);

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanOrEqual<U>(U value)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return ElementwiseGreaterThanOrEqual(column);
            }
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThanOrEqual, value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseLessThanOrEqual(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, DateTimeColumn);

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseLessThanOrEqual<U>(U value)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return ElementwiseLessThanOrEqual(column);
            }
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThanOrEqual, value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseGreaterThan(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, DateTimeColumn);

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseGreaterThan<U>(U value)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return ElementwiseGreaterThan(column);
            }
            return HandleOperationImplementation(ComparisonOperation.ElementwiseGreaterThan, value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseLessThan(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, DateTimeColumn);

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseLessThan<U>(U value)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return ElementwiseLessThan(column);
            }
            return HandleOperationImplementation(ComparisonOperation.ElementwiseLessThan, value);
        }

    }
}
