
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
        public override DataFrameColumn Add<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return HandleOperationImplementation(BinaryScalarOperation.Add, column, inPlace);
            }
            return HandleOperationImplementation(BinaryScalarOperation.Add, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseAdd<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryScalarOperation.Add, value, inPlace);
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
        public override DataFrameColumn Subtract<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return HandleOperationImplementation(BinaryScalarOperation.Subtract, column, inPlace);
            }
            return HandleOperationImplementation(BinaryScalarOperation.Subtract, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseSubtract<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryScalarOperation.Subtract, value, inPlace);
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
        public override DataFrameColumn Multiply<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return HandleOperationImplementation(BinaryScalarOperation.Multiply, column, inPlace);
            }
            return HandleOperationImplementation(BinaryScalarOperation.Multiply, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseMultiply<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryScalarOperation.Multiply, value, inPlace);
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
        public override DataFrameColumn Divide<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return HandleOperationImplementation(BinaryScalarOperation.Divide, column, inPlace);
            }
            return HandleOperationImplementation(BinaryScalarOperation.Divide, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseDivide<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryScalarOperation.Divide, value, inPlace);
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
        public override DataFrameColumn Modulo<U>(U value, bool inPlace = false)
        {
            DataFrameColumn column = value as DataFrameColumn;
            if (column != null)
            {
                return HandleOperationImplementation(BinaryScalarOperation.Modulo, column, inPlace);
            }
            return HandleOperationImplementation(BinaryScalarOperation.Modulo, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn ReverseModulo<U>(U value, bool inPlace = false)
        {
            return HandleReverseOperationImplementation(BinaryScalarOperation.Modulo, value, inPlace);
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
        public override PrimitiveDataFrameColumn<bool> And(bool value, bool inPlace = false)
        {
            return HandleBitwiseOperationImplementation(BinaryScalarOperation.And, value, inPlace);
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
        public override PrimitiveDataFrameColumn<bool> Or(bool value, bool inPlace = false)
        {
            return HandleBitwiseOperationImplementation(BinaryScalarOperation.Or, value, inPlace);
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
        public override PrimitiveDataFrameColumn<bool> Xor(bool value, bool inPlace = false)
        {
            return HandleBitwiseOperationImplementation(BinaryScalarOperation.Xor, value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn LeftShift(int value, bool inPlace = false)
        {
            return LeftShiftImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn RightShift(int value, bool inPlace = false)
        {
            return RightShiftImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseEquals(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return ElementwiseEqualsImplementation(boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return ElementwiseEqualsImplementation(byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return ElementwiseEqualsImplementation(charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return ElementwiseEqualsImplementation(decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return ElementwiseEqualsImplementation(doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return ElementwiseEqualsImplementation(floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return ElementwiseEqualsImplementation(intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return ElementwiseEqualsImplementation(longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return ElementwiseEqualsImplementation(sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return ElementwiseEqualsImplementation(shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return ElementwiseEqualsImplementation(uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return ElementwiseEqualsImplementation(ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return ElementwiseEqualsImplementation(ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return ElementwiseEqualsImplementation(DateTimeColumn);
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
            return ElementwiseEqualsImplementation(value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseNotEquals(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return ElementwiseNotEqualsImplementation(boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return ElementwiseNotEqualsImplementation(byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return ElementwiseNotEqualsImplementation(charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return ElementwiseNotEqualsImplementation(decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return ElementwiseNotEqualsImplementation(doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return ElementwiseNotEqualsImplementation(floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return ElementwiseNotEqualsImplementation(intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return ElementwiseNotEqualsImplementation(longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return ElementwiseNotEqualsImplementation(sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return ElementwiseNotEqualsImplementation(shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return ElementwiseNotEqualsImplementation(uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return ElementwiseNotEqualsImplementation(ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return ElementwiseNotEqualsImplementation(ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return ElementwiseNotEqualsImplementation(DateTimeColumn);
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
            return ElementwiseNotEqualsImplementation(value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanOrEqual(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return ElementwiseGreaterThanOrEqualImplementation(DateTimeColumn);

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
            return ElementwiseGreaterThanOrEqualImplementation(value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseLessThanOrEqual(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return ElementwiseLessThanOrEqualImplementation(boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return ElementwiseLessThanOrEqualImplementation(byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return ElementwiseLessThanOrEqualImplementation(charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return ElementwiseLessThanOrEqualImplementation(decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return ElementwiseLessThanOrEqualImplementation(doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return ElementwiseLessThanOrEqualImplementation(floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return ElementwiseLessThanOrEqualImplementation(intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return ElementwiseLessThanOrEqualImplementation(longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return ElementwiseLessThanOrEqualImplementation(sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return ElementwiseLessThanOrEqualImplementation(shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return ElementwiseLessThanOrEqualImplementation(uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return ElementwiseLessThanOrEqualImplementation(ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return ElementwiseLessThanOrEqualImplementation(ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return ElementwiseLessThanOrEqualImplementation(DateTimeColumn);

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
            return ElementwiseLessThanOrEqualImplementation(value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseGreaterThan(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return ElementwiseGreaterThanImplementation(boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return ElementwiseGreaterThanImplementation(byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return ElementwiseGreaterThanImplementation(charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return ElementwiseGreaterThanImplementation(decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return ElementwiseGreaterThanImplementation(doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return ElementwiseGreaterThanImplementation(floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return ElementwiseGreaterThanImplementation(intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return ElementwiseGreaterThanImplementation(longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return ElementwiseGreaterThanImplementation(sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return ElementwiseGreaterThanImplementation(shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return ElementwiseGreaterThanImplementation(uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return ElementwiseGreaterThanImplementation(ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return ElementwiseGreaterThanImplementation(ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return ElementwiseGreaterThanImplementation(DateTimeColumn);

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
            return ElementwiseGreaterThanImplementation(value);
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ElementwiseLessThan(DataFrameColumn column)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return ElementwiseLessThanImplementation(boolColumn);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return ElementwiseLessThanImplementation(byteColumn);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return ElementwiseLessThanImplementation(charColumn);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return ElementwiseLessThanImplementation(decimalColumn);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return ElementwiseLessThanImplementation(doubleColumn);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return ElementwiseLessThanImplementation(floatColumn);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return ElementwiseLessThanImplementation(intColumn);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return ElementwiseLessThanImplementation(longColumn);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return ElementwiseLessThanImplementation(sbyteColumn);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return ElementwiseLessThanImplementation(shortColumn);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return ElementwiseLessThanImplementation(uintColumn);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return ElementwiseLessThanImplementation(ulongColumn);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return ElementwiseLessThanImplementation(ushortColumn);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return ElementwiseLessThanImplementation(DateTimeColumn);

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
            return ElementwiseLessThanImplementation(value);
        }

        internal DataFrameColumn LeftShiftImplementation(int value, bool inPlace)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                    PrimitiveDataFrameColumn<byte> byteColumn = this as PrimitiveDataFrameColumn<byte>;
                    PrimitiveDataFrameColumn<byte> newbyteColumn = inPlace ? byteColumn : byteColumn.Clone();
                    newbyteColumn._columnContainer.LeftShift(value);
                    return newbyteColumn;
                case Type charType when charType == typeof(char):
                    PrimitiveDataFrameColumn<char> charColumn = this as PrimitiveDataFrameColumn<char>;
                    PrimitiveDataFrameColumn<char> newcharColumn = inPlace ? charColumn : charColumn.Clone();
                    newcharColumn._columnContainer.LeftShift(value);
                    return newcharColumn;
                case Type decimalType when decimalType == typeof(decimal):
                    throw new NotSupportedException();
                case Type doubleType when doubleType == typeof(double):
                    throw new NotSupportedException();
                case Type floatType when floatType == typeof(float):
                    throw new NotSupportedException();
                case Type intType when intType == typeof(int):
                    PrimitiveDataFrameColumn<int> intColumn = this as PrimitiveDataFrameColumn<int>;
                    PrimitiveDataFrameColumn<int> newintColumn = inPlace ? intColumn : intColumn.Clone();
                    newintColumn._columnContainer.LeftShift(value);
                    return newintColumn;
                case Type longType when longType == typeof(long):
                    PrimitiveDataFrameColumn<long> longColumn = this as PrimitiveDataFrameColumn<long>;
                    PrimitiveDataFrameColumn<long> newlongColumn = inPlace ? longColumn : longColumn.Clone();
                    newlongColumn._columnContainer.LeftShift(value);
                    return newlongColumn;
                case Type sbyteType when sbyteType == typeof(sbyte):
                    PrimitiveDataFrameColumn<sbyte> sbyteColumn = this as PrimitiveDataFrameColumn<sbyte>;
                    PrimitiveDataFrameColumn<sbyte> newsbyteColumn = inPlace ? sbyteColumn : sbyteColumn.Clone();
                    newsbyteColumn._columnContainer.LeftShift(value);
                    return newsbyteColumn;
                case Type shortType when shortType == typeof(short):
                    PrimitiveDataFrameColumn<short> shortColumn = this as PrimitiveDataFrameColumn<short>;
                    PrimitiveDataFrameColumn<short> newshortColumn = inPlace ? shortColumn : shortColumn.Clone();
                    newshortColumn._columnContainer.LeftShift(value);
                    return newshortColumn;
                case Type uintType when uintType == typeof(uint):
                    PrimitiveDataFrameColumn<uint> uintColumn = this as PrimitiveDataFrameColumn<uint>;
                    PrimitiveDataFrameColumn<uint> newuintColumn = inPlace ? uintColumn : uintColumn.Clone();
                    newuintColumn._columnContainer.LeftShift(value);
                    return newuintColumn;
                case Type ulongType when ulongType == typeof(ulong):
                    PrimitiveDataFrameColumn<ulong> ulongColumn = this as PrimitiveDataFrameColumn<ulong>;
                    PrimitiveDataFrameColumn<ulong> newulongColumn = inPlace ? ulongColumn : ulongColumn.Clone();
                    newulongColumn._columnContainer.LeftShift(value);
                    return newulongColumn;
                case Type ushortType when ushortType == typeof(ushort):
                    PrimitiveDataFrameColumn<ushort> ushortColumn = this as PrimitiveDataFrameColumn<ushort>;
                    PrimitiveDataFrameColumn<ushort> newushortColumn = inPlace ? ushortColumn : ushortColumn.Clone();
                    newushortColumn._columnContainer.LeftShift(value);
                    return newushortColumn;
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn RightShiftImplementation(int value, bool inPlace)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                    PrimitiveDataFrameColumn<byte> byteColumn = this as PrimitiveDataFrameColumn<byte>;
                    PrimitiveDataFrameColumn<byte> newbyteColumn = inPlace ? byteColumn : byteColumn.Clone();
                    newbyteColumn._columnContainer.RightShift(value);
                    return newbyteColumn;
                case Type charType when charType == typeof(char):
                    PrimitiveDataFrameColumn<char> charColumn = this as PrimitiveDataFrameColumn<char>;
                    PrimitiveDataFrameColumn<char> newcharColumn = inPlace ? charColumn : charColumn.Clone();
                    newcharColumn._columnContainer.RightShift(value);
                    return newcharColumn;
                case Type decimalType when decimalType == typeof(decimal):
                    throw new NotSupportedException();
                case Type doubleType when doubleType == typeof(double):
                    throw new NotSupportedException();
                case Type floatType when floatType == typeof(float):
                    throw new NotSupportedException();
                case Type intType when intType == typeof(int):
                    PrimitiveDataFrameColumn<int> intColumn = this as PrimitiveDataFrameColumn<int>;
                    PrimitiveDataFrameColumn<int> newintColumn = inPlace ? intColumn : intColumn.Clone();
                    newintColumn._columnContainer.RightShift(value);
                    return newintColumn;
                case Type longType when longType == typeof(long):
                    PrimitiveDataFrameColumn<long> longColumn = this as PrimitiveDataFrameColumn<long>;
                    PrimitiveDataFrameColumn<long> newlongColumn = inPlace ? longColumn : longColumn.Clone();
                    newlongColumn._columnContainer.RightShift(value);
                    return newlongColumn;
                case Type sbyteType when sbyteType == typeof(sbyte):
                    PrimitiveDataFrameColumn<sbyte> sbyteColumn = this as PrimitiveDataFrameColumn<sbyte>;
                    PrimitiveDataFrameColumn<sbyte> newsbyteColumn = inPlace ? sbyteColumn : sbyteColumn.Clone();
                    newsbyteColumn._columnContainer.RightShift(value);
                    return newsbyteColumn;
                case Type shortType when shortType == typeof(short):
                    PrimitiveDataFrameColumn<short> shortColumn = this as PrimitiveDataFrameColumn<short>;
                    PrimitiveDataFrameColumn<short> newshortColumn = inPlace ? shortColumn : shortColumn.Clone();
                    newshortColumn._columnContainer.RightShift(value);
                    return newshortColumn;
                case Type uintType when uintType == typeof(uint):
                    PrimitiveDataFrameColumn<uint> uintColumn = this as PrimitiveDataFrameColumn<uint>;
                    PrimitiveDataFrameColumn<uint> newuintColumn = inPlace ? uintColumn : uintColumn.Clone();
                    newuintColumn._columnContainer.RightShift(value);
                    return newuintColumn;
                case Type ulongType when ulongType == typeof(ulong):
                    PrimitiveDataFrameColumn<ulong> ulongColumn = this as PrimitiveDataFrameColumn<ulong>;
                    PrimitiveDataFrameColumn<ulong> newulongColumn = inPlace ? ulongColumn : ulongColumn.Clone();
                    newulongColumn._columnContainer.RightShift(value);
                    return newulongColumn;
                case Type ushortType when ushortType == typeof(ushort):
                    PrimitiveDataFrameColumn<ushort> ushortColumn = this as PrimitiveDataFrameColumn<ushort>;
                    PrimitiveDataFrameColumn<ushort> newushortColumn = inPlace ? ushortColumn : ushortColumn.Clone();
                    newushortColumn._columnContainer.RightShift(value);
                    return newushortColumn;
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseEqualsImplementation<U>(PrimitiveDataFrameColumn<U> column)
            where U : unmanaged
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    if (typeof(U) != typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<U>)._columnContainer.ElementwiseEquals(column._columnContainer));
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseEquals(column._columnContainer));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseEquals(column.CloneAsDecimalColumn()._columnContainer));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    if (typeof(U) != typeof(DateTime))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<U>)._columnContainer.ElementwiseEquals(column._columnContainer));
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseEquals(column._columnContainer));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseEquals((column as PrimitiveDataFrameColumn<decimal>)._columnContainer));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseEquals(column.CloneAsDoubleColumn()._columnContainer));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseEqualsImplementation<U>(U value)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    if (typeof(U) != typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<bool>)._columnContainer.ElementwiseEquals(Unsafe.As<U, bool>(ref value)));
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseEquals(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseEquals(DecimalConverter<U>.Instance.GetDecimal(value)));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    if (typeof(U) != typeof(DateTime))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<DateTime>)._columnContainer.ElementwiseEquals(Unsafe.As<U, DateTime>(ref value)));
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseEquals(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseEquals(DecimalConverter<U>.Instance.GetDecimal(value)));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseEquals(DoubleConverter<U>.Instance.GetDouble(value)));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseNotEqualsImplementation<U>(PrimitiveDataFrameColumn<U> column)
            where U : unmanaged
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    if (typeof(U) != typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<U>)._columnContainer.ElementwiseNotEquals(column._columnContainer));
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseNotEquals(column._columnContainer));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseNotEquals(column.CloneAsDecimalColumn()._columnContainer));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    if (typeof(U) != typeof(DateTime))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<U>)._columnContainer.ElementwiseNotEquals(column._columnContainer));
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseNotEquals(column._columnContainer));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseNotEquals((column as PrimitiveDataFrameColumn<decimal>)._columnContainer));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseNotEquals(column.CloneAsDoubleColumn()._columnContainer));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseNotEqualsImplementation<U>(U value)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    if (typeof(U) != typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<bool>)._columnContainer.ElementwiseNotEquals(Unsafe.As<U, bool>(ref value)));
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseNotEquals(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseNotEquals(DecimalConverter<U>.Instance.GetDecimal(value)));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    if (typeof(U) != typeof(DateTime))
                    {
                        throw new NotSupportedException();
                    }
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<DateTime>)._columnContainer.ElementwiseNotEquals(Unsafe.As<U, DateTime>(ref value)));
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseNotEquals(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseNotEquals(DecimalConverter<U>.Instance.GetDecimal(value)));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseNotEquals(DoubleConverter<U>.Instance.GetDouble(value)));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanOrEqualImplementation<U>(PrimitiveDataFrameColumn<U> column)
            where U : unmanaged
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThanOrEqual(column._columnContainer));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThanOrEqual(column.CloneAsDecimalColumn()._columnContainer));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThanOrEqual(column._columnContainer));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThanOrEqual((column as PrimitiveDataFrameColumn<decimal>)._columnContainer));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseGreaterThanOrEqual(column.CloneAsDoubleColumn()._columnContainer));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanOrEqualImplementation<U>(U value)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThanOrEqual(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThanOrEqual(DecimalConverter<U>.Instance.GetDecimal(value)));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThanOrEqual(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThanOrEqual(DecimalConverter<U>.Instance.GetDecimal(value)));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseGreaterThanOrEqual(DoubleConverter<U>.Instance.GetDouble(value)));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseLessThanOrEqualImplementation<U>(PrimitiveDataFrameColumn<U> column)
            where U : unmanaged
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThanOrEqual(column._columnContainer));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThanOrEqual(column.CloneAsDecimalColumn()._columnContainer));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThanOrEqual(column._columnContainer));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThanOrEqual((column as PrimitiveDataFrameColumn<decimal>)._columnContainer));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseLessThanOrEqual(column.CloneAsDoubleColumn()._columnContainer));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseLessThanOrEqualImplementation<U>(U value)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThanOrEqual(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThanOrEqual(DecimalConverter<U>.Instance.GetDecimal(value)));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThanOrEqual(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThanOrEqual(DecimalConverter<U>.Instance.GetDecimal(value)));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseLessThanOrEqual(DoubleConverter<U>.Instance.GetDouble(value)));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanImplementation<U>(PrimitiveDataFrameColumn<U> column)
            where U : unmanaged
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThan(column._columnContainer));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThan(column.CloneAsDecimalColumn()._columnContainer));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThan(column._columnContainer));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThan((column as PrimitiveDataFrameColumn<decimal>)._columnContainer));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseGreaterThan(column.CloneAsDoubleColumn()._columnContainer));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseGreaterThanImplementation<U>(U value)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThan(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThan(DecimalConverter<U>.Instance.GetDecimal(value)));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseGreaterThan(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseGreaterThan(DecimalConverter<U>.Instance.GetDecimal(value)));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseGreaterThan(DoubleConverter<U>.Instance.GetDouble(value)));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseLessThanImplementation<U>(PrimitiveDataFrameColumn<U> column)
            where U : unmanaged
        {
            if (column.Length != Length)
            {
                throw new ArgumentException(Strings.MismatchedColumnLengths, nameof(column));
            }
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThan(column._columnContainer));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThan(column.CloneAsDecimalColumn()._columnContainer));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<U> primitiveColumn = this as PrimitiveDataFrameColumn<U>;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThan(column._columnContainer));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThan((column as PrimitiveDataFrameColumn<decimal>)._columnContainer));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseLessThan(column.CloneAsDoubleColumn()._columnContainer));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> ElementwiseLessThanImplementation<U>(U value)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    throw new NotSupportedException();
                case Type decimalType when decimalType == typeof(decimal):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThan(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThan(DecimalConverter<U>.Instance.GetDecimal(value)));
                    }
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                    throw new NotSupportedException();
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> primitiveColumn = this;
                        return new BooleanDataFrameColumn(Name, primitiveColumn._columnContainer.ElementwiseLessThan(Unsafe.As<U, T>(ref value)));
                    }
                    else
                    {
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            return new BooleanDataFrameColumn(Name, decimalColumn._columnContainer.ElementwiseLessThan(DecimalConverter<U>.Instance.GetDecimal(value)));
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            return new BooleanDataFrameColumn(Name, doubleColumn._columnContainer.ElementwiseLessThan(DoubleConverter<U>.Instance.GetDouble(value)));
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
    }
}
