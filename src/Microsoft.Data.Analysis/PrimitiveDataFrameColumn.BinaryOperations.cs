
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
                    return AddImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return AddImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return AddImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return AddImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return AddImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return AddImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return AddImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return AddImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return AddImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return AddImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return AddImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return AddImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return AddImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return AddImplementation(DateTimeColumn, inPlace);

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
            return AddImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Subtract(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return SubtractImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return SubtractImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return SubtractImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return SubtractImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return SubtractImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return SubtractImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return SubtractImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return SubtractImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return SubtractImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return SubtractImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return SubtractImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return SubtractImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return SubtractImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return SubtractImplementation(DateTimeColumn, inPlace);

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
                return Subtract(column, inPlace);
            }
            return SubtractImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Multiply(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return MultiplyImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return MultiplyImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return MultiplyImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return MultiplyImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return MultiplyImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return MultiplyImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return MultiplyImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return MultiplyImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return MultiplyImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return MultiplyImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return MultiplyImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return MultiplyImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return MultiplyImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return MultiplyImplementation(DateTimeColumn, inPlace);

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
                return Multiply(column, inPlace);
            }
            return MultiplyImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Divide(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return DivideImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return DivideImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return DivideImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return DivideImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return DivideImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return DivideImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return DivideImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return DivideImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return DivideImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return DivideImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return DivideImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return DivideImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return DivideImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return DivideImplementation(DateTimeColumn, inPlace);

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
                return Divide(column, inPlace);
            }
            return DivideImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Modulo(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return ModuloImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return ModuloImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return ModuloImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return ModuloImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return ModuloImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return ModuloImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return ModuloImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return ModuloImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return ModuloImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return ModuloImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return ModuloImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return ModuloImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return ModuloImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return ModuloImplementation(DateTimeColumn, inPlace);

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
                return Modulo(column, inPlace);
            }
            return ModuloImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn And(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return AndImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return AndImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return AndImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return AndImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return AndImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return AndImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return AndImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return AndImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return AndImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return AndImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return AndImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return AndImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return AndImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return AndImplementation(DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> And(bool value, bool inPlace = false)
        {
            return AndImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Or(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return OrImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return OrImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return OrImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return OrImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return OrImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return OrImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return OrImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return OrImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return OrImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return OrImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return OrImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return OrImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return OrImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return OrImplementation(DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> Or(bool value, bool inPlace = false)
        {
            return OrImplementation(value, inPlace);
        }

        /// <inheritdoc/>
        public override DataFrameColumn Xor(DataFrameColumn column, bool inPlace = false)
        {
            switch (column)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    return XorImplementation(boolColumn, inPlace);
                case PrimitiveDataFrameColumn<byte> byteColumn:
                    return XorImplementation(byteColumn, inPlace);
                case PrimitiveDataFrameColumn<char> charColumn:
                    return XorImplementation(charColumn, inPlace);
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    return XorImplementation(decimalColumn, inPlace);
                case PrimitiveDataFrameColumn<double> doubleColumn:
                    return XorImplementation(doubleColumn, inPlace);
                case PrimitiveDataFrameColumn<float> floatColumn:
                    return XorImplementation(floatColumn, inPlace);
                case PrimitiveDataFrameColumn<int> intColumn:
                    return XorImplementation(intColumn, inPlace);
                case PrimitiveDataFrameColumn<long> longColumn:
                    return XorImplementation(longColumn, inPlace);
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                    return XorImplementation(sbyteColumn, inPlace);
                case PrimitiveDataFrameColumn<short> shortColumn:
                    return XorImplementation(shortColumn, inPlace);
                case PrimitiveDataFrameColumn<uint> uintColumn:
                    return XorImplementation(uintColumn, inPlace);
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                    return XorImplementation(ulongColumn, inPlace);
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                    return XorImplementation(ushortColumn, inPlace);
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    return XorImplementation(DateTimeColumn, inPlace);

                default:
                    throw new NotSupportedException();
            }
        }

        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> Xor(bool value, bool inPlace = false)
        {
            return XorImplementation(value, inPlace);
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


        internal DataFrameColumn AddImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Add(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Add(column.CloneAsDecimalColumn()._columnContainer);
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Add(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Add((column as PrimitiveDataFrameColumn<decimal>)._columnContainer);
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Add(column.CloneAsDoubleColumn()._columnContainer);
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn AddImplementation<U>(U value, bool inPlace)
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Add(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Add(DecimalConverter<U>.Instance.GetDecimal(value));
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Add(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Add(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Add(DoubleConverter<U>.Instance.GetDouble(value));
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn SubtractImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Subtract(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Subtract(column.CloneAsDecimalColumn()._columnContainer);
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Subtract(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Subtract((column as PrimitiveDataFrameColumn<decimal>)._columnContainer);
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Subtract(column.CloneAsDoubleColumn()._columnContainer);
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn SubtractImplementation<U>(U value, bool inPlace)
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Subtract(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Subtract(DecimalConverter<U>.Instance.GetDecimal(value));
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Subtract(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Subtract(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Subtract(DoubleConverter<U>.Instance.GetDouble(value));
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn MultiplyImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Multiply(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Multiply(column.CloneAsDecimalColumn()._columnContainer);
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Multiply(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Multiply((column as PrimitiveDataFrameColumn<decimal>)._columnContainer);
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Multiply(column.CloneAsDoubleColumn()._columnContainer);
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn MultiplyImplementation<U>(U value, bool inPlace)
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Multiply(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Multiply(DecimalConverter<U>.Instance.GetDecimal(value));
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Multiply(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Multiply(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Multiply(DoubleConverter<U>.Instance.GetDouble(value));
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn DivideImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Divide(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Divide(column.CloneAsDecimalColumn()._columnContainer);
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Divide(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Divide((column as PrimitiveDataFrameColumn<decimal>)._columnContainer);
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Divide(column.CloneAsDoubleColumn()._columnContainer);
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn DivideImplementation<U>(U value, bool inPlace)
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Divide(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Divide(DecimalConverter<U>.Instance.GetDecimal(value));
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Divide(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Divide(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Divide(DoubleConverter<U>.Instance.GetDouble(value));
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn ModuloImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Modulo(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Modulo(column.CloneAsDecimalColumn()._columnContainer);
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<U> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Modulo(column._columnContainer);
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedColumnValueType, typeof(T)), nameof(column));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Modulo((column as PrimitiveDataFrameColumn<decimal>)._columnContainer);
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Modulo(column.CloneAsDoubleColumn()._columnContainer);
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn ModuloImplementation<U>(U value, bool inPlace)
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Modulo(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                        decimalColumn._columnContainer.Modulo(DecimalConverter<U>.Instance.GetDecimal(value));
                        return decimalColumn;
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
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? primitiveColumn : primitiveColumn.Clone();
                        newColumn._columnContainer.Modulo(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        if (typeof(U) == typeof(decimal))
                        {
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.Modulo(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> doubleColumn = CloneAsDoubleColumn();
                            doubleColumn._columnContainer.Modulo(DoubleConverter<U>.Instance.GetDouble(value));
                            return doubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn AndImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                    PrimitiveDataFrameColumn<U> typedColumn = this as PrimitiveDataFrameColumn<U>;
                    PrimitiveDataFrameColumn<U> retColumn = inPlace ? typedColumn : typedColumn.Clone();
                    retColumn._columnContainer.And(column._columnContainer);
                    return retColumn;
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type decimalType when decimalType == typeof(decimal):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> AndImplementation<U>(U value, bool inPlace)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    if (typeof(U) != typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    PrimitiveDataFrameColumn<bool> typedColumn = this as PrimitiveDataFrameColumn<bool>;
                    PrimitiveDataFrameColumn<bool> retColumn = inPlace ? typedColumn : typedColumn.Clone();
                    retColumn._columnContainer.And(Unsafe.As<U, bool>(ref value));
                    return retColumn as PrimitiveDataFrameColumn<bool>;
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type decimalType when decimalType == typeof(decimal):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn OrImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                    PrimitiveDataFrameColumn<U> typedColumn = this as PrimitiveDataFrameColumn<U>;
                    PrimitiveDataFrameColumn<U> retColumn = inPlace ? typedColumn : typedColumn.Clone();
                    retColumn._columnContainer.Or(column._columnContainer);
                    return retColumn;
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type decimalType when decimalType == typeof(decimal):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> OrImplementation<U>(U value, bool inPlace)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    if (typeof(U) != typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    PrimitiveDataFrameColumn<bool> typedColumn = this as PrimitiveDataFrameColumn<bool>;
                    PrimitiveDataFrameColumn<bool> retColumn = inPlace ? typedColumn : typedColumn.Clone();
                    retColumn._columnContainer.Or(Unsafe.As<U, bool>(ref value));
                    return retColumn as PrimitiveDataFrameColumn<bool>;
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type decimalType when decimalType == typeof(decimal):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                default:
                    throw new NotSupportedException();
            }
        }

        internal DataFrameColumn XorImplementation<U>(PrimitiveDataFrameColumn<U> column, bool inPlace)
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
                    PrimitiveDataFrameColumn<U> typedColumn = this as PrimitiveDataFrameColumn<U>;
                    PrimitiveDataFrameColumn<U> retColumn = inPlace ? typedColumn : typedColumn.Clone();
                    retColumn._columnContainer.Xor(column._columnContainer);
                    return retColumn;
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type decimalType when decimalType == typeof(decimal):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                default:
                    throw new NotSupportedException();
            }
        }

        internal PrimitiveDataFrameColumn<bool> XorImplementation<U>(U value, bool inPlace)
        {
            switch (typeof(T))
            {
                case Type boolType when boolType == typeof(bool):
                    if (typeof(U) != typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    PrimitiveDataFrameColumn<bool> typedColumn = this as PrimitiveDataFrameColumn<bool>;
                    PrimitiveDataFrameColumn<bool> retColumn = inPlace ? typedColumn : typedColumn.Clone();
                    retColumn._columnContainer.Xor(Unsafe.As<U, bool>(ref value));
                    return retColumn as PrimitiveDataFrameColumn<bool>;
                case Type byteType when byteType == typeof(byte):
                case Type charType when charType == typeof(char):
                case Type decimalType when decimalType == typeof(decimal):
                case Type doubleType when doubleType == typeof(double):
                case Type floatType when floatType == typeof(float):
                case Type intType when intType == typeof(int):
                case Type longType when longType == typeof(long):
                case Type sbyteType when sbyteType == typeof(sbyte):
                case Type shortType when shortType == typeof(short):
                case Type uintType when uintType == typeof(uint):
                case Type ulongType when ulongType == typeof(ulong):
                case Type ushortType when ushortType == typeof(ushort):
                case Type DateTimeType when DateTimeType == typeof(DateTime):
                default:
                    throw new NotSupportedException();
            }
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
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<bool>)._columnContainer.ElementwiseEquals(Unsafe.As<U, bool>(ref value)));
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
                    return new BooleanDataFrameColumn(Name, (this as PrimitiveDataFrameColumn<bool>)._columnContainer.ElementwiseNotEquals(Unsafe.As<U, bool>(ref value)));
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
