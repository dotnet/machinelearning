
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumn.ReversedBinaryOperations.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Microsoft.Data.Analysis
{
    public partial class PrimitiveDataFrameColumn<T> : DataFrameColumn
        where T : unmanaged
    {

        /// <inheritdoc/>
        public override DataFrameColumn ReverseAdd<U>(U value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    throw new NotSupportedException();
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseAdd(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneTruncating<decimal>();
                        clonedDecimalColumn._columnContainer.ReverseAdd(DecimalConverter<U>.Instance.GetDecimal(value));
                        return clonedDecimalColumn;
                    }
                case PrimitiveDataFrameColumn<byte> byteColumn:
                case PrimitiveDataFrameColumn<char> charColumn:
                case PrimitiveDataFrameColumn<double> doubleColumn:
                case PrimitiveDataFrameColumn<float> floatColumn:
                case PrimitiveDataFrameColumn<int> intColumn:
                case PrimitiveDataFrameColumn<long> longColumn:
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                case PrimitiveDataFrameColumn<short> shortColumn:
                case PrimitiveDataFrameColumn<uint> uintColumn:
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseAdd(Unsafe.As<U, T>(ref value));
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneTruncating<decimal>();
                            decimalColumn._columnContainer.ReverseAdd(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneTruncating<double>();
                            clonedDoubleColumn._columnContainer.ReverseAdd(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn ReverseSubtract<U>(U value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    throw new NotSupportedException();
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseSubtract(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneTruncating<decimal>();
                        clonedDecimalColumn._columnContainer.ReverseSubtract(DecimalConverter<U>.Instance.GetDecimal(value));
                        return clonedDecimalColumn;
                    }
                case PrimitiveDataFrameColumn<byte> byteColumn:
                case PrimitiveDataFrameColumn<char> charColumn:
                case PrimitiveDataFrameColumn<double> doubleColumn:
                case PrimitiveDataFrameColumn<float> floatColumn:
                case PrimitiveDataFrameColumn<int> intColumn:
                case PrimitiveDataFrameColumn<long> longColumn:
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                case PrimitiveDataFrameColumn<short> shortColumn:
                case PrimitiveDataFrameColumn<uint> uintColumn:
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseSubtract(Unsafe.As<U, T>(ref value));
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneTruncating<decimal>();
                            decimalColumn._columnContainer.ReverseSubtract(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneTruncating<double>();
                            clonedDoubleColumn._columnContainer.ReverseSubtract(DoubleConverter<U>.Instance.GetDouble(value));

                            switch (typeof(U))
                            {
                                case Type byteType2 when byteType2 == typeof(byte):
                                    return clonedDoubleColumn.CloneTruncating<byte>();
                                case Type charType2 when charType2 == typeof(char):
                                    return clonedDoubleColumn.CloneTruncating<char>();
                                case Type doubleType2 when doubleType2 == typeof(double):
                                    return clonedDoubleColumn.CloneTruncating<double>();
                                case Type floatType2 when floatType2 == typeof(float):
                                    return clonedDoubleColumn.CloneTruncating<float>();
                                case Type intType2 when intType2 == typeof(int):
                                    return clonedDoubleColumn.CloneTruncating<int>();
                                case Type longType2 when longType2 == typeof(long):
                                    return clonedDoubleColumn.CloneTruncating<long>();
                                case Type sbyteType2 when sbyteType2 == typeof(sbyte):
                                    return clonedDoubleColumn.CloneTruncating<sbyte>();
                                case Type shortType2 when shortType2 == typeof(short):
                                    return clonedDoubleColumn.CloneTruncating<short>();
                                case Type uintType2 when uintType2 == typeof(uint):
                                    return clonedDoubleColumn.CloneTruncating<uint>();
                                case Type ulongType2 when ulongType2 == typeof(ulong):
                                    return clonedDoubleColumn.CloneTruncating<ulong>();
                                case Type ushortType2 when ushortType2 == typeof(ushort):
                                    return clonedDoubleColumn.CloneTruncating<ushort>();
                                default:
                                    throw new NotSupportedException();
                            }
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn ReverseMultiply<U>(U value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    throw new NotSupportedException();
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseMultiply(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneTruncating<decimal>();
                        clonedDecimalColumn._columnContainer.ReverseMultiply(DecimalConverter<U>.Instance.GetDecimal(value));
                        return clonedDecimalColumn;
                    }
                case PrimitiveDataFrameColumn<byte> byteColumn:
                case PrimitiveDataFrameColumn<char> charColumn:
                case PrimitiveDataFrameColumn<double> doubleColumn:
                case PrimitiveDataFrameColumn<float> floatColumn:
                case PrimitiveDataFrameColumn<int> intColumn:
                case PrimitiveDataFrameColumn<long> longColumn:
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                case PrimitiveDataFrameColumn<short> shortColumn:
                case PrimitiveDataFrameColumn<uint> uintColumn:
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseMultiply(Unsafe.As<U, T>(ref value));
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneTruncating<decimal>();
                            decimalColumn._columnContainer.ReverseMultiply(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneTruncating<double>();
                            clonedDoubleColumn._columnContainer.ReverseMultiply(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn ReverseDivide<U>(U value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    throw new NotSupportedException();
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseDivide(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneTruncating<decimal>();
                        clonedDecimalColumn._columnContainer.ReverseDivide(DecimalConverter<U>.Instance.GetDecimal(value));
                        return clonedDecimalColumn;
                    }
                case PrimitiveDataFrameColumn<byte> byteColumn:
                case PrimitiveDataFrameColumn<char> charColumn:
                case PrimitiveDataFrameColumn<double> doubleColumn:
                case PrimitiveDataFrameColumn<float> floatColumn:
                case PrimitiveDataFrameColumn<int> intColumn:
                case PrimitiveDataFrameColumn<long> longColumn:
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                case PrimitiveDataFrameColumn<short> shortColumn:
                case PrimitiveDataFrameColumn<uint> uintColumn:
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseDivide(Unsafe.As<U, T>(ref value));
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneTruncating<decimal>();
                            decimalColumn._columnContainer.ReverseDivide(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {

                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneTruncating<double>();
                            clonedDoubleColumn._columnContainer.ReverseDivide(DoubleConverter<U>.Instance.GetDouble(value));

                            switch (typeof(U))
                            {
                                case Type byteType2 when byteType2 == typeof(byte):
                                    return clonedDoubleColumn.CloneTruncating<byte>();
                                case Type charType2 when charType2 == typeof(char):
                                    return clonedDoubleColumn.CloneTruncating<char>();
                                case Type doubleType2 when doubleType2 == typeof(double):
                                    return clonedDoubleColumn.CloneTruncating<double>();
                                case Type floatType2 when floatType2 == typeof(float):
                                    return clonedDoubleColumn.CloneTruncating<float>();
                                case Type intType2 when intType2 == typeof(int):
                                    return clonedDoubleColumn.CloneTruncating<int>();
                                case Type longType2 when longType2 == typeof(long):
                                    return clonedDoubleColumn.CloneTruncating<long>();
                                case Type sbyteType2 when sbyteType2 == typeof(sbyte):
                                    return clonedDoubleColumn.CloneTruncating<sbyte>();
                                case Type shortType2 when shortType2 == typeof(short):
                                    return clonedDoubleColumn.CloneTruncating<short>();
                                case Type uintType2 when uintType2 == typeof(uint):
                                    return clonedDoubleColumn.CloneTruncating<uint>();
                                case Type ulongType2 when ulongType2 == typeof(ulong):
                                    return clonedDoubleColumn.CloneTruncating<ulong>();
                                case Type ushortType2 when ushortType2 == typeof(ushort):
                                    return clonedDoubleColumn.CloneTruncating<ushort>();
                                default:
                                    throw new NotSupportedException();
                            }
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override DataFrameColumn ReverseModulo<U>(U value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    throw new NotSupportedException();
                case PrimitiveDataFrameColumn<decimal> decimalColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseModulo(Unsafe.As<U, T>(ref value));
                        return newColumn;
                    }
                    else
                    {
                        if (inPlace)
                        {
                            throw new ArgumentException(string.Format(Strings.MismatchedValueType, typeof(T)), nameof(value));
                        }
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneTruncating<decimal>();
                        clonedDecimalColumn._columnContainer.ReverseModulo(DecimalConverter<U>.Instance.GetDecimal(value));
                        return clonedDecimalColumn;
                    }
                case PrimitiveDataFrameColumn<byte> byteColumn:
                case PrimitiveDataFrameColumn<char> charColumn:
                case PrimitiveDataFrameColumn<double> doubleColumn:
                case PrimitiveDataFrameColumn<float> floatColumn:
                case PrimitiveDataFrameColumn<int> intColumn:
                case PrimitiveDataFrameColumn<long> longColumn:
                case PrimitiveDataFrameColumn<sbyte> sbyteColumn:
                case PrimitiveDataFrameColumn<short> shortColumn:
                case PrimitiveDataFrameColumn<uint> uintColumn:
                case PrimitiveDataFrameColumn<ulong> ulongColumn:
                case PrimitiveDataFrameColumn<ushort> ushortColumn:
                case PrimitiveDataFrameColumn<DateTime> DateTimeColumn:
                    if (typeof(U) == typeof(bool))
                    {
                        throw new NotSupportedException();
                    }
                    if (typeof(U) == typeof(T))
                    {
                        // No conversions
                        PrimitiveDataFrameColumn<T> newColumn = inPlace ? this : Clone();
                        newColumn._columnContainer.ReverseModulo(Unsafe.As<U, T>(ref value));
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneTruncating<decimal>();
                            decimalColumn._columnContainer.ReverseModulo(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneTruncating<double>();
                            clonedDoubleColumn._columnContainer.ReverseModulo(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ReverseAnd(bool value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    PrimitiveDataFrameColumn<bool> retColumn = inPlace ? boolColumn : boolColumn.Clone();
                    retColumn._columnContainer.ReverseAnd(value);
                    return retColumn;
                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ReverseOr(bool value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    PrimitiveDataFrameColumn<bool> retColumn = inPlace ? boolColumn : boolColumn.Clone();
                    retColumn._columnContainer.ReverseOr(value);
                    return retColumn;
                default:
                    throw new NotSupportedException();
            }
        }
        /// <inheritdoc/>
        public override PrimitiveDataFrameColumn<bool> ReverseXor(bool value, bool inPlace = false)
        {
            switch (this)
            {
                case PrimitiveDataFrameColumn<bool> boolColumn:
                    PrimitiveDataFrameColumn<bool> retColumn = inPlace ? boolColumn : boolColumn.Clone();
                    retColumn._columnContainer.ReverseXor(value);
                    return retColumn;
                default:
                    throw new NotSupportedException();
            }
        }
    }
}
