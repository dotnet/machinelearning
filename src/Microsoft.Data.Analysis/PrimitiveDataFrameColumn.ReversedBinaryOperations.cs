
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
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneAsDecimalColumn();
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.ReverseAdd(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneAsDoubleColumn();
                            clonedDoubleColumn._columnContainer.ReverseAdd(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
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
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneAsDecimalColumn();
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.ReverseSubtract(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneAsDoubleColumn();
                            clonedDoubleColumn._columnContainer.ReverseSubtract(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
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
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneAsDecimalColumn();
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.ReverseMultiply(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneAsDoubleColumn();
                            clonedDoubleColumn._columnContainer.ReverseMultiply(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
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
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneAsDecimalColumn();
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.ReverseDivide(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneAsDoubleColumn();
                            clonedDoubleColumn._columnContainer.ReverseDivide(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
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
                        PrimitiveDataFrameColumn<decimal> clonedDecimalColumn = CloneAsDecimalColumn();
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
                            PrimitiveDataFrameColumn<decimal> decimalColumn = CloneAsDecimalColumn();
                            decimalColumn._columnContainer.ReverseModulo(DecimalConverter<U>.Instance.GetDecimal(value));
                            return decimalColumn;
                        }
                        else
                        {
                            PrimitiveDataFrameColumn<double> clonedDoubleColumn = CloneAsDoubleColumn();
                            clonedDoubleColumn._columnContainer.ReverseModulo(DoubleConverter<U>.Instance.GetDouble(value));
                            return clonedDoubleColumn;
                        }
                    }
                default:
                    throw new NotSupportedException();
            }
        }
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
