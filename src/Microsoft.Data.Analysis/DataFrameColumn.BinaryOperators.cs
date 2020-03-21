
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameColumn.BinaryOperators.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public abstract partial class DataFrameColumn
    {
        #pragma warning disable 1591
        public static DataFrameColumn operator +(DataFrameColumn left, DataFrameColumn right)
        {
            return left.Add(right);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, byte value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(byte value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, decimal value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(decimal value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, double value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(double value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, float value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(float value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, int value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(int value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, long value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(long value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, sbyte value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(sbyte value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, short value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(short value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, uint value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(uint value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, ulong value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(ulong value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }

        public static DataFrameColumn operator +(DataFrameColumn column, ushort value)
        {
            return column.Add(value);
        }

        public static DataFrameColumn operator +(ushort value, DataFrameColumn column)
        {
            return column.ReverseAdd(value);
        }


        public static DataFrameColumn operator -(DataFrameColumn left, DataFrameColumn right)
        {
            return left.Subtract(right);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, byte value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(byte value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, decimal value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(decimal value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, double value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(double value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, float value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(float value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, int value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(int value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, long value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(long value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, sbyte value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(sbyte value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, short value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(short value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, uint value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(uint value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, ulong value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(ulong value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }

        public static DataFrameColumn operator -(DataFrameColumn column, ushort value)
        {
            return column.Subtract(value);
        }

        public static DataFrameColumn operator -(ushort value, DataFrameColumn column)
        {
            return column.ReverseSubtract(value);
        }


        public static DataFrameColumn operator *(DataFrameColumn left, DataFrameColumn right)
        {
            return left.Multiply(right);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, byte value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(byte value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, decimal value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(decimal value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, double value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(double value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, float value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(float value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, int value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(int value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, long value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(long value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, sbyte value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(sbyte value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, short value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(short value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, uint value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(uint value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, ulong value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(ulong value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }

        public static DataFrameColumn operator *(DataFrameColumn column, ushort value)
        {
            return column.Multiply(value);
        }

        public static DataFrameColumn operator *(ushort value, DataFrameColumn column)
        {
            return column.ReverseMultiply(value);
        }


        public static DataFrameColumn operator /(DataFrameColumn left, DataFrameColumn right)
        {
            return left.Divide(right);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, byte value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(byte value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, decimal value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(decimal value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, double value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(double value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, float value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(float value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, int value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(int value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, long value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(long value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, sbyte value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(sbyte value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, short value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(short value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, uint value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(uint value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, ulong value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(ulong value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }

        public static DataFrameColumn operator /(DataFrameColumn column, ushort value)
        {
            return column.Divide(value);
        }

        public static DataFrameColumn operator /(ushort value, DataFrameColumn column)
        {
            return column.ReverseDivide(value);
        }


        public static DataFrameColumn operator %(DataFrameColumn left, DataFrameColumn right)
        {
            return left.Modulo(right);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, byte value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(byte value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, decimal value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(decimal value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, double value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(double value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, float value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(float value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, int value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(int value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, long value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(long value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, sbyte value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(sbyte value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, short value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(short value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, uint value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(uint value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, ulong value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(ulong value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }

        public static DataFrameColumn operator %(DataFrameColumn column, ushort value)
        {
            return column.Modulo(value);
        }

        public static DataFrameColumn operator %(ushort value, DataFrameColumn column)
        {
            return column.ReverseModulo(value);
        }


        public static DataFrameColumn operator &(DataFrameColumn left, DataFrameColumn right)
        {
            return left.And(right);
        }

        public static DataFrameColumn operator &(DataFrameColumn column, bool value)
        {
            return column.And(value);
        }

        public static DataFrameColumn operator &(bool value, DataFrameColumn column)
        {
            return column.ReverseAnd(value);
        }

        public static DataFrameColumn operator |(DataFrameColumn left, DataFrameColumn right)
        {
            return left.Or(right);
        }

        public static DataFrameColumn operator |(DataFrameColumn column, bool value)
        {
            return column.Or(value);
        }

        public static DataFrameColumn operator |(bool value, DataFrameColumn column)
        {
            return column.ReverseOr(value);
        }

        public static DataFrameColumn operator ^(DataFrameColumn left, DataFrameColumn right)
        {
            return left.Xor(right);
        }

        public static DataFrameColumn operator ^(DataFrameColumn column, bool value)
        {
            return column.Xor(value);
        }

        public static DataFrameColumn operator ^(bool value, DataFrameColumn column)
        {
            return column.ReverseXor(value);
        }

        public static DataFrameColumn operator <<(DataFrameColumn column, int value)
        {
            return column.LeftShift(value);
        }

        public static DataFrameColumn operator >>(DataFrameColumn column, int value)
        {
            return column.RightShift(value);
        }

    }
}
