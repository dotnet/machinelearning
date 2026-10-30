
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameBinaryOperators.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    public partial class DataFrame
    {
#pragma warning disable 1591
        public static DataFrame operator +(DataFrame df, byte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(byte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, decimal value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(decimal value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, double value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(double value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, float value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(float value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, int value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(int value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, long value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(long value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, sbyte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(sbyte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, short value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(short value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, uint value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(uint value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, ulong value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(ulong value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator +(DataFrame df, ushort value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Add(value);
        }

        public static DataFrame operator +(ushort value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseAdd(value);
        }

        public static DataFrame operator -(DataFrame df, byte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(byte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, decimal value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(decimal value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, double value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(double value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, float value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(float value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, int value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(int value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, long value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(long value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, sbyte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(sbyte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, short value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(short value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, uint value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(uint value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, ulong value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(ulong value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator -(DataFrame df, ushort value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Subtract(value);
        }

        public static DataFrame operator -(ushort value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseSubtract(value);
        }

        public static DataFrame operator *(DataFrame df, byte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(byte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, decimal value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(decimal value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, double value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(double value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, float value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(float value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, int value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(int value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, long value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(long value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, sbyte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(sbyte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, short value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(short value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, uint value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(uint value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, ulong value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(ulong value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator *(DataFrame df, ushort value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Multiply(value);
        }

        public static DataFrame operator *(ushort value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseMultiply(value);
        }

        public static DataFrame operator /(DataFrame df, byte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(byte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, decimal value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(decimal value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, double value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(double value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, float value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(float value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, int value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(int value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, long value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(long value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, sbyte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(sbyte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, short value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(short value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, uint value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(uint value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, ulong value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(ulong value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator /(DataFrame df, ushort value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Divide(value);
        }

        public static DataFrame operator /(ushort value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseDivide(value);
        }

        public static DataFrame operator %(DataFrame df, byte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(byte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, decimal value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(decimal value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, double value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(double value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, float value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(float value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, int value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(int value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, long value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(long value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, sbyte value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(sbyte value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, short value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(short value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, uint value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(uint value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, ulong value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(ulong value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator %(DataFrame df, ushort value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.Modulo(value);
        }

        public static DataFrame operator %(ushort value, DataFrame df)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.ReverseModulo(value);
        }

        public static DataFrame operator <<(DataFrame df, int value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.LeftShift(value);
        }

        public static DataFrame operator >>(DataFrame df, int value)
        {
            if (df == null)
            {
                throw new ArgumentNullException(nameof(df));
            }

            return df.RightShift(value);
        }

    }
}
