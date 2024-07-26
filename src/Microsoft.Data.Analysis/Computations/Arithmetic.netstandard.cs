
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from Arithmetic.tt. Do not modify directly

#if !NET8_0_OR_GREATER

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Microsoft.Data.Analysis
{
    ////////////////////////////////////////
    //Factory Class                       //
    ////////////////////////////////////////

    internal static partial class Arithmetic
    {
        #region Nested classes for Arithmetics

        internal class BoolArithmetic : Arithmetic<bool>
        {

            protected override void And(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] & y);
                    i++;
                }
            }

            protected override void And(bool x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (bool)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(bool x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (bool)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (bool)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(bool x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (bool)(x ^ y[i]);
                    i++;
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<bool> x, bool y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }
        }
        internal class ByteArithmetic : Arithmetic<byte>
        {

            protected override void Add(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref byte xRef = ref MemoryMarshal.GetReference(x);
                    ref byte yRef = ref MemoryMarshal.GetReference(y);
                    ref byte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<byte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref byte xRef = ref MemoryMarshal.GetReference(x);
                    ref byte yRef = ref MemoryMarshal.GetReference(y);
                    ref byte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<byte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref byte xRef = ref MemoryMarshal.GetReference(x);
                    ref byte yRef = ref MemoryMarshal.GetReference(y);
                    ref byte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<byte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x * y[i]);
                    i++;
                }
            }

            protected override byte Divide(byte x, byte y)
            {
                return (byte)(x / y);
            }

            protected override void Divide(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref byte xRef = ref MemoryMarshal.GetReference(x);
                    ref byte yRef = ref MemoryMarshal.GetReference(y);
                    ref byte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<byte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x / y[i]);
                    i++;
                }
            }

            protected override byte Modulo(byte x, byte y)
            {
                return (byte)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] & y);
                    i++;
                }
            }

            protected override void And(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<byte> x, byte y, Span<byte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (byte)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(byte x, ReadOnlySpan<byte> y, Span<byte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (byte)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<byte> x, int y, Span<byte> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (byte)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<byte> x, int y, Span<byte> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (byte)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<byte> x, byte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<byte> x, byte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<byte> x, byte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<byte> x, byte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<byte> x, byte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<byte> x, byte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class CharArithmetic : Arithmetic<char>
        {

            protected override void Add(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref char xRef = ref MemoryMarshal.GetReference(x);
                    ref char yRef = ref MemoryMarshal.GetReference(y);
                    ref char dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<char>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref char xRef = ref MemoryMarshal.GetReference(x);
                    ref char yRef = ref MemoryMarshal.GetReference(y);
                    ref char dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<char>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref char xRef = ref MemoryMarshal.GetReference(x);
                    ref char yRef = ref MemoryMarshal.GetReference(y);
                    ref char dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<char>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x * y[i]);
                    i++;
                }
            }

            protected override char Divide(char x, char y)
            {
                return (char)(x / y);
            }

            protected override void Divide(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref char xRef = ref MemoryMarshal.GetReference(x);
                    ref char yRef = ref MemoryMarshal.GetReference(y);
                    ref char dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<char>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x / y[i]);
                    i++;
                }
            }

            protected override char Modulo(char x, char y)
            {
                return (char)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] & y);
                    i++;
                }
            }

            protected override void And(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<char> x, char y, Span<char> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (char)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(char x, ReadOnlySpan<char> y, Span<char> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (char)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<char> x, int y, Span<char> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (char)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<char> x, int y, Span<char> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (char)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<char> x, char y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<char> x, char y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<char> x, char y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<char> x, char y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<char> x, char y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<char> x, ReadOnlySpan<char> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<char> x, char y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class DecimalArithmetic : Arithmetic<decimal>
        {

            protected override void Add(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<decimal> x, decimal y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(decimal x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (decimal)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<decimal> x, decimal y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(decimal x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (decimal)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<decimal> x, decimal y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(decimal x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (decimal)(x * y[i]);
                    i++;
                }
            }

            protected override decimal Divide(decimal x, decimal y)
            {
                return (decimal)(x / y);
            }

            protected override void Divide(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<decimal> x, decimal y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(decimal x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (decimal)(x / y[i]);
                    i++;
                }
            }

            protected override decimal Modulo(decimal x, decimal y)
            {
                return (decimal)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<decimal> x, decimal y, Span<decimal> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (decimal)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(decimal x, ReadOnlySpan<decimal> y, Span<decimal> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (decimal)(x % y[i]);
                    i++;
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<decimal> x, decimal y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<decimal> x, decimal y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<decimal> x, decimal y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<decimal> x, decimal y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<decimal> x, decimal y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<decimal> x, decimal y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class DoubleArithmetic : Arithmetic<double>
        {

            protected override void Add(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref double xRef = ref MemoryMarshal.GetReference(x);
                    ref double yRef = ref MemoryMarshal.GetReference(y);
                    ref double dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<double>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<double> x, double y, Span<double> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(double x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (double)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref double xRef = ref MemoryMarshal.GetReference(x);
                    ref double yRef = ref MemoryMarshal.GetReference(y);
                    ref double dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<double>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<double> x, double y, Span<double> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(double x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (double)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref double xRef = ref MemoryMarshal.GetReference(x);
                    ref double yRef = ref MemoryMarshal.GetReference(y);
                    ref double dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<double>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<double> x, double y, Span<double> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(double x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (double)(x * y[i]);
                    i++;
                }
            }

            protected override double Divide(double x, double y)
            {
                return (double)(x / y);
            }

            protected override void Divide(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref double xRef = ref MemoryMarshal.GetReference(x);
                    ref double yRef = ref MemoryMarshal.GetReference(y);
                    ref double dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<double>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<double> x, double y, Span<double> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(double x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (double)(x / y[i]);
                    i++;
                }
            }

            protected override double Modulo(double x, double y)
            {
                return (double)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<double> x, double y, Span<double> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (double)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(double x, ReadOnlySpan<double> y, Span<double> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (double)(x % y[i]);
                    i++;
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<double> x, double y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<double> x, double y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<double> x, double y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<double> x, double y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<double> x, double y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<double> x, double y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class FloatArithmetic : Arithmetic<float>
        {

            protected override void Add(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref float xRef = ref MemoryMarshal.GetReference(x);
                    ref float yRef = ref MemoryMarshal.GetReference(y);
                    ref float dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<float>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<float> x, float y, Span<float> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(float x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (float)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref float xRef = ref MemoryMarshal.GetReference(x);
                    ref float yRef = ref MemoryMarshal.GetReference(y);
                    ref float dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<float>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<float> x, float y, Span<float> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(float x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (float)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref float xRef = ref MemoryMarshal.GetReference(x);
                    ref float yRef = ref MemoryMarshal.GetReference(y);
                    ref float dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<float>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<float> x, float y, Span<float> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(float x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (float)(x * y[i]);
                    i++;
                }
            }

            protected override float Divide(float x, float y)
            {
                return (float)(x / y);
            }

            protected override void Divide(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref float xRef = ref MemoryMarshal.GetReference(x);
                    ref float yRef = ref MemoryMarshal.GetReference(y);
                    ref float dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<float>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<float> x, float y, Span<float> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(float x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (float)(x / y[i]);
                    i++;
                }
            }

            protected override float Modulo(float x, float y)
            {
                return (float)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<float> x, float y, Span<float> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (float)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(float x, ReadOnlySpan<float> y, Span<float> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (float)(x % y[i]);
                    i++;
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<float> x, float y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<float> x, float y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<float> x, float y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<float> x, float y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<float> x, float y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<float> x, float y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class IntArithmetic : Arithmetic<int>
        {

            protected override void Add(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref int xRef = ref MemoryMarshal.GetReference(x);
                    ref int yRef = ref MemoryMarshal.GetReference(y);
                    ref int dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<int>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref int xRef = ref MemoryMarshal.GetReference(x);
                    ref int yRef = ref MemoryMarshal.GetReference(y);
                    ref int dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<int>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref int xRef = ref MemoryMarshal.GetReference(x);
                    ref int yRef = ref MemoryMarshal.GetReference(y);
                    ref int dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<int>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x * y[i]);
                    i++;
                }
            }

            protected override int Divide(int x, int y)
            {
                return (int)(x / y);
            }

            protected override void Divide(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref int xRef = ref MemoryMarshal.GetReference(x);
                    ref int yRef = ref MemoryMarshal.GetReference(y);
                    ref int dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<int>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x / y[i]);
                    i++;
                }
            }

            protected override int Modulo(int x, int y)
            {
                return (int)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] & y);
                    i++;
                }
            }

            protected override void And(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (int)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(int x, ReadOnlySpan<int> y, Span<int> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (int)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (int)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<int> x, int y, Span<int> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (int)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<int> x, int y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<int> x, int y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<int> x, int y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<int> x, int y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<int> x, int y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<int> x, int y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class LongArithmetic : Arithmetic<long>
        {

            protected override void Add(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref long xRef = ref MemoryMarshal.GetReference(x);
                    ref long yRef = ref MemoryMarshal.GetReference(y);
                    ref long dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<long>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref long xRef = ref MemoryMarshal.GetReference(x);
                    ref long yRef = ref MemoryMarshal.GetReference(y);
                    ref long dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<long>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref long xRef = ref MemoryMarshal.GetReference(x);
                    ref long yRef = ref MemoryMarshal.GetReference(y);
                    ref long dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<long>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x * y[i]);
                    i++;
                }
            }

            protected override long Divide(long x, long y)
            {
                return (long)(x / y);
            }

            protected override void Divide(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref long xRef = ref MemoryMarshal.GetReference(x);
                    ref long yRef = ref MemoryMarshal.GetReference(y);
                    ref long dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<long>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x / y[i]);
                    i++;
                }
            }

            protected override long Modulo(long x, long y)
            {
                return (long)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] & y);
                    i++;
                }
            }

            protected override void And(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<long> x, long y, Span<long> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (long)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(long x, ReadOnlySpan<long> y, Span<long> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (long)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<long> x, int y, Span<long> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (long)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<long> x, int y, Span<long> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (long)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<long> x, long y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<long> x, long y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<long> x, long y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<long> x, long y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<long> x, long y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<long> x, long y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class SByteArithmetic : Arithmetic<sbyte>
        {

            protected override void Add(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref sbyte xRef = ref MemoryMarshal.GetReference(x);
                    ref sbyte yRef = ref MemoryMarshal.GetReference(y);
                    ref sbyte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<sbyte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref sbyte xRef = ref MemoryMarshal.GetReference(x);
                    ref sbyte yRef = ref MemoryMarshal.GetReference(y);
                    ref sbyte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<sbyte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref sbyte xRef = ref MemoryMarshal.GetReference(x);
                    ref sbyte yRef = ref MemoryMarshal.GetReference(y);
                    ref sbyte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<sbyte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x * y[i]);
                    i++;
                }
            }

            protected override sbyte Divide(sbyte x, sbyte y)
            {
                return (sbyte)(x / y);
            }

            protected override void Divide(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref sbyte xRef = ref MemoryMarshal.GetReference(x);
                    ref sbyte yRef = ref MemoryMarshal.GetReference(y);
                    ref sbyte dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<sbyte>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x / y[i]);
                    i++;
                }
            }

            protected override sbyte Modulo(sbyte x, sbyte y)
            {
                return (sbyte)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] & y);
                    i++;
                }
            }

            protected override void And(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<sbyte> x, sbyte y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (sbyte)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(sbyte x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (sbyte)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<sbyte> x, int y, Span<sbyte> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (sbyte)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<sbyte> x, int y, Span<sbyte> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (sbyte)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<sbyte> x, sbyte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<sbyte> x, sbyte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<sbyte> x, sbyte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<sbyte> x, sbyte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<sbyte> x, sbyte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<sbyte> x, sbyte y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class ShortArithmetic : Arithmetic<short>
        {

            protected override void Add(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref short xRef = ref MemoryMarshal.GetReference(x);
                    ref short yRef = ref MemoryMarshal.GetReference(y);
                    ref short dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<short>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref short xRef = ref MemoryMarshal.GetReference(x);
                    ref short yRef = ref MemoryMarshal.GetReference(y);
                    ref short dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<short>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref short xRef = ref MemoryMarshal.GetReference(x);
                    ref short yRef = ref MemoryMarshal.GetReference(y);
                    ref short dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<short>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x * y[i]);
                    i++;
                }
            }

            protected override short Divide(short x, short y)
            {
                return (short)(x / y);
            }

            protected override void Divide(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref short xRef = ref MemoryMarshal.GetReference(x);
                    ref short yRef = ref MemoryMarshal.GetReference(y);
                    ref short dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<short>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x / y[i]);
                    i++;
                }
            }

            protected override short Modulo(short x, short y)
            {
                return (short)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] & y);
                    i++;
                }
            }

            protected override void And(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<short> x, short y, Span<short> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (short)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(short x, ReadOnlySpan<short> y, Span<short> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (short)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<short> x, int y, Span<short> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (short)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<short> x, int y, Span<short> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (short)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<short> x, short y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<short> x, short y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<short> x, short y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<short> x, short y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<short> x, short y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<short> x, short y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class UIntArithmetic : Arithmetic<uint>
        {

            protected override void Add(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref uint xRef = ref MemoryMarshal.GetReference(x);
                    ref uint yRef = ref MemoryMarshal.GetReference(y);
                    ref uint dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<uint>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref uint xRef = ref MemoryMarshal.GetReference(x);
                    ref uint yRef = ref MemoryMarshal.GetReference(y);
                    ref uint dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<uint>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref uint xRef = ref MemoryMarshal.GetReference(x);
                    ref uint yRef = ref MemoryMarshal.GetReference(y);
                    ref uint dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<uint>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x * y[i]);
                    i++;
                }
            }

            protected override uint Divide(uint x, uint y)
            {
                return (uint)(x / y);
            }

            protected override void Divide(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref uint xRef = ref MemoryMarshal.GetReference(x);
                    ref uint yRef = ref MemoryMarshal.GetReference(y);
                    ref uint dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<uint>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x / y[i]);
                    i++;
                }
            }

            protected override uint Modulo(uint x, uint y)
            {
                return (uint)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] & y);
                    i++;
                }
            }

            protected override void And(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<uint> x, uint y, Span<uint> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (uint)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(uint x, ReadOnlySpan<uint> y, Span<uint> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (uint)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<uint> x, int y, Span<uint> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (uint)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<uint> x, int y, Span<uint> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (uint)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<uint> x, uint y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<uint> x, uint y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<uint> x, uint y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<uint> x, uint y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<uint> x, uint y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<uint> x, uint y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class ULongArithmetic : Arithmetic<ulong>
        {

            protected override void Add(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ulong xRef = ref MemoryMarshal.GetReference(x);
                    ref ulong yRef = ref MemoryMarshal.GetReference(y);
                    ref ulong dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ulong>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ulong xRef = ref MemoryMarshal.GetReference(x);
                    ref ulong yRef = ref MemoryMarshal.GetReference(y);
                    ref ulong dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ulong>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ulong xRef = ref MemoryMarshal.GetReference(x);
                    ref ulong yRef = ref MemoryMarshal.GetReference(y);
                    ref ulong dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ulong>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x * y[i]);
                    i++;
                }
            }

            protected override ulong Divide(ulong x, ulong y)
            {
                return (ulong)(x / y);
            }

            protected override void Divide(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ulong xRef = ref MemoryMarshal.GetReference(x);
                    ref ulong yRef = ref MemoryMarshal.GetReference(y);
                    ref ulong dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ulong>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x / y[i]);
                    i++;
                }
            }

            protected override ulong Modulo(ulong x, ulong y)
            {
                return (ulong)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] & y);
                    i++;
                }
            }

            protected override void And(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<ulong> x, ulong y, Span<ulong> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ulong)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(ulong x, ReadOnlySpan<ulong> y, Span<ulong> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ulong)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<ulong> x, int y, Span<ulong> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (ulong)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<ulong> x, int y, Span<ulong> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (ulong)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<ulong> x, ulong y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<ulong> x, ulong y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ulong> x, ulong y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ulong> x, ulong y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<ulong> x, ulong y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<ulong> x, ulong y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class UShortArithmetic : Arithmetic<ushort>
        {

            protected override void Add(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ushort xRef = ref MemoryMarshal.GetReference(x);
                    ref ushort yRef = ref MemoryMarshal.GetReference(y);
                    ref ushort dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ushort>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) + Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] + y[i]);
                    i++;
                }
            }

            protected override void Add(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] + y);
                    i++;
                }
            }

            protected override void Add(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x + y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ushort xRef = ref MemoryMarshal.GetReference(x);
                    ref ushort yRef = ref MemoryMarshal.GetReference(y);
                    ref ushort dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ushort>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) - Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] - y[i]);
                    i++;
                }
            }

            protected override void Subtract(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] - y);
                    i++;
                }
            }

            protected override void Subtract(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x - y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ushort xRef = ref MemoryMarshal.GetReference(x);
                    ref ushort yRef = ref MemoryMarshal.GetReference(y);
                    ref ushort dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ushort>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) * Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] * y[i]);
                    i++;
                }
            }

            protected override void Multiply(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] * y);
                    i++;
                }
            }

            protected override void Multiply(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x * y[i]);
                    i++;
                }
            }

            protected override ushort Divide(ushort x, ushort y)
            {
                return (ushort)(x / y);
            }

            protected override void Divide(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref ushort xRef = ref MemoryMarshal.GetReference(x);
                    ref ushort yRef = ref MemoryMarshal.GetReference(y);
                    ref ushort dRef = ref MemoryMarshal.GetReference(destination);

                    var vectorSize = Vector<ushort>.Count;
                    var oneVectorFromEnd = x.Length - vectorSize;

                    if (oneVectorFromEnd >= 0)
                    {
                        // Loop handling one vector at a time.
                        do
                        {
                            Arithmetic.AsVector(ref dRef, i) = (Arithmetic.AsVector(ref xRef, i) / Arithmetic.AsVector(ref yRef, i));

                            i += vectorSize;
                        }
                        while (i <= oneVectorFromEnd);
                    }
                }

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] / y[i]);
                    i++;
                }
            }

            protected override void Divide(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] / y);
                    i++;
                }
            }

            protected override void Divide(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x / y[i]);
                    i++;
                }
            }

            protected override ushort Modulo(ushort x, ushort y)
            {
                return (ushort)(x % y);
            }

            protected override void Modulo(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] % y[i]);
                    i++;
                }
            }

            protected override void Modulo(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] % y);
                    i++;
                }
            }

            protected override void Modulo(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x % y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] & y[i]);
                    i++;
                }
            }

            protected override void And(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] & y);
                    i++;
                }
            }

            protected override void And(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x & y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] | y[i]);
                    i++;
                }
            }

            protected override void Or(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] | y);
                    i++;
                }
            }

            protected override void Or(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x | y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] ^ y[i]);
                    i++;
                }
            }

            protected override void Xor(ReadOnlySpan<ushort> x, ushort y, Span<ushort> destination)
            {
                int i = 0;

                while (i < x.Length)
                {
                    destination[i] = (ushort)(x[i] ^ y);
                    i++;
                }
            }

            protected override void Xor(ushort x, ReadOnlySpan<ushort> y, Span<ushort> destination)
            {
                int i = 0;

                while (i < y.Length)
                {
                    destination[i] = (ushort)(x ^ y[i]);
                    i++;
                }
            }

            protected override void LeftShift(ReadOnlySpan<ushort> x, int y, Span<ushort> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (ushort)(x[i] << y);
            }

            protected override void RightShift(ReadOnlySpan<ushort> x, int y, Span<ushort> destination)
            {
                for (var i = 0; i < x.Length; i++)
                    destination[i] = (ushort)(x[i] >> y);
            }

            protected override void ElementwiseEquals(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<ushort> x, ushort y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<ushort> x, ushort y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y[i]);
                }
            }

            protected override void ElementwiseGreaterThanOrEqual(ReadOnlySpan<ushort> x, ushort y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] >= y);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y[i]);
                }
            }

            protected override void ElementwiseLessThanOrEqual(ReadOnlySpan<ushort> x, ushort y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] <= y);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y[i]);
                }
            }

            protected override void ElementwiseGreaterThan(ReadOnlySpan<ushort> x, ushort y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] > y);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y[i]);
                }
            }

            protected override void ElementwiseLessThan(ReadOnlySpan<ushort> x, ushort y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] < y);
                }
            }
        }
        internal class DateTimeArithmetic : Arithmetic<DateTime>
        {

            protected override void ElementwiseEquals(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y[i]);
                }
            }

            protected override void ElementwiseEquals(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] == y);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<DateTime> x, ReadOnlySpan<DateTime> y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y[i]);
                }
            }

            protected override void ElementwiseNotEquals(ReadOnlySpan<DateTime> x, DateTime y, Span<bool> destination)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    destination[i] = (x[i] != y);
                }
            }
        }
        #endregion

        internal static IArithmetic<T> GetArithmetic<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(bool))
                return (IArithmetic<T>)new BoolArithmetic();
            else if (typeof(T) == typeof(byte))
                return (IArithmetic<T>)new ByteArithmetic();
            else if (typeof(T) == typeof(char))
                return (IArithmetic<T>)new CharArithmetic();
            else if (typeof(T) == typeof(decimal))
                return (IArithmetic<T>)new DecimalArithmetic();
            else if (typeof(T) == typeof(double))
                return (IArithmetic<T>)new DoubleArithmetic();
            else if (typeof(T) == typeof(float))
                return (IArithmetic<T>)new FloatArithmetic();
            else if (typeof(T) == typeof(int))
                return (IArithmetic<T>)new IntArithmetic();
            else if (typeof(T) == typeof(long))
                return (IArithmetic<T>)new LongArithmetic();
            else if (typeof(T) == typeof(sbyte))
                return (IArithmetic<T>)new SByteArithmetic();
            else if (typeof(T) == typeof(short))
                return (IArithmetic<T>)new ShortArithmetic();
            else if (typeof(T) == typeof(uint))
                return (IArithmetic<T>)new UIntArithmetic();
            else if (typeof(T) == typeof(ulong))
                return (IArithmetic<T>)new ULongArithmetic();
            else if (typeof(T) == typeof(ushort))
                return (IArithmetic<T>)new UShortArithmetic();
            else if (typeof(T) == typeof(DateTime))
                return (IArithmetic<T>)new DateTimeArithmetic();
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ref Vector<T> AsVector<T>(ref T start, int offset)
            where T : struct => ref Unsafe.As<T, Vector<T>>(ref Unsafe.Add(ref start, offset));
    }
}
#endif