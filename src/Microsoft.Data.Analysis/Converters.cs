
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from Converters.tt. Do not modify directly

using System;
using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    internal interface IByteConverter<T>
    {
        byte GetByte(T value);
    }
    internal static class ByteConverter<T>
    {
        public static IByteConverter<T> Instance { get; } = ByteConverter.GetByteConverter<T>();
    }
    internal static class ByteConverter
    {
        public static IByteConverter<T> GetByteConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IByteConverter<T>)new ByteByteConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IByteConverter<T>)new SByteByteConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteByteConverter : IByteConverter<byte>
    {
        public byte GetByte(byte value)
        {
            return (byte)value;
        }
    }
    internal class SByteByteConverter : IByteConverter<sbyte>
    {
        public byte GetByte(sbyte value)
        {
            return (byte)value;
        }
    }
    internal interface ISByteConverter<T>
    {
        sbyte GetSByte(T value);
    }
    internal static class SByteConverter<T>
    {
        public static ISByteConverter<T> Instance { get; } = SByteConverter.GetSByteConverter<T>();
    }
    internal static class SByteConverter
    {
        public static ISByteConverter<T> GetSByteConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (ISByteConverter<T>)new ByteSByteConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (ISByteConverter<T>)new SByteSByteConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteSByteConverter : ISByteConverter<byte>
    {
        public sbyte GetSByte(byte value)
        {
            return (sbyte)value;
        }
    }
    internal class SByteSByteConverter : ISByteConverter<sbyte>
    {
        public sbyte GetSByte(sbyte value)
        {
            return (sbyte)value;
        }
    }
    internal interface IShortConverter<T>
    {
        short GetShort(T value);
    }
    internal static class ShortConverter<T>
    {
        public static IShortConverter<T> Instance { get; } = ShortConverter.GetShortConverter<T>();
    }
    internal static class ShortConverter
    {
        public static IShortConverter<T> GetShortConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IShortConverter<T>)new ByteShortConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IShortConverter<T>)new SByteShortConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IShortConverter<T>)new ShortShortConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IShortConverter<T>)new UShortShortConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteShortConverter : IShortConverter<byte>
    {
        public short GetShort(byte value)
        {
            return (short)value;
        }
    }
    internal class SByteShortConverter : IShortConverter<sbyte>
    {
        public short GetShort(sbyte value)
        {
            return (short)value;
        }
    }
    internal class ShortShortConverter : IShortConverter<short>
    {
        public short GetShort(short value)
        {
            return (short)value;
        }
    }
    internal class UShortShortConverter : IShortConverter<ushort>
    {
        public short GetShort(ushort value)
        {
            return (short)value;
        }
    }
    internal interface IUShortConverter<T>
    {
        ushort GetUShort(T value);
    }
    internal static class UShortConverter<T>
    {
        public static IUShortConverter<T> Instance { get; } = UShortConverter.GetUShortConverter<T>();
    }
    internal static class UShortConverter
    {
        public static IUShortConverter<T> GetUShortConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IUShortConverter<T>)new ByteUShortConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IUShortConverter<T>)new SByteUShortConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IUShortConverter<T>)new ShortUShortConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IUShortConverter<T>)new UShortUShortConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteUShortConverter : IUShortConverter<byte>
    {
        public ushort GetUShort(byte value)
        {
            return (ushort)value;
        }
    }
    internal class SByteUShortConverter : IUShortConverter<sbyte>
    {
        public ushort GetUShort(sbyte value)
        {
            return (ushort)value;
        }
    }
    internal class ShortUShortConverter : IUShortConverter<short>
    {
        public ushort GetUShort(short value)
        {
            return (ushort)value;
        }
    }
    internal class UShortUShortConverter : IUShortConverter<ushort>
    {
        public ushort GetUShort(ushort value)
        {
            return (ushort)value;
        }
    }
    internal interface IIntConverter<T>
    {
        int GetInt(T value);
    }
    internal static class IntConverter<T>
    {
        public static IIntConverter<T> Instance { get; } = IntConverter.GetIntConverter<T>();
    }
    internal static class IntConverter
    {
        public static IIntConverter<T> GetIntConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IIntConverter<T>)new ByteIntConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IIntConverter<T>)new SByteIntConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IIntConverter<T>)new ShortIntConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IIntConverter<T>)new UShortIntConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IIntConverter<T>)new IntIntConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IIntConverter<T>)new UIntIntConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteIntConverter : IIntConverter<byte>
    {
        public int GetInt(byte value)
        {
            return (int)value;
        }
    }
    internal class SByteIntConverter : IIntConverter<sbyte>
    {
        public int GetInt(sbyte value)
        {
            return (int)value;
        }
    }
    internal class ShortIntConverter : IIntConverter<short>
    {
        public int GetInt(short value)
        {
            return (int)value;
        }
    }
    internal class UShortIntConverter : IIntConverter<ushort>
    {
        public int GetInt(ushort value)
        {
            return (int)value;
        }
    }
    internal class IntIntConverter : IIntConverter<int>
    {
        public int GetInt(int value)
        {
            return (int)value;
        }
    }
    internal class UIntIntConverter : IIntConverter<uint>
    {
        public int GetInt(uint value)
        {
            return (int)value;
        }
    }
    internal interface IUIntConverter<T>
    {
        uint GetUInt(T value);
    }
    internal static class UIntConverter<T>
    {
        public static IUIntConverter<T> Instance { get; } = UIntConverter.GetUIntConverter<T>();
    }
    internal static class UIntConverter
    {
        public static IUIntConverter<T> GetUIntConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IUIntConverter<T>)new ByteUIntConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IUIntConverter<T>)new SByteUIntConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IUIntConverter<T>)new ShortUIntConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IUIntConverter<T>)new UShortUIntConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IUIntConverter<T>)new IntUIntConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IUIntConverter<T>)new UIntUIntConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteUIntConverter : IUIntConverter<byte>
    {
        public uint GetUInt(byte value)
        {
            return (uint)value;
        }
    }
    internal class SByteUIntConverter : IUIntConverter<sbyte>
    {
        public uint GetUInt(sbyte value)
        {
            return (uint)value;
        }
    }
    internal class ShortUIntConverter : IUIntConverter<short>
    {
        public uint GetUInt(short value)
        {
            return (uint)value;
        }
    }
    internal class UShortUIntConverter : IUIntConverter<ushort>
    {
        public uint GetUInt(ushort value)
        {
            return (uint)value;
        }
    }
    internal class IntUIntConverter : IUIntConverter<int>
    {
        public uint GetUInt(int value)
        {
            return (uint)value;
        }
    }
    internal class UIntUIntConverter : IUIntConverter<uint>
    {
        public uint GetUInt(uint value)
        {
            return (uint)value;
        }
    }
    internal interface ILongConverter<T>
    {
        long GetLong(T value);
    }
    internal static class LongConverter<T>
    {
        public static ILongConverter<T> Instance { get; } = LongConverter.GetLongConverter<T>();
    }
    internal static class LongConverter
    {
        public static ILongConverter<T> GetLongConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (ILongConverter<T>)new ByteLongConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (ILongConverter<T>)new SByteLongConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (ILongConverter<T>)new ShortLongConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (ILongConverter<T>)new UShortLongConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (ILongConverter<T>)new IntLongConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (ILongConverter<T>)new UIntLongConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (ILongConverter<T>)new LongLongConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (ILongConverter<T>)new ULongLongConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteLongConverter : ILongConverter<byte>
    {
        public long GetLong(byte value)
        {
            return (long)value;
        }
    }
    internal class SByteLongConverter : ILongConverter<sbyte>
    {
        public long GetLong(sbyte value)
        {
            return (long)value;
        }
    }
    internal class ShortLongConverter : ILongConverter<short>
    {
        public long GetLong(short value)
        {
            return (long)value;
        }
    }
    internal class UShortLongConverter : ILongConverter<ushort>
    {
        public long GetLong(ushort value)
        {
            return (long)value;
        }
    }
    internal class IntLongConverter : ILongConverter<int>
    {
        public long GetLong(int value)
        {
            return (long)value;
        }
    }
    internal class UIntLongConverter : ILongConverter<uint>
    {
        public long GetLong(uint value)
        {
            return (long)value;
        }
    }
    internal class LongLongConverter : ILongConverter<long>
    {
        public long GetLong(long value)
        {
            return (long)value;
        }
    }
    internal class ULongLongConverter : ILongConverter<ulong>
    {
        public long GetLong(ulong value)
        {
            return (long)value;
        }
    }
    internal interface IULongConverter<T>
    {
        ulong GetULong(T value);
    }
    internal static class ULongConverter<T>
    {
        public static IULongConverter<T> Instance { get; } = ULongConverter.GetULongConverter<T>();
    }
    internal static class ULongConverter
    {
        public static IULongConverter<T> GetULongConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IULongConverter<T>)new ByteULongConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IULongConverter<T>)new SByteULongConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IULongConverter<T>)new ShortULongConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IULongConverter<T>)new UShortULongConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IULongConverter<T>)new IntULongConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IULongConverter<T>)new UIntULongConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IULongConverter<T>)new LongULongConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IULongConverter<T>)new ULongULongConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteULongConverter : IULongConverter<byte>
    {
        public ulong GetULong(byte value)
        {
            return (ulong)value;
        }
    }
    internal class SByteULongConverter : IULongConverter<sbyte>
    {
        public ulong GetULong(sbyte value)
        {
            return (ulong)value;
        }
    }
    internal class ShortULongConverter : IULongConverter<short>
    {
        public ulong GetULong(short value)
        {
            return (ulong)value;
        }
    }
    internal class UShortULongConverter : IULongConverter<ushort>
    {
        public ulong GetULong(ushort value)
        {
            return (ulong)value;
        }
    }
    internal class IntULongConverter : IULongConverter<int>
    {
        public ulong GetULong(int value)
        {
            return (ulong)value;
        }
    }
    internal class UIntULongConverter : IULongConverter<uint>
    {
        public ulong GetULong(uint value)
        {
            return (ulong)value;
        }
    }
    internal class LongULongConverter : IULongConverter<long>
    {
        public ulong GetULong(long value)
        {
            return (ulong)value;
        }
    }
    internal class ULongULongConverter : IULongConverter<ulong>
    {
        public ulong GetULong(ulong value)
        {
            return (ulong)value;
        }
    }
    internal interface IFloatConverter<T>
    {
        float GetFloat(T value);
    }
    internal static class FloatConverter<T>
    {
        public static IFloatConverter<T> Instance { get; } = FloatConverter.GetFloatConverter<T>();
    }
    internal static class FloatConverter
    {
        public static IFloatConverter<T> GetFloatConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IFloatConverter<T>)new ByteFloatConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IFloatConverter<T>)new SByteFloatConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IFloatConverter<T>)new ShortFloatConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IFloatConverter<T>)new UShortFloatConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IFloatConverter<T>)new IntFloatConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IFloatConverter<T>)new UIntFloatConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IFloatConverter<T>)new LongFloatConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IFloatConverter<T>)new ULongFloatConverter();
            }
            if (typeof(T) == typeof(float))
            {
                return (IFloatConverter<T>)new FloatFloatConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteFloatConverter : IFloatConverter<byte>
    {
        public float GetFloat(byte value)
        {
            return (float)value;
        }
    }
    internal class SByteFloatConverter : IFloatConverter<sbyte>
    {
        public float GetFloat(sbyte value)
        {
            return (float)value;
        }
    }
    internal class ShortFloatConverter : IFloatConverter<short>
    {
        public float GetFloat(short value)
        {
            return (float)value;
        }
    }
    internal class UShortFloatConverter : IFloatConverter<ushort>
    {
        public float GetFloat(ushort value)
        {
            return (float)value;
        }
    }
    internal class IntFloatConverter : IFloatConverter<int>
    {
        public float GetFloat(int value)
        {
            return (float)value;
        }
    }
    internal class UIntFloatConverter : IFloatConverter<uint>
    {
        public float GetFloat(uint value)
        {
            return (float)value;
        }
    }
    internal class LongFloatConverter : IFloatConverter<long>
    {
        public float GetFloat(long value)
        {
            return (float)value;
        }
    }
    internal class ULongFloatConverter : IFloatConverter<ulong>
    {
        public float GetFloat(ulong value)
        {
            return (float)value;
        }
    }
    internal class FloatFloatConverter : IFloatConverter<float>
    {
        public float GetFloat(float value)
        {
            return (float)value;
        }
    }
    internal interface IDoubleConverter<T>
    {
        double GetDouble(T value);
    }
    internal static class DoubleConverter<T>
    {
        public static IDoubleConverter<T> Instance { get; } = DoubleConverter.GetDoubleConverter<T>();
    }
    internal static class DoubleConverter
    {
        public static IDoubleConverter<T> GetDoubleConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IDoubleConverter<T>)new ByteDoubleConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IDoubleConverter<T>)new SByteDoubleConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IDoubleConverter<T>)new ShortDoubleConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IDoubleConverter<T>)new UShortDoubleConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IDoubleConverter<T>)new IntDoubleConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IDoubleConverter<T>)new UIntDoubleConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IDoubleConverter<T>)new LongDoubleConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IDoubleConverter<T>)new ULongDoubleConverter();
            }
            if (typeof(T) == typeof(float))
            {
                return (IDoubleConverter<T>)new FloatDoubleConverter();
            }
            if (typeof(T) == typeof(double))
            {
                return (IDoubleConverter<T>)new DoubleDoubleConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteDoubleConverter : IDoubleConverter<byte>
    {
        public double GetDouble(byte value)
        {
            return (double)value;
        }
    }
    internal class SByteDoubleConverter : IDoubleConverter<sbyte>
    {
        public double GetDouble(sbyte value)
        {
            return (double)value;
        }
    }
    internal class ShortDoubleConverter : IDoubleConverter<short>
    {
        public double GetDouble(short value)
        {
            return (double)value;
        }
    }
    internal class UShortDoubleConverter : IDoubleConverter<ushort>
    {
        public double GetDouble(ushort value)
        {
            return (double)value;
        }
    }
    internal class IntDoubleConverter : IDoubleConverter<int>
    {
        public double GetDouble(int value)
        {
            return (double)value;
        }
    }
    internal class UIntDoubleConverter : IDoubleConverter<uint>
    {
        public double GetDouble(uint value)
        {
            return (double)value;
        }
    }
    internal class LongDoubleConverter : IDoubleConverter<long>
    {
        public double GetDouble(long value)
        {
            return (double)value;
        }
    }
    internal class ULongDoubleConverter : IDoubleConverter<ulong>
    {
        public double GetDouble(ulong value)
        {
            return (double)value;
        }
    }
    internal class FloatDoubleConverter : IDoubleConverter<float>
    {
        public double GetDouble(float value)
        {
            return (double)value;
        }
    }
    internal class DoubleDoubleConverter : IDoubleConverter<double>
    {
        public double GetDouble(double value)
        {
            return (double)value;
        }
    }
    internal interface IDecimalConverter<T>
    {
        decimal GetDecimal(T value);
    }
    internal static class DecimalConverter<T>
    {
        public static IDecimalConverter<T> Instance { get; } = DecimalConverter.GetDecimalConverter<T>();
    }
    internal static class DecimalConverter
    {
        public static IDecimalConverter<T> GetDecimalConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IDecimalConverter<T>)new ByteDecimalConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IDecimalConverter<T>)new SByteDecimalConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IDecimalConverter<T>)new ShortDecimalConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IDecimalConverter<T>)new UShortDecimalConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IDecimalConverter<T>)new IntDecimalConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IDecimalConverter<T>)new UIntDecimalConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IDecimalConverter<T>)new LongDecimalConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IDecimalConverter<T>)new ULongDecimalConverter();
            }
            if (typeof(T) == typeof(float))
            {
                return (IDecimalConverter<T>)new FloatDecimalConverter();
            }
            if (typeof(T) == typeof(double))
            {
                return (IDecimalConverter<T>)new DoubleDecimalConverter();
            }
            if (typeof(T) == typeof(decimal))
            {
                return (IDecimalConverter<T>)new DecimalDecimalConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteDecimalConverter : IDecimalConverter<byte>
    {
        public decimal GetDecimal(byte value)
        {
            return (decimal)value;
        }
    }
    internal class SByteDecimalConverter : IDecimalConverter<sbyte>
    {
        public decimal GetDecimal(sbyte value)
        {
            return (decimal)value;
        }
    }
    internal class ShortDecimalConverter : IDecimalConverter<short>
    {
        public decimal GetDecimal(short value)
        {
            return (decimal)value;
        }
    }
    internal class UShortDecimalConverter : IDecimalConverter<ushort>
    {
        public decimal GetDecimal(ushort value)
        {
            return (decimal)value;
        }
    }
    internal class IntDecimalConverter : IDecimalConverter<int>
    {
        public decimal GetDecimal(int value)
        {
            return (decimal)value;
        }
    }
    internal class UIntDecimalConverter : IDecimalConverter<uint>
    {
        public decimal GetDecimal(uint value)
        {
            return (decimal)value;
        }
    }
    internal class LongDecimalConverter : IDecimalConverter<long>
    {
        public decimal GetDecimal(long value)
        {
            return (decimal)value;
        }
    }
    internal class ULongDecimalConverter : IDecimalConverter<ulong>
    {
        public decimal GetDecimal(ulong value)
        {
            return (decimal)value;
        }
    }
    internal class FloatDecimalConverter : IDecimalConverter<float>
    {
        public decimal GetDecimal(float value)
        {
            return (decimal)value;
        }
    }
    internal class DoubleDecimalConverter : IDecimalConverter<double>
    {
        public decimal GetDecimal(double value)
        {
            return (decimal)value;
        }
    }
    internal class DecimalDecimalConverter : IDecimalConverter<decimal>
    {
        public decimal GetDecimal(decimal value)
        {
            return (decimal)value;
        }
    }
}
