
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
    internal interface IInt16Converter<T>
    {
        short GetInt16(T value);
    }
    internal static class Int16Converter<T>
    {
        public static IInt16Converter<T> Instance { get; } = Int16Converter.GetInt16Converter<T>();
    }
    internal static class Int16Converter
    {
        public static IInt16Converter<T> GetInt16Converter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IInt16Converter<T>)new ByteInt16Converter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IInt16Converter<T>)new SByteInt16Converter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IInt16Converter<T>)new Int16Int16Converter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IInt16Converter<T>)new UInt16Int16Converter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteInt16Converter : IInt16Converter<byte>
    {
        public short GetInt16(byte value)
        {
            return (short)value;
        }
    }
    internal class SByteInt16Converter : IInt16Converter<sbyte>
    {
        public short GetInt16(sbyte value)
        {
            return (short)value;
        }
    }
    internal class Int16Int16Converter : IInt16Converter<short>
    {
        public short GetInt16(short value)
        {
            return (short)value;
        }
    }
    internal class UInt16Int16Converter : IInt16Converter<ushort>
    {
        public short GetInt16(ushort value)
        {
            return (short)value;
        }
    }
    internal interface IUInt16Converter<T>
    {
        ushort GetUInt16(T value);
    }
    internal static class UInt16Converter<T>
    {
        public static IUInt16Converter<T> Instance { get; } = UInt16Converter.GetUInt16Converter<T>();
    }
    internal static class UInt16Converter
    {
        public static IUInt16Converter<T> GetUInt16Converter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IUInt16Converter<T>)new ByteUInt16Converter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IUInt16Converter<T>)new SByteUInt16Converter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IUInt16Converter<T>)new Int16UInt16Converter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IUInt16Converter<T>)new UInt16UInt16Converter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteUInt16Converter : IUInt16Converter<byte>
    {
        public ushort GetUInt16(byte value)
        {
            return (ushort)value;
        }
    }
    internal class SByteUInt16Converter : IUInt16Converter<sbyte>
    {
        public ushort GetUInt16(sbyte value)
        {
            return (ushort)value;
        }
    }
    internal class Int16UInt16Converter : IUInt16Converter<short>
    {
        public ushort GetUInt16(short value)
        {
            return (ushort)value;
        }
    }
    internal class UInt16UInt16Converter : IUInt16Converter<ushort>
    {
        public ushort GetUInt16(ushort value)
        {
            return (ushort)value;
        }
    }
    internal interface IInt32Converter<T>
    {
        int GetInt32(T value);
    }
    internal static class Int32Converter<T>
    {
        public static IInt32Converter<T> Instance { get; } = Int32Converter.GetInt32Converter<T>();
    }
    internal static class Int32Converter
    {
        public static IInt32Converter<T> GetInt32Converter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IInt32Converter<T>)new ByteInt32Converter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IInt32Converter<T>)new SByteInt32Converter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IInt32Converter<T>)new Int16Int32Converter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IInt32Converter<T>)new UInt16Int32Converter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IInt32Converter<T>)new Int32Int32Converter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IInt32Converter<T>)new UInt32Int32Converter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteInt32Converter : IInt32Converter<byte>
    {
        public int GetInt32(byte value)
        {
            return (int)value;
        }
    }
    internal class SByteInt32Converter : IInt32Converter<sbyte>
    {
        public int GetInt32(sbyte value)
        {
            return (int)value;
        }
    }
    internal class Int16Int32Converter : IInt32Converter<short>
    {
        public int GetInt32(short value)
        {
            return (int)value;
        }
    }
    internal class UInt16Int32Converter : IInt32Converter<ushort>
    {
        public int GetInt32(ushort value)
        {
            return (int)value;
        }
    }
    internal class Int32Int32Converter : IInt32Converter<int>
    {
        public int GetInt32(int value)
        {
            return (int)value;
        }
    }
    internal class UInt32Int32Converter : IInt32Converter<uint>
    {
        public int GetInt32(uint value)
        {
            return (int)value;
        }
    }
    internal interface IUInt32Converter<T>
    {
        uint GetUInt32(T value);
    }
    internal static class UInt32Converter<T>
    {
        public static IUInt32Converter<T> Instance { get; } = UInt32Converter.GetUInt32Converter<T>();
    }
    internal static class UInt32Converter
    {
        public static IUInt32Converter<T> GetUInt32Converter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IUInt32Converter<T>)new ByteUInt32Converter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IUInt32Converter<T>)new SByteUInt32Converter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IUInt32Converter<T>)new Int16UInt32Converter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IUInt32Converter<T>)new UInt16UInt32Converter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IUInt32Converter<T>)new Int32UInt32Converter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IUInt32Converter<T>)new UInt32UInt32Converter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteUInt32Converter : IUInt32Converter<byte>
    {
        public uint GetUInt32(byte value)
        {
            return (uint)value;
        }
    }
    internal class SByteUInt32Converter : IUInt32Converter<sbyte>
    {
        public uint GetUInt32(sbyte value)
        {
            return (uint)value;
        }
    }
    internal class Int16UInt32Converter : IUInt32Converter<short>
    {
        public uint GetUInt32(short value)
        {
            return (uint)value;
        }
    }
    internal class UInt16UInt32Converter : IUInt32Converter<ushort>
    {
        public uint GetUInt32(ushort value)
        {
            return (uint)value;
        }
    }
    internal class Int32UInt32Converter : IUInt32Converter<int>
    {
        public uint GetUInt32(int value)
        {
            return (uint)value;
        }
    }
    internal class UInt32UInt32Converter : IUInt32Converter<uint>
    {
        public uint GetUInt32(uint value)
        {
            return (uint)value;
        }
    }
    internal interface IInt64Converter<T>
    {
        long GetInt64(T value);
    }
    internal static class Int64Converter<T>
    {
        public static IInt64Converter<T> Instance { get; } = Int64Converter.GetInt64Converter<T>();
    }
    internal static class Int64Converter
    {
        public static IInt64Converter<T> GetInt64Converter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IInt64Converter<T>)new ByteInt64Converter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IInt64Converter<T>)new SByteInt64Converter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IInt64Converter<T>)new Int16Int64Converter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IInt64Converter<T>)new UInt16Int64Converter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IInt64Converter<T>)new Int32Int64Converter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IInt64Converter<T>)new UInt32Int64Converter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IInt64Converter<T>)new Int64Int64Converter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IInt64Converter<T>)new UInt64Int64Converter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteInt64Converter : IInt64Converter<byte>
    {
        public long GetInt64(byte value)
        {
            return (long)value;
        }
    }
    internal class SByteInt64Converter : IInt64Converter<sbyte>
    {
        public long GetInt64(sbyte value)
        {
            return (long)value;
        }
    }
    internal class Int16Int64Converter : IInt64Converter<short>
    {
        public long GetInt64(short value)
        {
            return (long)value;
        }
    }
    internal class UInt16Int64Converter : IInt64Converter<ushort>
    {
        public long GetInt64(ushort value)
        {
            return (long)value;
        }
    }
    internal class Int32Int64Converter : IInt64Converter<int>
    {
        public long GetInt64(int value)
        {
            return (long)value;
        }
    }
    internal class UInt32Int64Converter : IInt64Converter<uint>
    {
        public long GetInt64(uint value)
        {
            return (long)value;
        }
    }
    internal class Int64Int64Converter : IInt64Converter<long>
    {
        public long GetInt64(long value)
        {
            return (long)value;
        }
    }
    internal class UInt64Int64Converter : IInt64Converter<ulong>
    {
        public long GetInt64(ulong value)
        {
            return (long)value;
        }
    }
    internal interface IUInt64Converter<T>
    {
        ulong GetUInt64(T value);
    }
    internal static class UInt64Converter<T>
    {
        public static IUInt64Converter<T> Instance { get; } = UInt64Converter.GetUInt64Converter<T>();
    }
    internal static class UInt64Converter
    {
        public static IUInt64Converter<T> GetUInt64Converter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (IUInt64Converter<T>)new ByteUInt64Converter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (IUInt64Converter<T>)new SByteUInt64Converter();
            }
            if (typeof(T) == typeof(short))
            {
                return (IUInt64Converter<T>)new Int16UInt64Converter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IUInt64Converter<T>)new UInt16UInt64Converter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IUInt64Converter<T>)new Int32UInt64Converter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IUInt64Converter<T>)new UInt32UInt64Converter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IUInt64Converter<T>)new Int64UInt64Converter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IUInt64Converter<T>)new UInt64UInt64Converter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteUInt64Converter : IUInt64Converter<byte>
    {
        public ulong GetUInt64(byte value)
        {
            return (ulong)value;
        }
    }
    internal class SByteUInt64Converter : IUInt64Converter<sbyte>
    {
        public ulong GetUInt64(sbyte value)
        {
            return (ulong)value;
        }
    }
    internal class Int16UInt64Converter : IUInt64Converter<short>
    {
        public ulong GetUInt64(short value)
        {
            return (ulong)value;
        }
    }
    internal class UInt16UInt64Converter : IUInt64Converter<ushort>
    {
        public ulong GetUInt64(ushort value)
        {
            return (ulong)value;
        }
    }
    internal class Int32UInt64Converter : IUInt64Converter<int>
    {
        public ulong GetUInt64(int value)
        {
            return (ulong)value;
        }
    }
    internal class UInt32UInt64Converter : IUInt64Converter<uint>
    {
        public ulong GetUInt64(uint value)
        {
            return (ulong)value;
        }
    }
    internal class Int64UInt64Converter : IUInt64Converter<long>
    {
        public ulong GetUInt64(long value)
        {
            return (ulong)value;
        }
    }
    internal class UInt64UInt64Converter : IUInt64Converter<ulong>
    {
        public ulong GetUInt64(ulong value)
        {
            return (ulong)value;
        }
    }
    internal interface ISingleConverter<T>
    {
        float GetSingle(T value);
    }
    internal static class SingleConverter<T>
    {
        public static ISingleConverter<T> Instance { get; } = SingleConverter.GetSingleConverter<T>();
    }
    internal static class SingleConverter
    {
        public static ISingleConverter<T> GetSingleConverter<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return (ISingleConverter<T>)new ByteSingleConverter();
            }
            if (typeof(T) == typeof(sbyte))
            {
                return (ISingleConverter<T>)new SByteSingleConverter();
            }
            if (typeof(T) == typeof(short))
            {
                return (ISingleConverter<T>)new Int16SingleConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (ISingleConverter<T>)new UInt16SingleConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (ISingleConverter<T>)new Int32SingleConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (ISingleConverter<T>)new UInt32SingleConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (ISingleConverter<T>)new Int64SingleConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (ISingleConverter<T>)new UInt64SingleConverter();
            }
            if (typeof(T) == typeof(float))
            {
                return (ISingleConverter<T>)new SingleSingleConverter();
            }
            throw new NotSupportedException();
        }
    }
    internal class ByteSingleConverter : ISingleConverter<byte>
    {
        public float GetSingle(byte value)
        {
            return (float)value;
        }
    }
    internal class SByteSingleConverter : ISingleConverter<sbyte>
    {
        public float GetSingle(sbyte value)
        {
            return (float)value;
        }
    }
    internal class Int16SingleConverter : ISingleConverter<short>
    {
        public float GetSingle(short value)
        {
            return (float)value;
        }
    }
    internal class UInt16SingleConverter : ISingleConverter<ushort>
    {
        public float GetSingle(ushort value)
        {
            return (float)value;
        }
    }
    internal class Int32SingleConverter : ISingleConverter<int>
    {
        public float GetSingle(int value)
        {
            return (float)value;
        }
    }
    internal class UInt32SingleConverter : ISingleConverter<uint>
    {
        public float GetSingle(uint value)
        {
            return (float)value;
        }
    }
    internal class Int64SingleConverter : ISingleConverter<long>
    {
        public float GetSingle(long value)
        {
            return (float)value;
        }
    }
    internal class UInt64SingleConverter : ISingleConverter<ulong>
    {
        public float GetSingle(ulong value)
        {
            return (float)value;
        }
    }
    internal class SingleSingleConverter : ISingleConverter<float>
    {
        public float GetSingle(float value)
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
                return (IDoubleConverter<T>)new Int16DoubleConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IDoubleConverter<T>)new UInt16DoubleConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IDoubleConverter<T>)new Int32DoubleConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IDoubleConverter<T>)new UInt32DoubleConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IDoubleConverter<T>)new Int64DoubleConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IDoubleConverter<T>)new UInt64DoubleConverter();
            }
            if (typeof(T) == typeof(float))
            {
                return (IDoubleConverter<T>)new SingleDoubleConverter();
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
    internal class Int16DoubleConverter : IDoubleConverter<short>
    {
        public double GetDouble(short value)
        {
            return (double)value;
        }
    }
    internal class UInt16DoubleConverter : IDoubleConverter<ushort>
    {
        public double GetDouble(ushort value)
        {
            return (double)value;
        }
    }
    internal class Int32DoubleConverter : IDoubleConverter<int>
    {
        public double GetDouble(int value)
        {
            return (double)value;
        }
    }
    internal class UInt32DoubleConverter : IDoubleConverter<uint>
    {
        public double GetDouble(uint value)
        {
            return (double)value;
        }
    }
    internal class Int64DoubleConverter : IDoubleConverter<long>
    {
        public double GetDouble(long value)
        {
            return (double)value;
        }
    }
    internal class UInt64DoubleConverter : IDoubleConverter<ulong>
    {
        public double GetDouble(ulong value)
        {
            return (double)value;
        }
    }
    internal class SingleDoubleConverter : IDoubleConverter<float>
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
                return (IDecimalConverter<T>)new Int16DecimalConverter();
            }
            if (typeof(T) == typeof(ushort))
            {
                return (IDecimalConverter<T>)new UInt16DecimalConverter();
            }
            if (typeof(T) == typeof(int))
            {
                return (IDecimalConverter<T>)new Int32DecimalConverter();
            }
            if (typeof(T) == typeof(uint))
            {
                return (IDecimalConverter<T>)new UInt32DecimalConverter();
            }
            if (typeof(T) == typeof(long))
            {
                return (IDecimalConverter<T>)new Int64DecimalConverter();
            }
            if (typeof(T) == typeof(ulong))
            {
                return (IDecimalConverter<T>)new UInt64DecimalConverter();
            }
            if (typeof(T) == typeof(float))
            {
                return (IDecimalConverter<T>)new SingleDecimalConverter();
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
    internal class Int16DecimalConverter : IDecimalConverter<short>
    {
        public decimal GetDecimal(short value)
        {
            return (decimal)value;
        }
    }
    internal class UInt16DecimalConverter : IDecimalConverter<ushort>
    {
        public decimal GetDecimal(ushort value)
        {
            return (decimal)value;
        }
    }
    internal class Int32DecimalConverter : IDecimalConverter<int>
    {
        public decimal GetDecimal(int value)
        {
            return (decimal)value;
        }
    }
    internal class UInt32DecimalConverter : IDecimalConverter<uint>
    {
        public decimal GetDecimal(uint value)
        {
            return (decimal)value;
        }
    }
    internal class Int64DecimalConverter : IDecimalConverter<long>
    {
        public decimal GetDecimal(long value)
        {
            return (decimal)value;
        }
    }
    internal class UInt64DecimalConverter : IDecimalConverter<ulong>
    {
        public decimal GetDecimal(ulong value)
        {
            return (decimal)value;
        }
    }
    internal class SingleDecimalConverter : IDecimalConverter<float>
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
