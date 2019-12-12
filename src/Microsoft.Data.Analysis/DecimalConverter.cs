using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
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
            if (typeof(T) == typeof(bool))
            {
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(byte))
            {
                return (IDecimalConverter<T>)new ByteDecimalConverter();
            }
            else if (typeof(T) == typeof(char))
            {
                return (IDecimalConverter<T>)new CharDecimalConverter();
            }
            else if (typeof(T) == typeof(decimal))
            {
                return (IDecimalConverter<T>)new DecimalDecimalConverter();
            }
            else if (typeof(T) == typeof(double))
            {
                return (IDecimalConverter<T>)new DoubleDecimalConverter();
            }
            else if (typeof(T) == typeof(float))
            {
                return (IDecimalConverter<T>)new FloatDecimalConverter();
            }
            else if (typeof(T) == typeof(int))
            {
                return (IDecimalConverter<T>)new IntDecimalConverter();
            }
            else if (typeof(T) == typeof(long))
            {
                return (IDecimalConverter<T>)new LongDecimalConverter();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (IDecimalConverter<T>)new SByteDecimalConverter();
            }
            else if (typeof(T) == typeof(short))
            {
                return (IDecimalConverter<T>)new ShortDecimalConverter();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (IDecimalConverter<T>)new UIntDecimalConverter();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (IDecimalConverter<T>)new ULongDecimalConverter();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (IDecimalConverter<T>)new UShortDecimalConverter();
            }
            throw new NotSupportedException();
        }
    }

    internal class ByteDecimalConverter : IDecimalConverter<byte>
    {
        public decimal GetDecimal(byte value)
        {
            return value;
        }
    }

    internal class CharDecimalConverter : IDecimalConverter<char>
    {
        public decimal GetDecimal(char value)
        {
            return value;
        }
    }

    internal class DecimalDecimalConverter : IDecimalConverter<decimal>
    {
        public decimal GetDecimal(decimal value)
        {
            return value;
        }
    }

    internal class DoubleDecimalConverter : IDecimalConverter<double>
    {
        public decimal GetDecimal(double value)
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

    internal class IntDecimalConverter : IDecimalConverter<int>
    {
        public decimal GetDecimal(int value)
        {
            return value;
        }
    }

    internal class LongDecimalConverter : IDecimalConverter<long>
    {
        public decimal GetDecimal(long value)
        {
            return value;
        }
    }

    internal class SByteDecimalConverter : IDecimalConverter<sbyte>
    {
        public decimal GetDecimal(sbyte value)
        {
            return value;
        }
    }

    internal class ShortDecimalConverter : IDecimalConverter<short>
    {
        public decimal GetDecimal(short value)
        {
            return value;
        }
    }

    internal class UIntDecimalConverter : IDecimalConverter<uint>
    {
        public decimal GetDecimal(uint value)
        {
            return value;
        }
    }

    internal class ULongDecimalConverter : IDecimalConverter<ulong>
    {
        public decimal GetDecimal(ulong value)
        {
            return value;
        }
    }

    internal class UShortDecimalConverter : IDecimalConverter<ushort>
    {
        public decimal GetDecimal(ushort value)
        {
            return value;
        }
    }
}
