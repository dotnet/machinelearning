using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
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
            if (typeof(T) == typeof(bool))
            {
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(byte))
            {
                return (IDoubleConverter<T>)new ByteDoubleConverter();
            }
            else if (typeof(T) == typeof(char))
            {
                return (IDoubleConverter<T>)new CharDoubleConverter();
            }
            else if (typeof(T) == typeof(decimal))
            {
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                return (IDoubleConverter<T>)new DoubleDoubleConverter();
            }
            else if (typeof(T) == typeof(float))
            {
                return (IDoubleConverter<T>)new FloatDoubleConverter();
            }
            else if (typeof(T) == typeof(int))
            {
                return (IDoubleConverter<T>)new IntDoubleConverter();
            }
            else if (typeof(T) == typeof(long))
            {
                return (IDoubleConverter<T>)new LongDoubleConverter();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (IDoubleConverter<T>)new SByteDoubleConverter();
            }
            else if (typeof(T) == typeof(short))
            {
                return (IDoubleConverter<T>)new ShortDoubleConverter();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (IDoubleConverter<T>)new UIntDoubleConverter();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (IDoubleConverter<T>)new ULongDoubleConverter();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (IDoubleConverter<T>)new UShortDoubleConverter();
            }
            throw new NotSupportedException();
        }
    }

    internal class ByteDoubleConverter : IDoubleConverter<byte>
    {
        public double GetDouble(byte value)
        {
            return value;
        }
    }

    internal class CharDoubleConverter : IDoubleConverter<char>
    {
        public double GetDouble(char value)
        {
            return value;
        }
    }

    internal class DoubleDoubleConverter : IDoubleConverter<double>
    {
        public double GetDouble(double value)
        {
            return value;
        }
    }

    internal class FloatDoubleConverter : IDoubleConverter<float>
    {
        public double GetDouble(float value)
        {
            return value;
        }
    }

    internal class IntDoubleConverter : IDoubleConverter<int>
    {
        public double GetDouble(int value)
        {
            return value;
        }
    }

    internal class LongDoubleConverter : IDoubleConverter<long>
    {
        public double GetDouble(long value)
        {
            return value;
        }
    }

    internal class SByteDoubleConverter : IDoubleConverter<sbyte>
    {
        public double GetDouble(sbyte value)
        {
            return value;
        }
    }

    internal class ShortDoubleConverter : IDoubleConverter<short>
    {
        public double GetDouble(short value)
        {
            return value;
        }
    }

    internal class UIntDoubleConverter : IDoubleConverter<uint>
    {
        public double GetDouble(uint value)
        {
            return value;
        }
    }

    internal class ULongDoubleConverter : IDoubleConverter<ulong>
    {
        public double GetDouble(ulong value)
        {
            return value;
        }
    }

    internal class UShortDoubleConverter : IDoubleConverter<ushort>
    {
        public double GetDouble(ushort value)
        {
            return value;
        }
    }
}
