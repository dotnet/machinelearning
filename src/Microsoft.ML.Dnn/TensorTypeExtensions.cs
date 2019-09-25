using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;
using Tensorflow;

namespace Microsoft.ML.Transforms
{
    [BestFriend]
    internal static class TensorTypeExtensions
    {
        public static void ToScalar<T>(this Tensor tensor, ref T dst) where T : unmanaged
        {
            if (typeof(T).as_dtype() == tensor.dtype)
            {
                unsafe
                {
                    dst = *(T*)tensor.buffer;
                }
            }
            else
            {
                unsafe
                {
#if _REGEN
                    #region Compute
		                switch (dtype.as_numpy_dtype().GetTypeCode())
		                {
			                %foreach supported_dtypes,supported_dtypes_lowercase%
			                case NPTypeCode.#1: return new T[] {Converts.ChangeType<T>(*(#2*) buffer)};
			                %
			                case NPTypeCode.String: return new T[] {Converts.ChangeType<T>((string)this)};
			                default:
				                throw new NotSupportedException();
		                }
                    #endregion
#else
                    #region Compute
                    switch (tensor.dtype.as_numpy_dtype().GetTypeCode())
                    {
                        case NPTypeCode.Boolean: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Boolean); break; };
                        case NPTypeCode.Byte: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Byte); break; };
                        case NPTypeCode.Int16: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Int16); break; };
                        case NPTypeCode.UInt16: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.UInt16); break; };
                        case NPTypeCode.Int32: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Int32); break; };
                        case NPTypeCode.UInt32: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.UInt32); break; };
                        case NPTypeCode.Int64: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Int64); break; };
                        case NPTypeCode.UInt64: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.UInt64); break; };
                        case NPTypeCode.Char: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Char); break; };
                        case NPTypeCode.Double: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Double); break; };
                        case NPTypeCode.Single: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Single); break; };
                        case NPTypeCode.String: { dst = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.String); break; };
                        default:
                            throw new NotSupportedException();
                    }
                    #endregion
#endif
                }
            }
        }

        public static void ToSpan<T>(this Tensor tensor, Span<T> values) where T: unmanaged
        {
            if (typeof(T).as_dtype() == tensor.dtype)
            {
                unsafe
                {
                    var len = (long)tensor.size;
                    fixed (T* dst = values)
                    {
                        var src = (T*)tensor.buffer;
                        len *= ((long)tensor.itemsize);
                        System.Buffer.MemoryCopy(src, dst, len, len);
                    }
                }
            }
        }

        public static void ToArray<T>(this Tensor tensor, ref T[] array) where T : unmanaged
        {
            var arrayLen = (ulong)array.Length;
            if (arrayLen == 0 || arrayLen < tensor.size)
            {
                array = new T[tensor.size];
                arrayLen = tensor.size;
            }

            if (typeof(T).as_dtype() == tensor.dtype)
            {
                unsafe
                {
                    var len = (long)tensor.size;
                    fixed (T* dst = array)
                    {
                        var src = (T*)tensor.buffer;
                        len *= ((long)tensor.itemsize);
                        System.Buffer.MemoryCopy(src, dst, len, len);
                    }
                }
            }
            else
            {
                unsafe
                {
                    var len = (long)tensor.size;
                    fixed (T* dstRet = array)
                    {
                        T* dst = dstRet; //local stack copy
#if _REGEN
                        #region Compute
		                switch (dtype.as_numpy_dtype().GetTypeCode())
		                {
			                %foreach supported_dtypes,supported_dtypes_lowercase%
			                case NPTypeCode.#1: new UnmanagedMemoryBlock<#2>((#2*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                %
			                default:
				                throw new NotSupportedException();
		                }
                        #endregion
#else
                        #region Compute
                        switch (tensor.dtype.as_numpy_dtype().GetTypeCode())
                        {
                            case NPTypeCode.Boolean: new UnmanagedMemoryBlock<bool>((bool*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Byte: new UnmanagedMemoryBlock<byte>((byte*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Int16: new UnmanagedMemoryBlock<short>((short*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.UInt16: new UnmanagedMemoryBlock<ushort>((ushort*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Int32: new UnmanagedMemoryBlock<int>((int*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.UInt32: new UnmanagedMemoryBlock<uint>((uint*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Int64: new UnmanagedMemoryBlock<long>((long*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.UInt64: new UnmanagedMemoryBlock<ulong>((ulong*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Char: new UnmanagedMemoryBlock<char>((char*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Double: new UnmanagedMemoryBlock<double>((double*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Single: new UnmanagedMemoryBlock<float>((float*)tensor.buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.String: throw new NotSupportedException("Unable to convert from string to other dtypes");
                            default:
                                throw new NotSupportedException();
                        }
                        #endregion
#endif
                    }
                }
            }
        }
    }
}
