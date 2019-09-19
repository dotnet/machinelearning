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
        public static void ToArray<T>(this Tensor tensor, T[] array) where T: unmanaged
        {
            var size = array.Length;
            if (typeof(T).as_dtype() == tensor.dtype)
            {
                if (tensor.NDims == 0 && size == 1)
                {
                    unsafe
                    {
                        array[0] = *(T*)tensor.buffer;
                    }
                }

                unsafe
                {
                    var len = (long)size;
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
                if (tensor.NDims == 0 && size == 1) //is it a scalar?
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
                            case NPTypeCode.Boolean: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Boolean); break;};
                            case NPTypeCode.Byte: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Byte); break; };
                            case NPTypeCode.Int16: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Int16); break; };
                            case NPTypeCode.UInt16: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.UInt16); break; };
                            case NPTypeCode.Int32: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Int32); break; };
                            case NPTypeCode.UInt32: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.UInt32); break; };
                            case NPTypeCode.Int64: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Int64); break; };
                            case NPTypeCode.UInt64: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.UInt64); break; };
                            case NPTypeCode.Char: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Char); break; };
                            case NPTypeCode.Double: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Double); break; };
                            case NPTypeCode.Single: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.Single); break; };
                            case NPTypeCode.String: { array[0] = Converts.ChangeType<T>(tensor.buffer, NPTypeCode.String); break; };
                            default:
                                throw new NotSupportedException();
                        }
                        #endregion
#endif
                    }
                }

                unsafe
                {
                    var len = (long)size;
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
