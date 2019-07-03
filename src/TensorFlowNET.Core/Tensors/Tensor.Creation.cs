using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.c_api;

namespace Tensorflow
{
    public partial class Tensor
    {
        /// <summary>
        /// if original buffer is free.
        /// </summary>
        private bool deallocator_called;

        public Tensor(IntPtr handle)
        {
            _handle = handle;
        }

        public Tensor(NDArray nd, TF_DataType? tensorDType = null)
        {
            _handle = Allocate(nd, tensorDType: tensorDType);
        }

        public unsafe Tensor(byte[] buffer)
        {
            var size = c_api.TF_StringEncodedSize((UIntPtr)buffer.Length);
            _handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

            IntPtr tensor = c_api.TF_TensorData(_handle);
            Marshal.WriteInt64(tensor, 0);
            fixed (byte* src = &buffer[0])
                c_api.TF_StringEncode(src, (UIntPtr)buffer.Length, (sbyte*)(tensor + sizeof(Int64)), size, status);

            status.Check(true);
        }

        private IntPtr Allocate(NDArray nd, TF_DataType? tensorDType = null)
        {
            if (tensorDType == TF_DataType.TF_STRING &&
                nd.dtype.Name == "Byte")
            {
                return new Tensor(nd.Data<byte>());
            }

            IntPtr dotHandle = IntPtr.Zero;
            ulong size = 0;

            if (nd.dtype.Name != "String")
            {
                dotHandle = Marshal.AllocHGlobal(nd.dtypesize * nd.size);
                size = (ulong)(nd.size * nd.dtypesize);
            }

            var dataType = ToTFDataType(nd.dtype);
            // shape
            var dims = nd.shape.Select(x => (long)x).ToArray();
            var nd1 = nd.ravel();
            switch (nd.dtype.Name)
            {
                case "Boolean":
                    var boolVals = Array.ConvertAll(nd1.Data<bool>(), x => Convert.ToByte(x));
                    Marshal.Copy(boolVals, 0, dotHandle, nd.size);
                    break;
                case "Int16":
                    Marshal.Copy(nd1.Data<short>(), 0, dotHandle, nd.size);
                    break;
                case "Int32":
                    Marshal.Copy(nd1.Data<int>(), 0, dotHandle, nd.size);
                    break;
                case "Int64":
                    Marshal.Copy(nd1.Data<long>(), 0, dotHandle, nd.size);
                    break;
                case "Single":
                    Marshal.Copy(nd1.Data<float>(), 0, dotHandle, nd.size);
                    break;
                case "Double":
                    Marshal.Copy(nd1.Data<double>(), 0, dotHandle, nd.size);
                    break;
                case "Byte":
                    Marshal.Copy(nd1.Data<byte>(), 0, dotHandle, nd.size);
                    break;
                case "String":
                    return new Tensor(UTF8Encoding.UTF8.GetBytes(nd.Data<string>(0)));
                default:
                    throw new NotImplementedException($"Marshal.Copy failed for {nd.dtype.Name}.");
            }


            
            // Free the original buffer and set flag
            Deallocator deallocator = (IntPtr values, IntPtr len, ref bool closure) =>
            {
                Marshal.FreeHGlobal(values);
                closure = true;
            };

            var tfHandle = c_api.TF_NewTensor(dataType,
                dims,
                dims.Length,
                dotHandle,
                (UIntPtr)size,
                deallocator,
                ref deallocator_called);

            return tfHandle;
        }

        private unsafe IntPtr Allocate(TF_DataType tensorDType, object value)
        {
            Deallocator deallocator = (IntPtr values, IntPtr len, ref bool closure) =>
            {
                Marshal.FreeHGlobal(values);
                closure = true;
            };
            if (tensorDType == TF_DataType.TF_BOOL)
            {
                var v = (bool*)Marshal.AllocHGlobal(sizeof(bool));
                *v = (bool)value;
                var size = sizeof(bool);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_UINT8)
            {
                var v = (int*)Marshal.AllocHGlobal(sizeof(byte));
                *v = (int)value;
                var size = sizeof(int);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_INT16)
            {
                var v = (ushort*)Marshal.AllocHGlobal(sizeof(ushort));
                *v = (ushort)value;
                var size = sizeof(ushort);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_COMPLEX128)
            {
                var v = (Complex*)Marshal.AllocHGlobal(sizeof(Complex));
                *v = (Complex)value;
                var size = sizeof(Complex);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_DOUBLE)
            {
                var v = (double*)Marshal.AllocHGlobal(sizeof(double));
                *v = (double)value;
                var size = sizeof(double);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_FLOAT)
            {
                var v = (float*)Marshal.AllocHGlobal(sizeof(float));
                *v = (float)value;
                var size = sizeof(float);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_INT32)
            {
                var v = (int*)Marshal.AllocHGlobal(sizeof(int));
                *v = (int)value;
                var size = sizeof(int);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_INT64)
            {
                var v = (long*)Marshal.AllocHGlobal(sizeof(long));
                *v = (long)value;
                var size = sizeof(long);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_INT8)
            {
                var v = (sbyte*)Marshal.AllocHGlobal(sizeof(sbyte));
                *v = (sbyte)value;
                var size = sizeof(sbyte);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_INT16)
            {
                var v = (short*)Marshal.AllocHGlobal(sizeof(short));
                *v = (short)value;
                var size = sizeof(short);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_UINT32)
            {
                var v = (long*)Marshal.AllocHGlobal(sizeof(long));
                *v = (long)value;
                var size = sizeof(long);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_UINT64)
            {
                var v = (float*)Marshal.AllocHGlobal(sizeof(float));
                *v = (float)value;
                var size = sizeof(float);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_UINT16)
            {
                var v = (ushort*)Marshal.AllocHGlobal(sizeof(ushort));
                *v = (ushort)value;
                var size = sizeof(ushort);
                //   private static extern IntPtr TF_NewTensor(TF_DataType dataType, IntPtr zeroDims, int num_dims, IntPtr data, System.UIntPtr len, Deallocator deallocator, ref bool deallocator_arg);
                return c_api.TF_NewTensor(tensorDType, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)size, deallocator: deallocator, deallocator_arg: ref deallocator_called);
            }
            else if (tensorDType == TF_DataType.TF_STRING)
            {
                return new Tensor(UTF8Encoding.UTF8.GetBytes((String)value));
            }
            else
            {
                throw new Exception($"Unsupported type");
            }
            return IntPtr.Zero;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _dtype = dtype;
            _id = ops.uid();
        }
    }
}
