using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
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

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _dtype = dtype;
            _id = ops.uid();
        }
    }
}
