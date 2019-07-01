using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Allocate and return a new Tensor.
        /// </summary>
        /// <param name="dtype">TF_DataType</param>
        /// <param name="dims">const int64_t*</param>
        /// <param name="num_dims">int</param>
        /// <param name="len">size_t</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_AllocateTensor(TF_DataType dtype, IntPtr dims, int num_dims, UIntPtr len);

        /// <summary>
        /// returns the sizeof() for the underlying type corresponding to the given TF_DataType enum value.
        /// </summary>
        /// <param name="dt"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern ulong TF_DataTypeSize(TF_DataType dt);

        /// <summary>
        /// Destroy a tensor.
        /// </summary>
        /// <param name="tensor"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteTensor(IntPtr tensor);

        /// <summary>
        /// Return the length of the tensor in the "dim_index" dimension.
        /// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="dim_index"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern long TF_Dim(IntPtr tensor, int dim_index);

        /// <summary>
        /// Return a new tensor that holds the bytes data[0,len-1]
        /// </summary>
        /// <param name="dataType"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        /// <param name="data"></param>
        /// <param name="len">num_bytes, ex: 6 * sizeof(float)</param>
        /// <param name="deallocator"></param>
        /// <param name="deallocator_arg"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewTensor(TF_DataType dataType, long[] dims, int num_dims, IntPtr data, UIntPtr len, Deallocator deallocator, ref bool deallocator_arg);

        /// <summary>
        /// Return the number of dimensions that the tensor has.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_NumDims(IntPtr tensor);

        /// <summary>
        /// Return the size of the underlying data in bytes.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern ulong TF_TensorByteSize(IntPtr tensor);

        /// <summary>
        /// Return a pointer to the underlying data buffer.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_TensorData(IntPtr tensor);

        /// <summary>
        /// Deletes `tensor` and returns a new TF_Tensor with the same content if
        /// possible. Returns nullptr and leaves `tensor` untouched if not.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_TensorMaybeMove(IntPtr tensor);

        /// <summary>
        /// Return the type of a tensor element.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_DataType TF_TensorType(IntPtr tensor);

        /// <summary>
        /// Return the size in bytes required to encode a string `len` bytes long into a
        /// TF_STRING tensor.
        /// </summary>
        /// <param name="len">size_t</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern UIntPtr TF_StringEncodedSize(UIntPtr len);

        /// <summary>
        /// Encode the string `src` (`src_len` bytes long) into `dst` in the format
        /// required by TF_STRING tensors. Does not write to memory more than `dst_len`
        /// bytes beyond `*dst`. `dst_len` should be at least
        /// TF_StringEncodedSize(src_len).
        /// </summary>
        /// <param name="src">const char*</param>
        /// <param name="src_len">size_t</param>
        /// <param name="dst">char*</param>
        /// <param name="dst_len">size_t</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>On success returns the size in bytes of the encoded string.</returns>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe ulong TF_StringEncode(byte* src, UIntPtr src_len, sbyte* dst, UIntPtr dst_len, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe ulong TF_StringEncode(IntPtr src, ulong src_len, IntPtr dst, ulong dst_len, IntPtr status);

        /// <summary>
        /// Decode a string encoded using TF_StringEncode.
        /// </summary>
        /// <param name="src">const char*</param>
        /// <param name="src_len">size_t</param>
        /// <param name="dst">const char**</param>
        /// <param name="dst_len">size_t*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern ulong TF_StringDecode(IntPtr src, ulong src_len, IntPtr dst, ref ulong dst_len, IntPtr status);
    }
}
