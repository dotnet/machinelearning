using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteBuffer(IntPtr buffer);

        /// <summary>
        /// Useful for passing *out* a protobuf.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewBuffer();

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GetBuffer(TF_Buffer buffer);

        /// <summary>
        /// Makes a copy of the input and sets an appropriate deallocator.  Useful for
        /// passing in read-only, input protobufs.
        /// </summary>
        /// <param name="proto">const void*</param>
        /// <param name="proto_len">size_t</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewBufferFromString(IntPtr proto, ulong proto_len);
    }
}
