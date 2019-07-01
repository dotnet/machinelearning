using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// C API for TensorFlow.
    /// Port from tensorflow\c\c_api.h
    /// 
    /// The API leans towards simplicity and uniformity instead of convenience
    /// since most usage will be by language specific wrappers.
    /// 
    /// The params type mapping between c_api and .NET
    /// TF_XX** => ref IntPtr (TF_Operation** op) => (ref IntPtr op)
    /// TF_XX* => IntPtr (TF_Graph* graph) => (IntPtr graph)
    /// struct => struct (TF_Output output) => (TF_Output output)
    /// struct* => struct[] (TF_Output* output) => (TF_Output[] output)
    /// struct* => struct* for ref
    /// const char* => string
    /// int32_t => int
    /// int64_t* => long[]
    /// size_t* => ulong[]
    /// size_t* => ref ulong
    /// void* => IntPtr
    /// string => IntPtr c_api.StringPiece(IntPtr)
    /// unsigned char => byte
    /// </summary>
    public partial class c_api
    {
        public const string TensorFlowLibName = "tensorflow";

        public static string StringPiece(IntPtr handle)
        {
            return handle == IntPtr.Zero ? String.Empty : Marshal.PtrToStringAnsi(handle);
        }

        public delegate void Deallocator(IntPtr data, IntPtr size, ref bool deallocator);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_Version();
    }
}
