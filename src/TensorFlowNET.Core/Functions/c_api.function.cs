using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Write out a serialized representation of `func` (as a FunctionDef protocol
        /// message) to `output_func_def` (allocated by TF_NewBuffer()).
        /// `output_func_def`'s underlying buffer will be freed when TF_DeleteBuffer()
        /// is called.
        /// </summary>
        /// <param name="func"></param>
        /// <param name="output_func_def"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_FunctionToFunctionDef(IntPtr func, IntPtr output_func_def, IntPtr status);


    }
}
