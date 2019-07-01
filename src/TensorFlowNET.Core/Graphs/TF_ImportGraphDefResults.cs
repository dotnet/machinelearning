using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_ImportGraphDefResults
    {
        public IntPtr return_tensors;
        public IntPtr return_nodes;
        public IntPtr missing_unused_key_names;
        public IntPtr missing_unused_key_indexes;
        public IntPtr missing_unused_key_names_data;
    }
}
