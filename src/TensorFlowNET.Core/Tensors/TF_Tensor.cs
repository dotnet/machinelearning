using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Tensor
    {
        public TF_DataType dtype;
        public IntPtr shape;
        public IntPtr buffer;
    }
}
