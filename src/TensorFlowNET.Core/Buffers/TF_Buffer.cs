using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Buffer
    {
        public IntPtr data;
        public ulong length;
        public IntPtr data_deallocator;
    }
}
