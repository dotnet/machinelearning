using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow.Functions
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Function
    {
        FunctionDef fdef;
    }
}
