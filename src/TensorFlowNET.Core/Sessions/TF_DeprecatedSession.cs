using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow.Sessions
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_DeprecatedSession
    {
        Session session;
    }
}
