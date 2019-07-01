using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public enum GateGradientType
    {
        GATE_NONE = 0,
        GATE_OP = 1,
        GATE_GRAPH = 2
    }
}
