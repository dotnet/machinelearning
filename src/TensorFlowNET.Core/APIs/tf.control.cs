using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static _ControlDependenciesController control_dependencies(Operation[] control_inputs) 
            => ops.control_dependencies(control_inputs);
    }
}
