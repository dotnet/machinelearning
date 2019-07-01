using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static class gen_resource_variable_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation assign_variable_op(Tensor resource, Tensor value, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("AssignVariableOp", name, new { resource, value });

            return _op;
        }
    }
}
