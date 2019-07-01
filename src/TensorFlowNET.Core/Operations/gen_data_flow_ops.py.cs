using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_data_flow_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor dynamic_stitch(Tensor[] indices, Tensor[] data, string name = null)
        {
            var _attr_N = indices.Length;
            var _op = _op_def_lib._apply_op_helper("DynamicStitch", name, new { indices, data });

            return _op.outputs[0];
        }
    }
}
