using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_io_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation save_v2(Tensor prefix, string[] tensor_names, string[] shape_and_slices, Tensor[] tensors, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("SaveV2", name: name, args: new { prefix, tensor_names, shape_and_slices, tensors });

            return _op;
        }

        public static Tensor[] restore_v2(Tensor prefix, string[] tensor_names, string[] shape_and_slices, TF_DataType[] dtypes, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("RestoreV2", name: name, args: new { prefix, tensor_names, shape_and_slices, dtypes });

            return _op.outputs;
        }

        public static Tensor read_file(string filename, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ReadFile", name: name, args: new { filename });

            return _op.outputs[0];
        }
    }
}
