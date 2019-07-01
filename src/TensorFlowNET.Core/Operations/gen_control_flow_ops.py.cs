using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class gen_control_flow_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation no_op(string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("NoOp", name, null);

            return _op;
        }

        /// <summary>
        /// Forwards `data` to the output port determined by `pred`.
        /// 
        /// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
        /// the data goes to `output_false`.
        /// 
        /// See also `RefSwitch` and `Merge`.
        /// </summary>
        /// <param name="data">A `Tensor`. The tensor to be forwarded to the appropriate output.</param>
        /// <param name="pred">A `Tensor` of type `bool`.
        /// A scalar that specifies which output port will receive data.
        /// </param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns>A tuple of `Tensor` objects (output_false, output_true).
        /// 
        /// output_false: A `Tensor`. Has the same type as `data`.
        /// output_true: A `Tensor`. Has the same type as `data`.
        /// </returns>
        public static Tensor[] @switch(Tensor data, Tensor pred, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Switch", name, new { data, pred });
            var _inputs_flat = _op.inputs;
            var _attrs = ("T", _op.get_attr("T"));
            // TODO: missing original code
            //_execute.record_gradient("Switch", _inputs_flat, _attrs, _result, name);
            return new []{_op.outputs[0], _op.outputs[1]};
        }

        public static (Tensor, Tensor) merge(Tensor[] inputs, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Merge", name, new { inputs });

            return (_op.outputs[0], _op.outputs[1]);
        }
    }
}
