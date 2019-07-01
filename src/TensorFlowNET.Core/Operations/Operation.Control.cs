using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow
{
    public partial class Operation
    {
        private ControlFlowContext _control_flow_context;

        /// <summary>
        /// Add this op to its control flow context.
        /// 
        /// This may add new ops and change this op's inputs. self.inputs must be
        /// available before calling this method.
        /// </summary>
        public void _control_flow_post_processing()
        {
            foreach(var input_tensor in inputs)
            {
                //TODO: implement below code dependency
                //control_flow_util.CheckInputFromValidContext(this, input_tensor.op);
            }

            if (_control_flow_context != null)
                _control_flow_context.AddOp(this);
        }

        public void _add_control_input(Operation op)
        {
            //c_api.TF_AddControlInput(_operDesc, op);
            c_api.AddControlInput(graph, _handle, op);
        }

        public void _add_control_inputs(Operation[] ops)
        {
            foreach (var op in ops)
                _add_control_input(op);
        }

        public void _set_control_flow_context(ControlFlowContext ctx)
        {
            _control_flow_context = ctx;
        }

        public ControlFlowContext _get_control_flow_context()
        {
            return _control_flow_context;
        }
    }
}
