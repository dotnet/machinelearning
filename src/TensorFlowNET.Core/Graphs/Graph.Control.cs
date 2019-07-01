using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;
using Tensorflow.Operations;

namespace Tensorflow
{
    public partial class Graph
    {
        // Current control flow context. It could be either CondContext or WhileContext 
        public ControlFlowContext _control_flow_context;

        // represents the nested with(...) statements
        public List<_ControlDependenciesController> _control_dependencies_stack { get; set; } = new List<_ControlDependenciesController>();

        /// <summary>
        /// For an op that takes `input_ops` as inputs, compute control inputs.
        /// </summary>
        /// <param name="input_ops">The data input ops for an op to be created.</param>
        /// <returns>A list of control inputs for the op to be created.</returns>
        private ITensorOrOperation[] _control_dependencies_for_inputs(ITensorOrOperation[] input_ops)
        {
            var ret = new List<ITensorOrOperation>();

            foreach (var controller in _control_dependencies_stack)
            {
                bool dominated = false;
                // If any of the input_ops already depends on the inputs from controller,
                // we say that the new op is dominated (by that input), and we therefore
                // do not need to add control dependencies for this controller's inputs.
                foreach (var op in input_ops)
                {
                    if (controller.op_in_group(op))
                    {
                        dominated = true;
                        break;
                    }
                }

                if (!dominated)
                    ret.AddRange(controller.control_inputs.Where(x => !input_ops.Contains(x)));
            }

            return ret.ToArray();
        }

        /// <summary>
        /// Returns a context manager that specifies control dependencies.
        /// 
        /// Use with the `with` keyword to specify that all operations constructed
        /// within the context should have control dependencies on
        /// `control_inputs`. 
        /// </summary>
        public _ControlDependenciesController control_dependencies(ITensorOrOperation[] control_inputs)
            => control_dependencies(control_inputs == null ? null : control_inputs.OfType<object>().ToArray());

        /// <summary>
        /// Returns a context manager that specifies control dependencies.
        /// 
        /// Use with the `with` keyword to specify that all operations constructed
        /// within the context should have control dependencies on
        /// `control_inputs`. 
        /// </summary>
        public _ControlDependenciesController control_dependencies(object[] control_inputs)
        {
            if (control_inputs == null)
                return new _ControlDependenciesController(this, null);

            var control_ops = new List<ITensorOrOperation>();
            foreach (var c in control_inputs)
            {
                switch (c)
                {
                    // TODO: implement IndexedSlices
                    //case IndexedSlices islice:
                    //    control_ops.Add(islice.op);
                    //    break;
                    case Tensor t:                       
                        control_ops.Add(t.op);
                        break;
                    case Operation op:
                        control_ops.Add(op);
                        break;
                    default:
                        var t1 = _as_graph_element(c);
                        if (t1 == null)
                            throw new TypeError($"Control input must be Operation or Tensor:{c}");
                        control_ops.Add(t1.op);
                        break;
                }
            }
            return new _ControlDependenciesController(this, control_ops);
        }

        /// <summary>
        /// Returns the current control flow context.
        /// </summary>
        /// <returns>A context object.</returns>
        public ControlFlowContext _get_control_flow_context()
        {
            return _control_flow_context;
        }

        /// <summary>
        /// Sets the current control flow context.
        /// </summary>
        /// <param name="ctx">a context object.</param>
        public void _set_control_flow_context(ControlFlowContext ctx)
        {
            _control_flow_context = ctx;
        }

        public void _push_control_dependencies_controller(_ControlDependenciesController controller)
        {
            _control_dependencies_stack.Add(controller);
        }

        public void _pop_control_dependencies_controller(_ControlDependenciesController controller)
        {
            _control_dependencies_stack.RemoveAt(_control_dependencies_stack.Count-1);
        }

        /// <summary>
        /// Record that the given op depends on all registered control dependencies.
        /// </summary>
        public void _record_op_seen_by_control_dependencies(Operation op)
        {
            foreach (var controller in _control_dependencies_stack)
                controller.add_op(op);
        }
    }
}
