using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using Tensorflow.Operations;

namespace Tensorflow
{
    /// <summary>
    /// Context manager for `control_dependencies()`
    /// </summary>
    public class _ControlDependenciesController : IPython
    {
        private Graph _graph;
        private List<ITensorOrOperation> _control_inputs_val;
        private List<ITensorOrOperation> _seen_nodes;
        private List<_ControlDependenciesController> _old_stack;
        private bool _new_stack;
        private ControlFlowContext _old_control_flow_context;

        public ITensorOrOperation[] control_inputs => _control_inputs_val.ToArray();

        /// <summary>
        /// Create a new `_ControlDependenciesController`.
        /// 
        /// A `_ControlDependenciesController` is the context manager for
        /// `with tf.control_dependencies()` blocks.These normally nest,
        /// as described in the documentation for `control_dependencies()`.
        /// 
        /// The `control_inputs` argument list control dependencies that must be
        /// added to the current set of control dependencies.Because of
        /// uniquification the set can be empty even if the caller passed a list of
        /// ops.The special value `None` indicates that we want to start a new
        /// empty set of control dependencies instead of extending the current set.
        /// 
        /// In that case we also clear the current control flow context, which is an
        /// additional mechanism to add control dependencies.
        /// </summary>
        /// <param name="graph">The graph that this controller is managing.</param>
        /// <param name="control_inputs">List of ops to use as control inputs in addition
        /// to the current control dependencies.None to indicate that
        /// the dependencies should be cleared.
        /// </param>
        public _ControlDependenciesController(Graph graph, List<ITensorOrOperation> control_inputs)
        {
            _graph = graph;
            if (control_inputs == null)
            {
                _control_inputs_val = new List<ITensorOrOperation>();
                _new_stack = true;
            }
            else
            {
                _control_inputs_val = control_inputs;
                _new_stack = false;
            }

            _seen_nodes = new List<ITensorOrOperation>();
        }

        public void add_op(ITensorOrOperation op)
        {
            _seen_nodes.Add(op);
        }

        public bool op_in_group(ITensorOrOperation op)
        {
            return _seen_nodes.Contains(op);
        }

        public void __enter__()
        {
            if (_new_stack)
            {
                // Clear the control_dependencies graph.
                _old_stack = _graph._control_dependencies_stack;
                _graph._control_dependencies_stack = new List<_ControlDependenciesController>();

                // Clear the control_flow_context too.
                _old_control_flow_context = _graph._get_control_flow_context();
                _graph._set_control_flow_context(null);
            }

            _graph._push_control_dependencies_controller(this);
        }

        public void __exit__()
        {
            _graph._pop_control_dependencies_controller(this);
            if (_new_stack)
            {
                _graph._control_dependencies_stack = _old_stack;
                _graph._set_control_flow_context(_old_control_flow_context);
            }
        }

        public void Dispose()
        {
            
        }
    }
}
