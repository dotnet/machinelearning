using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.ControlFlows;
using static Tensorflow.Python;

namespace Tensorflow.Operations
{
    /// <summary>
    /// Creates a `WhileContext`.
    /// </summary>
    public class WhileContext : ControlFlowContext
    {
        bool _back_prop=true;
        GradLoopState _grad_state =null;
        Tensor _maximum_iterations;
        int _parallel_iterations;
        bool _swap_memory;
        Tensor _pivot_for_pred;
        Tensor _pivot_for_body;
        Tensor[] _loop_exits;
        Tensor[] _loop_enters;

        public WhileContext(int parallel_iterations = 10,
            bool back_prop = true,
            bool swap_memory = false,
            string name = "while_context",
            GradLoopState grad_state = null,
            WhileContextDef context_def = null,
            string import_scope = null)
        {
            if (context_def != null)
            {
                _init_from_proto(context_def, import_scope: import_scope);
            }
            else
            {
                
            }

            _grad_state = grad_state;
        }

        private void _init_from_proto(WhileContextDef context_def, string import_scope = null)
        {
            var g = ops.get_default_graph();
            _name = ops.prepend_name_scope(context_def.ContextName, import_scope);
            if (!string.IsNullOrEmpty(context_def.MaximumIterationsName))
                _maximum_iterations = g.as_graph_element(ops.prepend_name_scope(context_def.MaximumIterationsName, import_scope)) as Tensor;
            _parallel_iterations = context_def.ParallelIterations;
            _back_prop = context_def.BackProp;
            _swap_memory = context_def.SwapMemory;
            _pivot_for_pred = g.as_graph_element(ops.prepend_name_scope(context_def.PivotForPredName, import_scope)) as Tensor;
            // We use this node to control constants created by the body lambda.
            _pivot_for_body = g.as_graph_element(ops.prepend_name_scope(context_def.PivotForBodyName, import_scope)) as Tensor;
            // The boolean tensor for loop termination condition.
            _pivot = g.as_graph_element(ops.prepend_name_scope(context_def.PivotName, import_scope)) as Tensor;
            // The list of exit tensors for loop variables.
            _loop_exits = new Tensor[context_def.LoopExitNames.Count];
            foreach (var (i, exit_name) in enumerate(context_def.LoopExitNames))
                _loop_exits[i] = g.as_graph_element(ops.prepend_name_scope(exit_name, import_scope)) as Tensor;
            // The list of enter tensors for loop variables.
            _loop_enters = new Tensor[context_def.LoopEnterNames.Count];
            foreach (var (i, enter_name) in enumerate(context_def.LoopEnterNames))
                _loop_enters[i] = g.as_graph_element(ops.prepend_name_scope(enter_name, import_scope)) as Tensor;

            __init__(values_def: context_def.ValuesDef, import_scope: import_scope);
        }

        public override WhileContext GetWhileContext()
        {
            return this;
        }


        public override GradLoopState grad_state => _grad_state;

        public override bool back_prop => _back_prop;

        public WhileContext from_proto(WhileContextDef proto, string import_scope)
        {
            var ret = new WhileContext(context_def: proto, import_scope: import_scope);

            ret.Enter();
            foreach (var nested_def in proto.NestedContexts)
                from_control_flow_context_def(nested_def, import_scope: import_scope);
            ret.Exit();
            return ret;
        }

        public object to_proto()
        {
            throw new NotImplementedException();
        }
    }
}
