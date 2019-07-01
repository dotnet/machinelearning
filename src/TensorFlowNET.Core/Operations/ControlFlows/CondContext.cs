using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations.ControlFlows;
using static Tensorflow.Python;

namespace Tensorflow.Operations
{
    /// <summary>
    /// The context for the conditional construct.
    /// </summary>
    public class CondContext : ControlFlowContext, IProtoBuf<CondContextDef, CondContext>
    {


        /// <summary>
        /// The boolean tensor for the cond predicate
        /// </summary>
        private Tensor _pred;

        public Tensor pred => _pred;

        /// <summary>
        /// 0 or 1 representing this branch
        /// </summary>
        private int _branch;

        private Dictionary<string, Tensor> _external_values = new Dictionary<string, Tensor>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pred">The `boolean` tensor for the conditional predicate.</param>
        /// <param name="pivot">The predicate tensor in this branch.</param>
        /// <param name="branch">0 or 1 representing this branch.</param>
        /// <param name="name">Name of the `CondContext` python object.</param>
        /// <param name="context_def"></param>
        /// <param name="import_scope"></param>
        public CondContext(Tensor pred = null,
            Tensor pivot = null,
            int branch = 0,
            string name = "cond_text",
            CondContextDef context_def = null,
            string import_scope = null)
        {
            if (pred == null && context_def == null) return;

            _name = ops.get_default_graph().unique_name(name);
            if (context_def != null)
            {
                _init_from_proto(context_def, import_scope: import_scope);
            }
            else
            {
                // Initializes the default fields.
                base.__init__();
                _pred = pred;
                _pivot = pivot;
                _branch = branch; // 0 or 1 representing this branch
                // Values considered to have been already seen in this context. pred is not
                // included in this context.
                _values.Add(pred.name);
                _external_values[pred.name] = pred;
                _values.Add(pivot.name);
                pivot.op._set_control_flow_context(this);
            }
        }

        private void _init_from_proto(CondContextDef context_def, string import_scope = null)
        {
            var g = ops.get_default_graph();
            _name = ops.prepend_name_scope(context_def.ContextName, import_scope);
            var p1 = ops.prepend_name_scope(context_def.PredName, import_scope);
            _pred = g.as_graph_element(p1) as Tensor;
            var p2 = ops.prepend_name_scope(context_def.PivotName, import_scope);
            _pivot = g.as_graph_element(p2) as Tensor;
            _branch = context_def.Branch;
            __init__(values_def: context_def.ValuesDef, import_scope: import_scope);
        }

        /// <summary>
        /// Add `val` to the current context and its outer context recursively.
        /// </summary>
        /// <param name="val"></param>
        public override Tensor AddValue(Tensor val)
        {
            Tensor result = null;
            if (_values.Contains(val.name))
            {
                // Use the real value if it comes from outer context. This is needed in
                // particular for nested conds.
                if (_external_values.ContainsKey(val.name))
                    result = _external_values[val.name];

                result = result == null ? val : result;
            }
            else
            {
                result = val;
                _values.Add(val.name);
                // TODO: _outer_context                
                if (_outer_context != null)
                {
                    result = _outer_context.AddValue(val);
                    _values.Add(result.name);
                    _external_values[result.name] = result;
                }
                
                with(ops.control_dependencies(null), ctrl =>
                {
                    var results = control_flow_ops._SwitchRefOrTensor(result, _pred);
                    result = results[_branch];
                    if (_outer_context != null)
                        _outer_context.AddInnerOp(result.op);
                });

                result.op.graph.prevent_fetching(result.op);
                result.op._set_control_flow_context(this);

                // Mark Switch output as seen by this context and any outer contexts,
                // just like what we do for normal op outputs in _AddOpInternal() below.
                ControlFlowContext ctxt = this;
                while (ctxt != null)
                {
                    ctxt.values.Add(result.name);
                    ctxt = ctxt.outer_context;
                }
                _external_values[val.name] = result;
            }
            return result;
        }

        /// <summary>
        /// Add the subgraph defined by fn() to the graph.
        /// </summary>
        public (T, Tensor) BuildCondBranch<T>(Func<T> fn)
        {
            // Add the subgraph defined by fn() to the graph.
            var pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);
            var original_result = fn();
            var post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);

            //TODO: port this chunck of missing code:
            /*
            if len(post_summaries) > len(pre_summaries):
                new_summaries = post_summaries[len(pre_summaries):]
                summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
                summary_ref[:] = pre_summaries
                with ops.control_dependencies(new_summaries):
                if original_result is None:
                    return no_op(), None
                else:
                    original_result = nest.map_structure(array_ops.identity,
                                                        original_result)
            */
            if (original_result == null)
                return (original_result, null);

            switch (original_result)
            {
                case Tensor result:
                    return (original_result, _BuildCondTensor(result));
                case Operation op:
                    return (original_result, _BuildCondTensor(op));
                case float[] fv:
                    {
                        var result = ops.convert_to_tensor(fv[0]);
                        return (original_result, _BuildCondTensor(result));
                    }
                default:
                    return (original_result, null);
            }
        }

        public (T[], Tensor[]) BuildCondBranch<T>(Func<T[]> fn)
        {
            // Add the subgraph defined by fn() to the graph.
            var pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);
            var original_result = fn();
            var post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION);

            switch (original_result)
            {
                case Tensor[] results:
                    return (original_result, results.Select(_BuildCondTensor).ToArray());
                case Operation[] results:
                    return (original_result, results.Select(_BuildCondTensor).ToArray());
                case float[] fv:
                    var result = ops.convert_to_tensor(fv[0]);
                    return (original_result, new Tensor[] { result });
                default:
                    return (original_result, new Tensor[0]);
            }
        }

        private Tensor _BuildCondTensor(ITensorOrOperation v)
        {
            switch (v)
            {
                case Operation op:
                    // Use pivot as the proxy for this op.
                    return control_flow_ops.with_dependencies(new Operation[] { op }, _pivot);
                case Tensor t:
                    return _ProcessOutputTensor(t);
                default:
                    return _ProcessOutputTensor(ops.convert_to_tensor(v));

            }
        }

        /// <summary>
        /// Process an output tensor of a conditional branch.
        /// </summary>
        private Tensor _ProcessOutputTensor(Tensor val)
        {
            var real_val = val;
            if (!_values.Contains(val.name))
            {
                // Handle the special case of lambda: x
                _values.Add(val.name);
                if (_outer_context != null)
                {
                    real_val = _outer_context.AddValue(val);
                    _values.Add(real_val.name);
                    _external_values[real_val.name] = real_val;
                }
                var results = control_flow_ops._SwitchRefOrTensor(real_val, _pred);
                real_val = results[_branch];
                _external_values[val.name] = real_val;
            }
            else
            {
                Tensor external_val = null;
                if (_external_values.ContainsKey(val.name))
                    external_val = _external_values[val.name];
                if (external_val != null)
                    real_val = external_val;
            }
            return real_val;
        }

        protected override void _AddOpInternal(Operation op)
        {
            if (op.inputs.Length == 0)
            {
                //If we're in a while loop, remove any control inputs from outside the
                // loop.
                _RemoveExternalControlEdges(op);
                if (!op.control_inputs.Any(input_op => OpInContext(input_op)))
                    op._add_control_input(_pivot.op);
            }
            else
            {
                // Make each input to 'op' available in this CondContext. If an input is
                // already part of this context there's nothing to do, but if it's
                // external, AddValue() will handle adding the appropriate Switch node and
                // other bookkeeping.
                for (int index = 0; index < op.inputs.Length; index++)
                {
                    var x = op.inputs[index];
                    Tensor real_x = null;
                    if (op.type == "Merge" && x.op.type == "NextIteration")
                    {
                        //# Edge case: if we're importing a while loop inside this CondContext,
                        //# AddValue() will not correctly handle the NextIteration inputs to
                        //# Merge node. The problem is that the NextIteration should also be
                        //# part of this context, but if we're importing it won't have been
                        //# processed and added to the context yet, so AddValue() will try to
                        //# add a Switch which results in an invalid graph. Instead, we use the
                        //# NextIteration input as-is here, and it will eventually be added to
                        //# the context via AddOp().
                        real_x = x;
                    }
                    else
                    {
                        real_x = AddValue(x);
                    }
                    if (real_x != x)
                        op._update_input(index, real_x);
                }
                // Remove any external control dependency on this op.
                _RemoveExternalControlEdges(op);
                // TODO: implement below code dependencies
                //if (op.graph._is_function(op.type) || op.type == "SymbolicGradient")
                //    op._add_control_input(_pivot.op);
            }

            // Mark op's outputs as seen by this context and any outer contexts.
            var output_names = op.outputs.Select(x => x.name).ToArray();
            ControlFlowContext ctxt = this;
            while (ctxt != null)
            {
                foreach (var name in output_names)
                    ctxt.values.Add(name);
                ctxt = ctxt.outer_context;
            }

            if (_outer_context != null || !control_flow_ops.IsLoopExit(op))
                op.graph.prevent_fetching(op);

            if (_outer_context != null)
                _outer_context.AddInnerOp(op);
        }

        public override GradLoopState grad_state
        {
            get
            {
                var whc = GetWhileContext();
                if (whc != null)
                    return whc.grad_state;
                return null;
            }
        }

        public override bool back_prop
        {
            get
            {
                var whc = GetWhileContext();
                if (whc != null)
                    return whc.back_prop;
                return false;
            }
        }

        public CondContextDef to_proto(string export_scope)
        {
            throw new NotImplementedException();
        }

        public CondContext from_proto(CondContextDef proto, string import_scope)
        {
            var ret = new CondContext(context_def: proto, import_scope: import_scope);

            ret.Enter();
            foreach (var nested_def in proto.NestedContexts)
                from_control_flow_context_def(nested_def, import_scope: import_scope);
            ret.Exit();
            return ret;
        }
    }
}