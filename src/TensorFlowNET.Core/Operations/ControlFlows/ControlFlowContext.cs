using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations.ControlFlows;
using static Tensorflow.ControlFlowContextDef;
using static Tensorflow.Python;

namespace Tensorflow.Operations
{
    /// <summary>
    /// The base class for control flow context.
    /// 
    /// The usage pattern is a sequence of(Enter, Exit) followed by a final
    /// ExitResult.
    /// 
    /// We maintain the following state for control flow contexts during graph
    /// construction:
    /// 1. graph has _control_flow_context: the current context used to
    /// construct new nodes.Changed by ctxt.Enter() and ctxt.Exit()
    /// 2. op has _control_flow_context: the context to which the op belongs.
    /// Set at the time the op is created.Immutable.
    /// 3. A ControlFlowContext has _outer_context: the context in which this
    /// context is created.Set at the time a context is created.Immutable.
    /// 4. A ControlFlowContext has _context_stack.
    /// Pushed and popped by ctxt.Enter() and ctxt.Exit()
    /// </summary>
    public abstract class ControlFlowContext : IPython
    {
        /// <summary>
        /// The predicate tensor in this branch
        /// </summary>
        protected Tensor _pivot;
        public Tensor pivot
        {
            get => _pivot;
        }

        protected Stack<ControlFlowContext> _context_stack;
        protected ControlFlowContext _outer_context;

        protected Dictionary<string, ITensorOrOperation> _external_values;

        public ControlFlowContext()
        {
            _context_stack = new Stack<ControlFlowContext>();
        }

        public string name { get => _name; }
        protected string _name;

        public void __init__(ValuesDef values_def = null, string import_scope = null)
        {
            _outer_context = ops.get_default_graph()._get_control_flow_context();
            if (values_def != null)
                _init_values_from_proto(values_def, import_scope: import_scope);
        }

        public void __enter__()
        {
        }

        /// <summary>
        /// Initializes values and external_values from `ValuesDef` protocol buffer.
        /// </summary>
        /// <param name="values_def"></param>
        /// <param name="import_scope"></param>
        protected void _init_values_from_proto(ValuesDef values_def, string import_scope = null)
        {
            _external_values = new Dictionary<string, ITensorOrOperation>();
            foreach (var value in values_def.Values)
                _values.Add(value);
            var g = ops.get_default_graph();
            foreach(var value in values_def.ExternalValues)
            {
                var k = ops.prepend_name_scope(value.Key, import_scope);
                var v = value.Value;
                _external_values[k] = g.as_graph_element(ops.prepend_name_scope(v, import_scope));
            }

            var op_names = _values.Where(x => !_external_values.ContainsKey(x))
                .Select(x => x.Split(':')[0])
                .ToArray();

            foreach (var op in op_names)
                (g.as_graph_element(op) as Operation)._set_control_flow_context(this);
        }

        public void __exit__()
        {
        }

        /// <summary>
        /// Enter this control flow context.
        /// </summary>
        public virtual void Enter()
        {
            var graph = ops.get_default_graph();
            _context_stack.Push(graph._get_control_flow_context());
            graph._set_control_flow_context(this);
        }

        /// <summary>
        /// Exit this control flow context.
        /// </summary>
        public virtual void Exit()
        {
            var graph = ops.get_default_graph();
            var last_context = _context_stack.Pop();
            graph._set_control_flow_context(last_context);
        }

        /// <summary>
        /// Add `op` to the current context.
        /// </summary>
        public void AddOp(Operation op)
        {
            _AddOpInternal(op);
        }

        public ControlFlowContext outer_context { get { return _outer_context; } }
        public HashSet<string> values => _values;

        public virtual GradLoopState grad_state => throw new NotImplementedException("abstract method");

        public virtual bool back_prop => throw new NotImplementedException("abstract method");

        public virtual Tensor AddValue(Tensor val)
        {
            // to be overridden
            return null;
        }

        /// <summary>
        /// Notifies a scope about an operator added to an inner scope.
        /// </summary>
        /// <param name="op"></param>
        public virtual void AddInnerOp(Operation op)
        {
            if (_outer_context != null)
                _outer_context.AddInnerOp(op);
        }

        protected HashSet<string> _values = new HashSet<string>();

        /// <summary>
        /// Add `op` to the current context.
        /// </summary>
        protected virtual void _AddOpInternal(Operation op)
        {
            
        }

        protected bool OpInContext(Operation op)
        {
            return IsContainingContext(op._get_control_flow_context(), this);
        }

        /// <summary>
        /// Returns true if `maybe_containing_ctxt` is or contains `ctxt`.
        /// </summary>
        public static bool IsContainingContext(ControlFlowContext ctxt, ControlFlowContext maybe_containing_ctxt)
        {
            while (ctxt != maybe_containing_ctxt)
            {
                if (ctxt == null)
                    return false;
                ctxt = ctxt.outer_context;
            }
            return true;
        }


        protected virtual void _RemoveExternalControlEdges(Operation op)
        {
            var internal_control_inputs = op.control_inputs;
        }

        /// <summary>
        /// Return the while context containing this context
        /// </summary>
        public virtual WhileContext GetWhileContext()
        {
            if (_outer_context != null)
                return _outer_context.GetWhileContext();
            return null;
        }

        /// <summary>
        /// Deserializes `context_def` into the appropriate ControlFlowContext.
        /// </summary>
        /// <param name="context_def">ControlFlowContextDef proto</param>
        /// <param name="import_scope">Name scope to add</param>
        /// <returns>A ControlFlowContext subclass</returns>
        protected ControlFlowContext from_control_flow_context_def(ControlFlowContextDef context_def, string import_scope = "")
        {
            switch (context_def.CtxtCase)
            {
                case CtxtOneofCase.CondCtxt:
                    return new CondContext().from_proto(context_def.CondCtxt, import_scope: import_scope);
                case CtxtOneofCase.WhileCtxt:
                    return new WhileContext().from_proto(context_def.WhileCtxt, import_scope: import_scope);
            }

            throw new NotImplementedException($"Unknown ControlFlowContextDef field: {context_def.CtxtCase}");
        }

        public object to_proto()
        {
            throw new NotImplementedException();
        }


        public void Dispose()
        {
        }

    }
}
