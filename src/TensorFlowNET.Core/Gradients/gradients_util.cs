using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class gradients_util
    {
        public static Tensor[] _GradientsHelper(Tensor[] ys,
            Tensor[] xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int aggregation_method = 0,
            Tensor[] stop_gradients = null,
            Graph src_graph = null)
        {
            if (src_graph == null)
                src_graph = ops.get_default_graph();

            // If src_graph is a _FuncGraph (i.e. a function body), gather it and all
            // ancestor graphs. This is necessary for correctly handling captured values.
            var curr_graph = src_graph;

            if (stop_gradients == null)
                stop_gradients = new Tensor[0];
            if (grad_ys == null)
                grad_ys = new Tensor[ys.Length];

            // Iterate over the collected ops.
            /**
             * grads: op => list of gradients received on each output endpoint of the
             * op.  The gradients for each endpoint are initially collected as a list.
             * When it is time to call the op's gradient function, for each endpoint we
             * aggregate the list of received gradients into a Add() Operation if there
             * is more than one.
             **/
            var grads = new Dictionary<string, List<List<Tensor>>>();

            with(ops.name_scope(name, "gradients",
                values: ys.Concat(xs).Concat(stop_gradients).Concat(grad_ys)), scope =>
                {
                    string grad_scope = scope;
                    // Get a uid for this call to gradients that can be used to help
                    // cluster ops for compilation.
                    var gradient_uid = ops.get_default_graph().unique_name("uid");
                    ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name: "y");
                    xs = ops.internal_convert_n_to_tensor_or_indexed_slices(xs, name: "x", as_ref: true);
                    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops, gradient_uid);

                    /** 
                     * The approach we take here is as follows: Create a list of all ops in the
                     * subgraph between the ys and xs.  Visit these ops in reverse order of ids
                     * to ensure that when we visit an op the gradients w.r.t its outputs have
                     * been collected.  Then aggregate these gradients if needed, call the op's
                     * gradient function, and add the generated gradients to the gradients for
                     * its input.
                     **/

                    // Initialize the pending count for ops in the connected subgraph from ys
                    // to the xs.
                    var to_ops = ys.Select(x => x.op).ToList();
                    var from_ops = xs.Select(x => x.op).ToList();
                    var stop_gradient_ops = stop_gradients.Select(x => x.op).ToList();
                    (var reachable_to_ops, var pending_count, var loop_state) = _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, new List<object>(), xs);

                    foreach (var (y, grad_y) in zip(ys, grad_ys))
                        _SetGrad(grads, y, grad_y);

                    // Initialize queue with to_ops.
                    var queue = new Queue<Operation>();
                    // Add the ops in 'to_ops' into the queue.
                    var to_ops_set = new List<Operation>();
                    foreach (var op in to_ops)
                    {
                        // 'ready' handles the case where one output gradient relies on
                        // another output's gradient.
                        if (!pending_count.ContainsKey(op.name))
                            pending_count[op.name] = 0;
                        bool ready = pending_count[op.name] == 0;
                        if (ready && !to_ops_set.Contains(op) && reachable_to_ops.Contains(op))
                        {
                            to_ops_set.Add(op);
                            queue.Enqueue(op);
                        }
                    }

                    var stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count, xs);
                    while (queue.Count > 0)
                    {
                        // generate gradient subgraph for op.
                        var op = queue.Dequeue();

                        _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops);
                        //if (loop_state != null)
                        //loop_state.EnterGradWhileContext(op, before: true);
                        var out_grads = _AggregatedGrads(grads, op, gradient_uid, loop_state, aggregation_method);

                        Tensor[] in_grads = null;
                        var is_partitioned_call = _IsPartitionedCall(op);
                        var is_func_call = false;
                        var has_out_grads = true;
                        if (has_out_grads && !stop_ops.Contains(op))
                        {
                            // A grad_fn must be defined, either as a function or as None
                            // for ops that do not have gradients.
                            var grad_fn = ops.get_gradient_function(op);

                            if (is_func_call)
                            {

                            }
                            else
                            {
                                foreach (var (i, out_grad) in enumerate(out_grads))
                                {
                                    if (out_grad == null)
                                    {
                                        if (loop_state != null)
                                            ;
                                        else
                                            out_grads[i] = control_flow_ops.ZerosLikeOutsideLoop(op, i);
                                    }
                                }

                                with(ops.name_scope(op.name + "_grad"), scope1 =>
                                {
                                    string name1 = scope1;
                                    if (grad_fn != null)
                                    {
                                        in_grads = _MaybeCompile(grad_scope, op, out_grads, null, grad_fn);
                                        _VerifyGeneratedGradients(in_grads, op);
                                    }

                                    if (gate_gradients && in_grads.Count(x => x != null) > 1)
                                    {
                                        ops._colocate_with_for_gradient(null, gradient_uid, ignore_existing: true);
                                        in_grads = control_flow_ops.tuple(in_grads);
                                    }
                                });
                            }
                        }
                        else
                        {
                            in_grads = new Tensor[_NonEagerInputs(op, xs).Count()];
                        }

                        var inputs = _NonEagerInputs(op, xs).ToList();
                        foreach (var (t_in, in_grad) in zip(inputs, in_grads))
                        {
                            if (in_grad != null)
                            {
                                if (in_grad is Tensor &&
                                    in_grad.Tag == null && // maybe a IndexedSlice
                                    t_in.dtype != TF_DataType.TF_RESOURCE)
                                {
                                    in_grad.shape = t_in.shape;
                                }

                                _SetGrad(grads, t_in, in_grad);
                            }
                        }

                        // Update pending count for the inputs of op and enqueue ready ops.
                        _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state, xs);
                    }
                });

            return xs.Select(x => _GetGrad(grads, x)).ToArray();
        }

        /// <summary>
        /// Fill in default values for grad_ys.
        /// </summary>
        /// <param name="grad_ys">List of gradients, can contain None.</param>
        /// <param name="ys">List of tensors.</param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="gradient_uid"></param>
        private static Tensor[] _DefaultGradYs(Tensor[] grad_ys, Tensor[] ys, bool colocate_gradients_with_ops, string gradient_uid = "__unsupported__")
        {
            var new_grad_ys = new List<Tensor>();

            for (int i = 0; i < grad_ys.Length; i++)
            {
                var grad_y = grad_ys[i];
                var y = ys[i];

                _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops);

                if (grad_y == null)
                {
                    if (y.dtype.is_complex())
                        throw new TypeAccessException($"Gradients of complex tensors must set grad_ys (y.dtype = {y.dtype})");
                    var shape = array_ops.shape(y);
                    var constant = constant_op.constant(y.dtype == TF_DataType.TF_DOUBLE ? (object)1.0 : (object)1.0f, name: $"grad_ys_{i}");
                    var fill = gen_array_ops.fill(shape, constant);
                    new_grad_ys.Add(fill);
                }
            }

            return new_grad_ys.ToArray();
        }

        private static void _maybe_colocate_with(Operation op, string gradient_uid, bool colocate_gradients_with_ops)
        {

        }

        /// <summary>
        /// Initialize the pending count for ops between two lists of Operations.
        /// 'pending_count[op]' indicates the number of backprop inputs
        /// to this operation.
        /// </summary>
        /// <param name="to_ops"></param>
        /// <param name="from_ops"></param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="func_graphs"></param>
        /// <param name="xs"></param>
        private static (Operation[], Dictionary<string, int>, object) _PendingCount(List<Operation> to_ops, List<Operation> from_ops, bool colocate_gradients_with_ops, List<object> func_graphs, Tensor[] xs)
        {
            // Mark reachable ops from from_ops.
            var reached_ops = new List<Operation>();
            _MarkReachedOps(from_ops, reached_ops, func_graphs);
            // X in reached_ops iff X is reachable from from_ops by a path of zero or more
            // backpropagatable tensors.

            var reachable_to_ops = to_ops.Where(x => reached_ops.Contains(x)).Select(x => x).ToArray();

            var between_ops = new List<Operation>();
            var between_op_list = new List<Operation>();

            Queue<Operation> queue = new Queue<Operation>(to_ops);
            while (queue.Count > 0)
            {
                var op = queue.Dequeue();
                if (reached_ops.Contains(op))
                {
                    between_ops.Add(op);
                    between_op_list.Insert(between_op_list.Count, op);
                    // Clear the boolean so we won't add the inputs again.
                    reached_ops.Remove(op);
                    foreach (var inp in _NonEagerInputs(op, xs))
                        queue.Enqueue(inp.op);
                }
            }
            // X in between_ops iff X is on a path of zero or more backpropagatable tensors
            // between from_ops and to_ops

            // 'loop_state' is None if there are no while loops.
            var loop_state = control_flow_ops.MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops);

            var pending_count = new Dictionary<string, int>();
            foreach (var op in between_op_list)
            {
                foreach (Tensor x in _NonEagerInputs(op, xs))
                {
                    if (between_ops.Contains(x.op))
                    {
                        if (!pending_count.ContainsKey(x.op.name))
                            pending_count[x.op.name] = 0;

                        pending_count[x.op.name] += 1;
                    }
                }
            }

            return (reachable_to_ops.ToArray(), pending_count, loop_state);
        }

        /// <summary>
        /// Sets gradient "grad" in "grads" for tensor "t".
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="t"></param>
        /// <param name="grad"></param>
        private static void _SetGrad(Dictionary<string, List<List<Tensor>>> grads, Tensor t, Tensor grad)
        {
            var op = t.op;
            var op_grads = grads.ContainsKey(op.name) ? grads[op.name] : null;
            if (op_grads == null)
            {
                op_grads = op.outputs.Select(x => new List<Tensor>()).ToList();
                grads[op.name] = op_grads;
            }
            var t_grads = op_grads[t.value_index];
            t_grads.Add(grad);
        }

        private static IEnumerable<Tensor> _NonEagerInputs(Operation op, Tensor[] xs)
        {
            for (int i = 0; i < op.inputs.Length; i++)
                yield return op.inputs[i];
        }

        private static Tensor[] _AggregatedGrads(Dictionary<string, List<List<Tensor>>> grads, Operation op, string gradient_uid, object loop_state, int aggregation_method = 0)
        {
            var out_grads = _GetGrads(grads, op);
            var return_grads = new Tensor[out_grads.Count];

            foreach (var (i, out_grad) in enumerate(out_grads))
            {
                if (loop_state != null)
                {

                }

                // Aggregate multiple gradients, and convert [] to None.
                if (out_grad.Count > 0)
                {
                    string used = "";
                    if (out_grad.Count < 2)
                    {
                        used = "nop";
                        if (out_grad.Count == 0)
                        {
                            throw new ValueError("_AggregatedGrads out_grad.Length == 0");
                        }

                        return_grads[i] = out_grad[0];
                    }
                    else
                    {
                        used = "add_n";
                        return_grads[i] = _MultiDeviceAddN(out_grad.ToArray(), gradient_uid);
                    }
                }
                else
                {
                    return_grads[i] = null;
                }
            }

            return return_grads;
        }

        /// <summary>
        /// Adds tensors from potentially multiple devices.
        /// </summary>
        /// <param name="tensor_list"></param>
        /// <param name="gradient_uid"></param>
        /// <returns></returns>
        private static Tensor _MultiDeviceAddN(Tensor[] tensor_list, string gradient_uid)
        {
            // Basic function structure comes from control_flow_ops.group().
            // Sort tensors according to their devices.
            var tensors_on_device = new Dictionary<string, List<Tensor>>();
            
            foreach (var tensor in tensor_list)
            {
                if (!tensors_on_device.ContainsKey(tensor.Device))
                    tensors_on_device[tensor.Device] = new List<Tensor>();

                tensors_on_device[tensor.Device].Add(tensor);
            }
                
            // For each device, add the tensors on that device first.
            var summands = new List<Tensor>();
            foreach(var dev in tensors_on_device.Keys)
            {
                var tensors = tensors_on_device[dev];
                ops._colocate_with_for_gradient(tensors[0].op, gradient_uid, ignore_existing: true);
                summands.Add(math_ops.add_n(tensors.ToArray()));
            }

            return math_ops.add_n(summands.ToArray());
        }

        /// <summary>
        /// The set of ops that terminate the gradient computation.
        /// </summary>
        /// <param name="from_ops">list of Operations.</param>
        /// <param name="stop_gradient_ops">list of Operations never to backprop through.</param>
        /// <param name="pending_count">mapping from operation to number of backprop inputs.</param>
        /// <param name="xs">list of Tensors.</param>
        /// <returns>The set of operations.</returns>
        private static Operation[] _StopOps(List<Operation> from_ops, List<Operation> stop_gradient_ops, Dictionary<string, int> pending_count, Tensor[] xs)
        {
            var stop_ops = new List<Operation>();

            foreach (var op in from_ops)
            {
                bool is_stop_op = true;
                foreach (var inp in _NonEagerInputs(op, xs))
                {
                    if (!pending_count.ContainsKey(inp.op.name))
                        pending_count[inp.op.name] = 0;

                    if (pending_count[inp.op.name] > 0)
                    {
                        is_stop_op = false;
                        break;
                    }
                }
                if (is_stop_op)
                    stop_ops.Insert(0, op);
            }
            stop_ops.AddRange(stop_gradient_ops.Where(x => !stop_ops.Contains(x)));
            return stop_ops.ToArray();
        }

        private static Tensor _GetGrad(Dictionary<string, List<List<Tensor>>> grads, Tensor t)
        {
            var op = t.op;
            if (!grads.ContainsKey(op.name))
                return null;
            var op_grads = grads[op.name];
            var t_grad = op_grads[t.value_index];
            return t_grad[0];
        }

        private static List<List<Tensor>> _GetGrads(Dictionary<string, List<List<Tensor>>> grads, Operation op)
        {
            if (grads.ContainsKey(op.name))
                return grads[op.name];
            else
                return op.outputs.Select(x => new List<Tensor>()).ToList();
        }

        /// <summary>
        /// Mark all ops reached from "from_ops"
        /// </summary>
        /// <param name="from_ops"></param>
        /// <param name="reached_ops"></param>
        /// <param name="func_graphs"></param>
        private static void _MarkReachedOps(List<Operation> from_ops, List<Operation> reached_ops, List<object> func_graphs)
        {
            Queue<Operation> queue = new Queue<Operation>(from_ops);
            while (queue.Count > 0)
            {
                var op = queue.Dequeue();

                if (!reached_ops.Contains(op))
                {
                    reached_ops.Add(op);
                    foreach (var output in op.outputs)
                    {
                        if (_IsBackpropagatable(output))
                        {
                            var c = _Consumers(output, func_graphs).ToList();
                            c.ForEach(x => queue.Enqueue(x));
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Returns the consumers of t, crossing closure boundaries where necessary.
        /// </summary>
        /// <param name="t"></param>
        /// <param name="func_graphs"></param>
        private static Operation[] _Consumers(Tensor t, List<object> func_graphs)
        {
            return t.consumers();
        }

        private static bool _IsBackpropagatable(Tensor tensor)
        {
            if (_IsTrainable(tensor))
            {
                return true;
            }
            else
            {
                var dtype = tensor.dtype.as_base_dtype();
                return new TF_DataType[] { TF_DataType.TF_BFLOAT16, TF_DataType.TF_VARIANT }.Contains(dtype);
            }
        }

        private static bool _IsTrainable(Tensor tensor)
        {
            var dtype = tensor.dtype.as_base_dtype();
            return new TF_DataType[] {TF_DataType.TF_HALF, TF_DataType.TF_FLOAT, TF_DataType.TF_DOUBLE,
                TF_DataType.TF_COMPLEX64, TF_DataType.TF_COMPLEX128, TF_DataType.TF_RESOURCE}.Contains(dtype);
        }

        private static bool _IsPartitionedCall(Operation op)
        {
            return op.OpType == "PartitionedCall" || op.OpType == "StatefulPartitionedCall";
        }

        /// <summary>
        /// Update pending count for the inputs of op and enqueue ready ops.
        /// </summary>
        /// <param name="grads"></param>
        /// <param name="op"></param>
        /// <param name="queue"></param>
        /// <param name="pending_count"></param>
        /// <param name="loop_state"></param>
        /// <param name="xs"></param>
        private static void _UpdatePendingAndEnqueueReady(Dictionary<string, List<List<Tensor>>> grads,
            Operation op,
            Queue<Operation> queue,
            Dictionary<string, int> pending_count,
            object loop_state,
            Tensor[] xs)
        {
            foreach (var x in _NonEagerInputs(op, xs))
            {
                if (!pending_count.ContainsKey(x.op.name))
                    pending_count[x.op.name] = 0;

                pending_count[x.op.name] -= 1;

                var ready = pending_count[x.op.name] == 0;

                if (loop_state != null && !ready)
                {

                }

                if (ready)
                {
                    if (control_flow_util.IsLoopExit(x.op))
                    {

                    }
                    else
                    {
                        queue.Enqueue(x.op);
                    }
                }
            }
        }

        private static Tensor[] _MaybeCompile(string scope, Operation op, Tensor[] out_grads, Action func, Func<Operation, Tensor[], Tensor[]> grad_fn)
        {
            scope = scope.EndsWith("/") ? scope.Substring(0, scope.Length - 1) : scope;
            return grad_fn(op, out_grads);
        }

        private static void _VerifyGeneratedGradients(Tensor[] grads, Operation op)
        {
            if (grads.Count() != op.inputs._inputs.Count())
                throw new ValueError($"Num gradients {grads.Length} generated for op {op.node_def} do not match num " +
                    $"inputs {op.inputs._inputs.Count()}");
        }
    }
}
