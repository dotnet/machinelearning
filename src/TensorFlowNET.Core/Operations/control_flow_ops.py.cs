using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations;
using Tensorflow.Operations.ControlFlows;
using util = Tensorflow.control_flow_util;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class control_flow_ops
    {
        public static Operation Assert(Tensor condition, object[] data, int? summarize = null, string name = null)
        {
            return with(ops.name_scope(name, "Assert", new { condition, data }), scope =>
            {
                name = scope;
                var xs = ops.convert_n_to_tensor(data);
                condition = ops.convert_to_tensor(condition, name: "Condition");
                Func<Operation[]> true_assert = () =>
                {
                    var assert = gen_logging_ops._assert(condition, data, summarize, name: "Assert");
                    return new Operation[] { assert };
                };

                Func<Operation[]> false_assert = () =>
                {
                    var op = gen_control_flow_ops.no_op();
                    return new Operation[] { op };
                };

                var guarded_assert = cond(condition, false_assert, true_assert, name: "AssertGuard");

                return guarded_assert[0].op;
            });
        }

        public static Operation group<T>(T[] inputs, string name = null) where T : ITensorOrOperation
        {
            return with(ops.name_scope(name, "group_deps", inputs), scope =>
            {
                name = scope;

                // Sorts *inputs according to their devices.
                var ops_on_device = new Dictionary<string, List<T>>();
                foreach (var inp in inputs)
                {
                    if (ops_on_device.ContainsKey(inp.Device))
                        ops_on_device[inp.Device].Add(inp);
                    else
                        ops_on_device[inp.Device] = new List<T> { inp };
                }

                // 1-level tree. The root node is the returned NoOp node.
                if (ops_on_device.Count == 1)
                {
                    var dev = ops_on_device.Keys.First();
                    var deps = ops_on_device.Values.First();
                    return _GroupControlDeps(dev, deps.Select(x => x.op).ToArray(), name);
                }

                // 2-level tree. The root node is the returned NoOp node.
                // deps contains 1 NoOp node for each device.
                throw new NotImplementedException("control_flow_ops.group");
            });
        }

        /// <summary>
        /// Does nothing. Only useful as a placeholder for control edges.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Operation no_op(string name = null)
            => gen_control_flow_ops.no_op(name: name);

        private static Operation _GroupControlDeps(string dev, Operation[] deps, string name = null)
        {
            return with(ops.control_dependencies(deps), ctl =>
            {
                if (dev == null)
                {
                    return gen_control_flow_ops.no_op(name);
                }
                else
                {
                    return gen_control_flow_ops.no_op(name);
                }
            });
        }

        /// <summary>
        /// Create the state for all the while loops involved in one gradients().
        /// </summary>
        /// <param name="between_op_list"></param>
        /// <param name="between_ops"></param>
        /// <param name="colocate_gradients_with_ops"></param>
        public static ControlFlowState MaybeCreateControlFlowState(List<Operation> between_op_list, List<Operation> between_ops, bool colocate_gradients_with_ops)
        {
            ControlFlowState loop_state = null;

            foreach (var op in between_op_list)
            {
                if (IsLoopExit(op))
                {
                    if(loop_state == null)
                    {
                        loop_state = new ControlFlowState();
                    }
                }
            }

            return loop_state;
        }

        public static bool IsLoopExit(Operation op)
        {
            return op.OpType == "Exit" || op.OpType == "RefExit";
        }

        public static Tensor[] tuple(Tensor[] tensors, string name = null, Operation[] control_inputs = null)
        {
            return with(ops.name_scope(name, "tuple", tensors), scope =>
            {
                name = scope;
                var gating_ops = tensors.Where(x => x != null).Select(x => x.op).ToList();

                if(control_inputs != null)
                {
                    foreach (var c in control_inputs)
                        gating_ops.Add(c);
                }

                // Note that in order to ensure ordering in the pbtxt, we must take care to
                // ensure the order here.
                gating_ops = gating_ops.OrderBy(x => x._id).ToList();
                var gate = group(gating_ops.ToArray());

                var tpl = new List<Tensor>();
                foreach(var t in tensors)
                {
                    if (t != null)
                        tpl.Add(with_dependencies(new Operation[] { gate }, t));
                    else
                        tpl.Add(null);
                }

                return tpl.ToArray();
            });
        }

        /// <summary>
        /// Produces the content of `output_tensor` only after `dependencies`.
        /// 
        /// In some cases, a user may want the output of an operation to be
        /// consumed externally only after some other dependencies have run
        /// first.This function ensures returns `output_tensor`, but only after all
        /// operations in `dependencies` have run.Note that this means that there is
        /// no guarantee that `output_tensor` will be evaluated after any `dependencies`
        /// have run.
        /// 
        /// See also `tf.tuple` and `tf.group`.
        /// </summary>
        /// <param name="dependencies">Iterable of operations to run before this op finishes.</param>
        /// <param name="output_tensor">A `Tensor` or `IndexedSlices` that will be returned.</param>
        /// <param name="name">(Optional) A name for this operation.</param>
        /// <returns>Same as `output_tensor`.</returns>
        public static Tensor with_dependencies(Operation[] dependencies, Tensor output_tensor, string name = null)
        {
            //TODO: missing original code
            //if context.executing_eagerly():
            //    return output_tensor
            var values = new List<object>();
            values.AddRange(dependencies);
            values.Add(output_tensor);

            return with(ops.name_scope(name, "control_dependency", values), scope =>
            {
                name = scope;
                // TODO: missing original code
                //with ops.colocate_with(output_tensor):
                {
                    return with(ops.control_dependencies(dependencies), ctl =>
                    {
                        output_tensor = ops.convert_to_tensor_or_composite(output_tensor);
                        return _Identity(output_tensor, name: name);
                    });
                }
            });
        }

        public static Tensor _Identity(Tensor data, string name = null)
        {
            data = ops.internal_convert_to_tensor_or_composite(data, as_ref: true);
            if ((int)data.dtype > 100)
                throw new NotImplementedException("_Identity");
            else
                return gen_array_ops.identity(data, name: name);
        }

        ///  <summary>
        ///  Forwards `data` to an output determined by `pred`.
        ///  If `pred` is false, the `data` input is forwarded to the first output.
        ///  Otherwise, the data goes to the second output.
        ///  
        ///  This op handles `Tensor`s and `IndexedSlices`.
        ///  </summary>
        ///  <param name="data">The tensor to be forwarded to the appropriate output.</param>
        ///  <param name="pred">A scalar that specifies which output port will receive data.</param>
        /// <param name="name"> A name for this operation (optional).</param>
        /// <returns>
        ///  `(output_false, output_true)`: If `pred` is true, data will be forwarded to
        /// `output_true`, otherwise it goes to `output_false`.
        /// </returns>
        public static Tensor[] _SwitchRefOrTensor(Tensor data, Tensor pred, string name = "Switch")
        {
            data = ops.convert_to_tensor_or_indexed_slices(data, name: "data");
            // NOTE(vrv): ops.colocate_with(data, ignore_existing=True) below
            // addresses the following scenario.
            //
            // Assume you execute Optimizer.apply_gradients() in a branch of a cond().
            //
            // 1. The update op is created inside a `with ops.colocate(var):` block
            //
            // 2. Some tensor `data` is captured and a switch is created in a
            //    `with ops.colocate_with(data):` block.
            //
            // with ops.colocate_with(var):
            //  with ops.colocate_with(data):
            //    op = ...
            //
            // var and data may be pinned to different devices, so we want to ops
            // created within ops.colocate_with(data) to ignore the existing stack.
            ops.colocate_with(data, ignore_existing: true);
            {
                if (data is Tensor)
                {
                    // TODO: ref_switch
                    //if (data.dtype._is_ref_dtype)
                    //    return control_flow_ops.ref_switch(data, pred, name = name);
                }
                return @switch(data, pred, name: name);
            }            
        }

        /// <summary>
        /// Return `true_fn()` if the predicate `pred` is true else `false_fn()`.
        /// 
        /// `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
        /// `false_fn` must have the same non-zero number and type of outputs.
        /// 
        /// **WARNING**: Any Tensors or Operations created outside of `true_fn` and
        /// `false_fn` will be executed regardless of which branch is selected at runtime.
        /// 
        /// Although this behavior is consistent with the dataflow model of TensorFlow,
        /// it has frequently surprised users who expected a lazier semantics.
        /// Consider the following simple program:
        /// 
        /// z = tf.multiply(a, b)
        /// result = tf.cond(x &lt; y, ()=> tf.add(x, z), ()=> tf.square(y))
        /// 
        /// If `x&lt;y`, the `tf.add` operation will be executed and `tf.square`
        /// operation will not be executed.Since `z` is needed for at least one
        /// branch of the `cond`, the `tf.multiply` operation is always executed,
        /// unconditionally.
        /// 
        /// Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
        /// call to `cond`, and not at all during `Session.run()`). `cond`
        /// stitches together the graph fragments created during the `true_fn` and
        /// `false_fn` calls with some additional graph nodes to ensure that the right
        /// branch gets executed depending on the value of `pred`.
        /// 
        /// `tf.cond` supports nested structures as implemented in
        /// `tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
        /// same(possibly nested) value structure of lists, tuples, and/or named tuples.
        /// Singleton lists and tuples form the only exceptions to this: when returned by
        /// `true_fn` and/or `false_fn`, they are implicitly unpacked to single values.
        /// This behavior is disabled by passing `strict= True`.
        /// </summary>
        /// <param name="pred"> A scalar determining whether to return the result of `true_fn` or
        /// `false_fn`.</param>
        /// <param name="true_fn">The callable to be performed if pred is true.</param>
        /// <param name="false_fn">The callable to be performed if pred is false.</param>
        /// <param name="strict"> A boolean that enables/disables 'strict' mode; see above.</param>
        /// <param name="name">Optional name prefix for the returned tensors.</param>
        /// <returns>Tensors returned by the call to either `true_fn` or `false_fn`. If the
        /// callables return a singleton list, the element is extracted from the list.</returns>
        public static Tensor cond(Tensor pred,
            Func<ITensorOrOperation> true_fn = null,
            Func<ITensorOrOperation> false_fn = null,
            bool strict = false,
            string name = null)
        {
            return with(ops.name_scope(name, "cond", new { pred }), delegate
            {
                // TODO: here a chunk of original code is missing
                /*
                  with ops.name_scope(name, "cond", [pred]):
                    if context.executing_eagerly():
                      if pred:
                        return _UnpackIfSingleton(true_fn())
                      return _UnpackIfSingleton(false_fn())
                */

                // Add the Switch to the graph.
                var switch_result= @switch(pred, pred);
                var p_2=switch_result[0];
                var p_1 = switch_result[1];
                var pivot_1 = array_ops.identity(p_1, name: "switch_t");
                var pivot_2 = array_ops.identity(p_2, name: "switch_f");
                pred = array_ops.identity(pred, name: "pred_id");

                // Disable the fetching of tensors that are only on one branch of cond.
                foreach (var tensor in new Tensor[] { p_1, p_2, pivot_1, pivot_2, pred })
                    tensor.op.graph.prevent_fetching(tensor.op);

                // Build the graph for the true branch in a new context.
                var context_t = new CondContext(pred: pred, pivot: pivot_1, branch: 1);
                ITensorOrOperation orig_res_t;
                Tensor res_t;
                try
                {
                    context_t.Enter();
                    (orig_res_t, res_t) = context_t.BuildCondBranch(true_fn);
                }
                finally
                {
                    context_t.Exit();
                }
                // Build the graph for the false branch in a new context.
                var context_f = new CondContext(pred: pred, pivot: pivot_2, branch: 0);
                ITensorOrOperation orig_res_f;
                Tensor res_f;
                try
                {
                    context_f.Enter();
                    (orig_res_f, res_f) = context_f.BuildCondBranch(false_fn);
                }
                finally
                {
                    context_f.Exit();
                }

                //TODO: missing original code
                //if not strict:
                //  orig_res_t = _UnpackIfSingleton(orig_res_t)
                //  orig_res_f = _UnpackIfSingleton(orig_res_f)
                /*
                # Check that the return values of the two branches have the same structure.
                try:
                    nest.assert_same_structure(orig_res_t, orig_res_f)
                except TypeError as e:
                    raise TypeError(
                        "Incompatible return types of true_fn and false_fn: {}".format(e))
                except ValueError as e:
                    raise ValueError(
                        "Incompatible return values of true_fn and false_fn: {}".format(e))*/

                var res_t_flat = new Tensor[] { res_t };
                var res_f_flat = new Tensor[] { res_f };

                foreach(var (val_x, val_y) in zip(res_t_flat, res_f_flat))
                {

                }
                
                var merges = zip(res_f_flat, res_t_flat)
                    .Select(pair => merge(new Tensor[] { pair.Item1, pair.Item2 }))
                    .ToArray();

                merges = _convert_flows_to_tensorarrays(new Tensor[] { (Tensor)orig_res_t }, merges);

                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_t);
                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_f);

                return merges[0];
            });
        }

        public static Tensor[] cond<T>(Tensor pred,
            Func<T[]> true_fn = null,
            Func<T[]> false_fn = null,
            bool strict = false,
            string name = null)
        {
            return with(ops.name_scope(name, "cond", new { pred }), delegate
            {
                // Add the Switch to the graph.
                var switch_result = @switch(pred, pred);
                var p_2 = switch_result[0];
                var p_1 = switch_result[1];
                var pivot_1 = array_ops.identity(p_1, name: "switch_t");
                var pivot_2 = array_ops.identity(p_2, name: "switch_f");
                pred = array_ops.identity(pred, name: "pred_id");

                // Disable the fetching of tensors that are only on one branch of cond.
                foreach (var tensor in new Tensor[] { p_1, p_2, pivot_1, pivot_2, pred })
                    tensor.op.graph.prevent_fetching(tensor.op);

                // Build the graph for the true branch in a new context.
                var context_t = new CondContext(pred: pred, pivot: pivot_1, branch: 1);
                context_t.Enter();
                var (orig_res_t, res_t) = context_t.BuildCondBranch(true_fn);
                context_t.Exit();

                // Build the graph for the false branch in a new context.
                var context_f = new CondContext(pred: pred, pivot: pivot_2, branch: 0);
                context_f.Enter();
                var (orig_res_f, res_f) = context_f.BuildCondBranch(false_fn);
                context_f.Exit();

                var res_t_flat = res_t;
                var res_f_flat = res_f;

                var merges = zip(res_f_flat, res_t_flat)
                    .Select(pair => merge(new Tensor[] { pair.Item1, pair.Item2 }))
                    .ToArray();

                merges = _convert_flows_to_tensorarrays(orig_res_t, merges);

                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_t);
                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_f);

                return merges;
            });
        }

        public static Tensor[] _convert_flows_to_tensorarrays<T>(T[] tensors_or_tensorarrays, Tensor[] tensors_or_flows)
        {
            // zip(tensors_or_tensorarrays, tensors_or_flows).Select((ta, t_or_flow) => ta).ToArray();
            return tensors_or_flows;
        }

        /// <summary>
        /// Returns the value of an available element of `inputs`.
        /// 
        /// This op tests each of the tensors in `inputs` in turn to determine if any of
        /// them is available.If it finds an available tensor, it returns it and its
        /// index in `inputs`.
        /// 
        /// It is an error if more than one tensor in `inputs` is available.If no tensor
        /// in `inputs` is available, the returned tensor and index are not set.
        /// 
        /// This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
        /// `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
        /// before merging.
        /// </summary>
        /// <param name="inputs">inputs: The input tensors, at most one of which is available.</param>
        /// <param name="name">A name for this operation (optional).</param>
        /// <returns></returns>
        public static Tensor merge(Tensor[] inputs, string name = null)
        {
            if (inputs.Any(x => x == null))
                throw new ValueError($"At least one of the merge inputs is null: {inputs}");
            return with(ops.name_scope(name, "Merge", inputs), scope =>
            {
                name = scope;
                inputs = inputs.Select(inp =>
                            ops.internal_convert_to_tensor_or_indexed_slices(inp, as_ref: true))
                        .ToArray();
                return gen_control_flow_ops.merge(inputs, name).Item1;
            });
        }

        /// <summary>
        /// Forwards `data` to an output determined by `pred`.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="pred"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        public static Tensor[] @switch(Tensor data, 
            Tensor pred, 
            TF_DataType dtype = TF_DataType.DtInvalid, 
            string name = null)
        {
            return with(ops.name_scope(name, "Switch", new { data, pred }), scope =>
            {
                name = scope;
                data = ops.internal_convert_to_tensor_or_indexed_slices(data,
                    dtype: dtype,
                    name: "data",
                    as_ref: true);

                pred = ops.convert_to_tensor(pred, name: "pred");

                return gen_control_flow_ops.@switch(data, pred, name: name);
            });
        }

        public static Tensor ZerosLikeOutsideLoop(Operation op, int index)
        {
            var val = op.outputs[index];
            if (!util.IsSwitch(op))
            {
                if (val.dtype == TF_DataType.TF_RESOURCE)
                    throw new NotImplementedException("ZerosLikeOutsideLoop");
                return array_ops.zeros_like(val, optimize: false);
            }

            throw new NotImplementedException("ZerosLikeOutsideLoop");
        }

        // TODO
        public static void while_loop(Func<Tensor, Tensor> func, Func<Tensor, Tensor> func1, Tensor[] tensors, int? i)
        {
            throw new NotImplementedException();
        }

    }
}
