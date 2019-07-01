using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework;
using Tensorflow.Train;
using static Tensorflow.Python;

namespace Tensorflow
{
    /// <summary>
    /// Base class for optimizers.
    /// This class defines the API to add Ops to train a model.  You never use this
    /// class directly, but instead instantiate one of its subclasses such as
    /// `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.
    /// </summary>
    public abstract class Optimizer : Trackable
    {
        // Values for gate_gradients.
        public static int GATE_NONE = 0;
        public static int GATE_OP = 1;
        public static int GATE_GRAPH = 2;

        string _name;
        public string Name => _name;
        protected float _lr;
        public float LearningRate => _lr;
        protected Tensor _lr_t;
        public Tensor LearningRateTensor => _lr_t;
        public bool _use_locking;
        public Dictionary<string, Dictionary<string, RefVariable>> _slots;
        public Dictionary<string, RefVariable> _non_slot_dict;
        public Dictionary<string, object> _deferred_slot_restorations;
        SlotCreator slot_creator = new SlotCreator();

        public Optimizer(float learning_rate, bool use_locking, string name = null)
        {
            if (String.IsNullOrEmpty(name))
                throw new NotImplementedException("Must specify the optimizer name");

            _name = name;
            _use_locking = use_locking;
            _lr = learning_rate;
            // Dictionary of slots.
            _slots = new Dictionary<string, Dictionary<string, RefVariable>>();
            _non_slot_dict = new Dictionary<string, RefVariable>();
            _deferred_slot_restorations = new Dictionary<string, object>();
        }

        /// <summary>
        /// Add operations to minimize `loss` by updating `var_list`
        ///  
        ///  This method simply combines calls `compute_gradients()` and
        ///  `apply_gradients()`. If you want to process the gradient before applying
        ///  them call `compute_gradients()` and `apply_gradients()` explicitly instead
        ///  of using this function.
        /// </summary>
        /// <param name="loss">A `Tensor` containing the value to minimize.</param>
        /// <param name="global_step">Optional `Variable` to increment by one after the
        /// variables have been updated.</param>
        /// <param name="var_list">Optional list or tuple of `Variable` objects to update to
        /// minimize `loss`.  Defaults to the list of variables collected in
        /// the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.</param>
        /// <param name="gate_gradients">
        /// How to gate the computation of gradients.  Can be
        /// `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
        /// </param>
        /// <param name="aggregation_method">
        /// Specifies the method used to combine gradient terms.
        /// Valid values are defined in the class `AggregationMethod`.
        /// </param>
        /// <param name="colocate_gradients_with_ops"></param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <param name="grad_loss">Optional. A `Tensor` holding the gradient computed for `loss`.</param>
        /// <returns>
        /// An Operation that updates the variables in `var_list`.  If `global_step`
        /// was not `None`, that operation also increments `global_step`.
        /// </returns>
        public Operation minimize(Tensor loss, 
            RefVariable global_step = null,
            List<RefVariable> var_list=null,
            GateGradientType gate_gradients = GateGradientType.GATE_OP,
            int? aggregation_method=null,
            bool colocate_gradients_with_ops = false, string name=null, Tensor grad_loss=null)
        {
            // TODO: strongly type aggregation_method
            var grads_and_vars = compute_gradients(loss, var_list:var_list,
                gate_gradients: gate_gradients, 
                aggregation_method:aggregation_method,
                colocate_gradients_with_ops: colocate_gradients_with_ops,
                grad_loss: grad_loss);

            var vars_with_grad = grads_and_vars.Where(x => x.Item1 != null).Select(x => x.Item2).ToArray();
            if (vars_with_grad.Length == 0)
                throw new ValueError($"No gradients provided for any variable, check your graph for ops" +
                    $" that do not support gradients, between variables {string.Join(",", vars_with_grad.Select(x => x.name))} and loss {loss}.");

            return apply_gradients(grads_and_vars, global_step:global_step, name:name);
        }

        /// <summary>
        /// Apply gradients to variables.
        /// 
        /// This is the second part of `minimize()`. It returns an `Operation` that
        /// applies gradients.
        /// </summary>
        /// <param name="grads_and_vars">List of (gradient, variable) pairs as returned by
        /// `compute_gradients()`.</param>
        /// <param name="global_step">Optional `Variable` to increment by one after the
        /// variables have been updated.</param>
        /// <param name="name">Optional name for the returned operation.  Default to the
        /// name passed to the `Optimizer` constructor.</param>
        /// <returns>
        /// An `Operation` that applies the specified gradients. If `global_step`
        /// was not None, that operation also increments `global_step`.</returns>
        public Operation apply_gradients(Tuple<Tensor, RefVariable>[] grads_and_vars, RefVariable global_step = null, string name = null)
        {
            // No DistributionStrategy case.
            var converted_grads_and_vars = new List<(Tensor, RefVariable, _OptimizableVariable)>();
            foreach (var (g, v) in grads_and_vars)
            {
                if(g != null)
                {
                    // Convert the grad to Tensor or IndexedSlices if necessary.
                    var gR = ops.convert_to_tensor_or_indexed_slices(g);
                    var p = _get_processor(v);
                    converted_grads_and_vars.Add((gR, v, p));
                }
            }

            var var_list = converted_grads_and_vars.Where(x => x.Item1 != null).Select(x => x.Item2).ToArray();
            if (var_list.Length == 0)
                throw new ValueError($"No gradients provided for any variable");

            ops.init_scope();
            _create_slots(var_list);

            var update_ops = new List<Operation>();
            return with(ops.name_scope(name, Name), scope =>
            {
                name = scope;
                _prepare();

                foreach(var (grad, var, processor) in converted_grads_and_vars)
                {
                    if (grad == null)
                        continue;

                    var scope_name = var.op.name;
                    with(ops.name_scope("update_" + scope_name), scope2 =>
                    {
                        var op = processor.update_op(this, grad);
                        update_ops.Add(op);
                    });
                }

                Operation apply_updates = null;
                if (global_step == null)
                {
                    apply_updates = _finish(update_ops.ToArray(), name);
                }
                else
                {
                    with(ops.control_dependencies(new object[] {_finish(update_ops.ToArray(), "update")}), dep =>
                    {
                        ops.colocate_with(global_step);
                        // TODO: port this if branch once ResourceVariable has been ported!
                        //if (global_step is ResourceVariable)
                        //{
                        //        # TODO(apassos): the implicit read in assign_add is slow; consider
                        //        # making it less so.
                        //        apply_updates = resource_variable_ops.assign_add_variable_op(
                        //            global_step.handle,
                        //            ops.convert_to_tensor(1, dtype = global_step.dtype),
                        //            name = name)
                        //}
                        //else
                        {
                            apply_updates = state_ops.assign_add(global_step, tf.constant(1), name: name);
                        }
                    });
                }

                if (!tf.context.executing_eagerly())
                {
                    var train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP) as List<object>;
                    if (!train_op.Contains(apply_updates))
                        train_op.Add(apply_updates);
                }

                return apply_updates;
            });
        }

        /// <summary>
        /// Create the beta1 and beta2 accumulators on the same device as the first
        /// variable. Sort the var_list to make sure this device is consistent across
        /// workers (these need to go on the same PS, otherwise some updates are
        /// silently ignored).
        /// </summary>
        /// <param name="var_list"></param>
        protected virtual void _create_slots(RefVariable[] var_list)
        {
            
        }

        /// <summary>
        /// Add an extra variable, not associated with a slot.
        /// </summary>
        /// <param name="initial_value"></param>
        /// <param name="name"></param>
        /// <param name="colocate_with"></param>
        protected RefVariable _create_non_slot_variable(float initial_value, string name, RefVariable colocate_with)
        {
            // Recommendation: Use OptimizerV2 if your optimizer uses non-slot variables.
            var graph = colocate_with.graph;
            var key = $"{name}.{graph.graph_key}";
            var v = _non_slot_dict.ContainsKey(key) ? _non_slot_dict[key] : null;
            if(v == null)
            {
                _maybe_initialize_trackable();
                v = variable_scope.default_variable_creator(
                    initial_value, name: name, trainable: false,
                    use_resource: resource_variable_ops.is_resource_variable(
                        colocate_with));

                // Restore this variable by name if necessary, but don't add a
                // Trackable dependency. Optimizers return the current graph's
                // non-slot variables from _checkpoint_dependencies explicitly rather
                // than unconditionally adding dependencies (since there may be multiple
                // non-slot variables with the same name in different graphs, trying to
                // save all of them would result in errors).
                _handle_deferred_dependencies(name, v);
                _non_slot_dict[key] = v;
            }

            return v;
        }

        public virtual Operation _finish(Operation[] update_ops, string name_scope)
        {
            return control_flow_ops.group(update_ops, name_scope);
        }

        public virtual Operation _apply_dense(Tensor grad, RefVariable var)
        {
            var alpha = math_ops.cast(LearningRateTensor, var.dtype.as_base_dtype());
            return gen_training_ops.apply_gradient_descent(var, alpha, grad, use_locking: _use_locking).op;
        }

        /// <summary>
        /// Add ops to apply sparse gradients to `var`, with repeated sparse indices.
        /// </summary>
        /// <param name="grad"></param>
        /// <param name="var"></param>
        /// <returns></returns>
        public virtual Operation _apply_sparse_duplicate_indices(IndexedSlices grad, RefVariable var)
        {
            var (summed_values, unique_indices) = _deduplicate_indexed_slices(values: grad.values, indices: grad.indices);
            var gradient_no_duplicate_indices = new IndexedSlices(
                indices: unique_indices,
                values: summed_values,
                dense_shape: grad.dense_shape);
            return _apply_sparse(gradient_no_duplicate_indices, var);
        }

        public virtual Operation _apply_sparse(IndexedSlices grad, RefVariable var)
        {
            throw new NotImplementedException("_apply_sparse");
        }

        public virtual (Tensor, Tensor) _deduplicate_indexed_slices(Tensor values, Tensor indices)
        {
            var (unique_indices, new_index_positions) = array_ops.unique(indices);
            var shape = array_ops.shape(unique_indices).slice(0);
            var summed_values = math_ops.unsorted_segment_sum(values, new_index_positions, shape);
            return (summed_values, unique_indices);
        }

        public virtual void _prepare()
        {

        }

        /// <summary>
        /// Return a slot named `name` created for `var` by the Optimizer.
        /// </summary>
        /// <param name="var"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        protected RefVariable get_slot(RefVariable var, string name)
        {
            var named_slots = _slots.ContainsKey(name) ? _slots[name] : null;
            if (named_slots == null)
                return null;

            return named_slots.ContainsKey(_var_key(var)) ? named_slots[_var_key(var)] : null;
        }

        private string _var_key(RefVariable var)
        {
            return $"{var.op.graph.graph_key}.{var.op.name}";
        }

        protected RefVariable _get_non_slot_variable(string name, Graph graph = null)
        {
            var key = $"{name}.{graph.graph_key}";
            var non_slot = _non_slot_dict.ContainsKey(key) ? _non_slot_dict[key] : null;

            return non_slot;
        }

        private _OptimizableVariable _get_processor(RefVariable v)
        {
            if(v is RefVariable)
            {
                return new _RefVariableProcessor(v);
            }
            else
            {
                throw new NotImplementedException("_get_processor");
            }
        }

        /// <summary>
        /// Compute gradients of `loss` for the variables in `var_list`.
        /// </summary>
        /// <param name="loss"></param>
        /// <param name="gate_gradients"></param>
        /// <returns>
        /// A list of (gradient, variable) pairs. Variable is always present, but
        /// gradient can be `None`.
        /// </returns>
        public Tuple<Tensor, RefVariable>[] compute_gradients(Tensor loss,
            List<RefVariable> var_list = null,
            int? aggregation_method = null,
            GateGradientType gate_gradients = GateGradientType.GATE_OP,
            bool colocate_gradients_with_ops = false,
            Tensor grad_loss = null)
        {
            // Scale loss if using a "mean" loss reduction and multiple replicas.
            loss = _scale_loss(loss);
            int num_towers = 1;


            var tmp = variables.trainable_variables();
            var vars = ops.get_collection<RefVariable>(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES);
            switch (tmp)
            {
                case List<RefVariable> values:
                    var_list = values.Concat(vars).ToList();
                    break;
                case List<VariableV1> values:
                    var_list = values.Select(x => x as RefVariable).Concat(vars).ToList();
                    break;
            }

            var_list = var_list.Concat(ops.get_collection<RefVariable>(ops.GraphKeys._STREAMING_MODEL_PORTS)).ToList();
            var processors = var_list.Select(v => optimizer._get_processor(v)).ToList();
            var var_refs = processors.Select(x => x.target()).ToArray();

            var grads = gradients_impl.gradients(new Tensor[] { loss }, var_refs, grad_ys: grad_loss == null ? null : new Tensor[] { grad_loss },
                gate_gradients: gate_gradients == GateGradientType.GATE_OP,
                aggregation_method: aggregation_method,
                colocate_gradients_with_ops: colocate_gradients_with_ops);

            if ((int)gate_gradients == Optimizer.GATE_GRAPH)
                grads = control_flow_ops.tuple(grads);

            var grads_and_vars = Python.zip(grads, var_list)
                .Select(x => new Tuple<Tensor, RefVariable>(x.Item1, x.Item2))
                .ToArray();

            return grads_and_vars;
        }

        private Tensor _scale_loss(Tensor loss_value)
        {
            ops.get_default_graph()._is_loss_scaled_by_optimizer = false;
            // TODO
            // if distribute_lib.get_loss_reduction() == ds_reduce_util.ReduceOp.MEAN:
            return loss_value;
        }

        protected T _call_if_callable<T>(T param)
        {
            return param;
        }

        /// <summary>
        /// Find or create a slot initialized with 0.0.
        /// </summary>
        /// <param name="var"></param>
        /// <param name="slot_name"></param>
        /// <param name="op_name"></param>
        /// <returns></returns>
        protected RefVariable _zeros_slot(RefVariable var, string slot_name, string op_name)
        {
            var named_slots = _slot_dict(slot_name);
            if (!named_slots.ContainsKey(_var_key(var)))
            {
                var new_slot_variable = slot_creator.create_zeros_slot(var, op_name);
                _restore_slot_variable(slot_name: slot_name, variable: var, slot_variable: new_slot_variable);
                named_slots[_var_key(var)] = new_slot_variable;
            }
            return named_slots[_var_key(var)];
        }

        /// <summary>
        /// Restore a newly created slot variable's value.
        /// </summary>
        protected void _restore_slot_variable(string slot_name, RefVariable variable, RefVariable slot_variable)
        {
            var variable_key = _var_key(variable);
            // TODO
        }

        protected Dictionary<string, RefVariable> _slot_dict(string slot_name)
        {
            var named_slots = _slots.ContainsKey(slot_name) ? _slots[slot_name] : null;
            if(named_slots == null)
            {
                named_slots = new Dictionary<string, RefVariable>();
                _slots[slot_name] = named_slots;
            }

            return named_slots;
        }
    }
}
