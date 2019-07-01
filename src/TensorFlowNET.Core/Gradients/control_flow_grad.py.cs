using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Gradients for operators defined in control_flow_ops.py.cs
    /// </summary>
    public class control_flow_grad
    {
        /// <summary>
        /// Gradients for a Switch op is calculated using a Merge op.
        ///
        /// If the switch is a loop switch, it will be visited twice. We create
        /// the merge on the first visit, and update the other input of the merge
        /// on the second visit. A next_iteration is also added on second visit.
        /// </summary>
        /// <returns></returns>
        public Tensor[] _SwitchGrad(Tensor op, Tensor[] grads)
        {
            throw new NotImplementedException("_SwitchGrad");
            //graph = ops.get_default_graph()
            //# pylint: disable=protected-access
            //op_ctxt = op._get_control_flow_context()
            //grad_ctxt = graph._get_control_flow_context()
            //# pylint: enable=protected-access
            //if isinstance(op_ctxt, WhileContext):
            //  merge_grad = grad_ctxt.grad_state.switch_map.get(op)
            //  if merge_grad is not None:
            //    # This is the second time this Switch is visited. It comes from
            //    # the non-exit branch of the Switch, so update the second input
            //    # to the Merge.
            //    # TODO(yuanbyu): Perform shape inference with this new input.
            //    if grad[1] is not None:
            //      # pylint: disable=protected-access
            //      control_flow_ops._AddNextAndBackEdge(merge_grad, grad[1],
            //                                           enforce_shape_invariant=False)
            //      # pylint: enable=protected-access
            //    return None, None
            //  elif grad[0] is not None:
            //    # This is the first time this Switch is visited. It comes from
            //    # the Exit branch, which is grad[0]. grad[1] is empty at this point.
            //    # Use grad[0] for both inputs to merge for now, but update the second
            //    # input of merge when we see this Switch the second time.
            //    merge_grad = merge([grad[0], grad[0]], name="b_switch")[0]
            //    grad_ctxt.grad_state.switch_map[op] = merge_grad
            //    return merge_grad, None
            //  else:
            //    # This is the first time this Switch is visited. It comes from the
            //    # Identity branch. Such a Switch has `None` gradient for the Exit branch,
            //    # meaning the output is not differentiable.
            //    return None, None
            //elif isinstance(op_ctxt, CondContext):
            //  zero_grad = grad[1 - op_ctxt.branch]
            //  # At this point, we have created zero_grad guarded by the right switch.
            //  # Unfortunately, we may still get None here for not trainable data types.
            //  if zero_grad is None:
            //    # For resource variables we get None always on the other branch, so bypass
            //    # this.
            //    if op.inputs[0].dtype == dtypes.resource:
            //      return merge(
            //          [grad[op_ctxt.branch]] * 2, name="cond_resource_grad")[0], None
            //    return None, None
            //  return merge(grad, name="cond_grad")[0], None
            //else:
            //  false_grad = switch(grad[0], op.inputs[1])[0]
            //  true_grad = switch(grad[1], op.inputs[1])[1]
            //  return merge([false_grad, true_grad])[0], None
        }

        /// <summary>
        /// Gradients for a Merge op are calculated using a Switch op.
        /// </summary>
        [RegisterGradient("Merge")]
        public static Tensor[] _MergeGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var _ = grads[1];
            var input_op = op.inputs[0].op;
            var graph = ops.get_default_graph();
            var op_ctxt = control_flow_util.GetOutputContext(input_op);
            var grad_ctxt = graph._get_control_flow_context();
            switch (op_ctxt)
            {
                case WhileContext cwhile:
                    {
                        return control_flow_ops._SwitchRefOrTensor(grad, grad_ctxt.pivot);
                    }
                case CondContext ccond:
                    {
                        var pred = ccond.pred;
                        if (grad_ctxt != null && grad_ctxt.grad_state != null)
                        {
                            //# This Merge node is part of a cond within a loop.
                            //# The backprop needs to have the value of this predicate for every
                            //# iteration. So we must have its values accumulated in the forward, and
                            //# use the accumulated values as the predicate for this backprop switch.
                            var grad_state = grad_ctxt.grad_state;
                            var real_pred = grad_state.history_map[pred.name] as Tensor;
                            if (real_pred == null)
                            {
                                //# Remember the value of pred for every iteration.
                                grad_ctxt = grad_state.grad_context;
                                grad_ctxt.Exit();
                                var history_pred = grad_state.AddForwardAccumulator(pred);
                                grad_ctxt.Enter();

                                //# Add the stack pop op. If pred.op is in a (outer) CondContext,
                                //# the stack pop will be guarded with a switch.
                                real_pred = grad_state.AddBackpropAccumulatedValue(history_pred, pred);
                                grad_state.history_map[pred.name] = real_pred;
                            }
                            pred = real_pred;
                        }
                        var results = control_flow_ops._SwitchRefOrTensor(grad, pred, name: "cond_grad");
                        return results;
                    }
                default:
                    {
                        var num_inputs = op.inputs.Length;
                        var cond = new Tensor[num_inputs];
                        for (int i = 0; i < num_inputs; i++)
                            cond[i] = math_ops.equal(op.outputs[1], i);
                        var result = cond.Select(t => control_flow_ops._SwitchRefOrTensor(grad, t)[1]).ToArray();
                        return result;
                    }
            }

        }

        public Tensor[] _RefMergeGrad(Operation op, Tensor[] grads)
        {
            return _MergeGrad(op, grads);
        }

        /// <summary>
        /// Gradients for an exit op are calculated using an Enter op.
        /// </summary>
        public Tensor[] _ExitGrad(Operation op, Tensor[] grads)
        {
            throw new NotImplementedException("_ExitGrad");
            //  graph = ops.get_default_graph()
            //# pylint: disable=protected-access
            //  op_ctxt = op._get_control_flow_context()
            //  grad_ctxt = graph._get_control_flow_context()
            //  # pylint: enable=protected-access
            //  if not grad_ctxt.back_prop:
            //    # The flag `back_prop` is set by users to suppress gradient
            //    # computation for this loop. If the attribute `back_prop` is false,
            //    # no gradient computation.
            //    return None

            //  if op_ctxt.grad_state:
            //    raise TypeError("Second-order gradient for while loops not supported.")

            //  if isinstance(grad, ops.Tensor) :
            //    grad_ctxt.AddName(grad.name)
            //  else:
            //    if not isinstance(grad, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
            //      raise TypeError("Type %s not supported" % type(grad))
            //    grad_ctxt.AddName(grad.values.name)
            //    grad_ctxt.AddName(grad.indices.name)
            //    dense_shape = grad.dense_shape
            //    if dense_shape is not None:
            //      grad_ctxt.AddName(dense_shape.name)
            //  grad_ctxt.Enter()
            //  # pylint: disable=protected-access
            //  result = control_flow_ops._Enter(
            //      grad, grad_ctxt.name, is_constant=False,
            //      parallel_iterations=grad_ctxt.parallel_iterations,
            //      name="b_exit")
            //  # pylint: enable=protected-access
            //  grad_ctxt.loop_enters.append(result)
            //  grad_ctxt.Exit()
            //  return result
        }

        /// <summary>
        /// A forward next_iteration is translated into a backprop identity.
        ///
        ///  Note that the backprop next_iteration is added in switch grad.
        /// </summary>
        public (object, Tensor[]) _NextIterationGrad(object _, Tensor[] grad)
        {
            return (_, grad);
        }

        public (object, Tensor[]) _RefNextIterationGrad(object _, Tensor[] grad)
        {
            return (_, grad);
        }

        /// <summary>
        /// Gradients for an Enter are calculated using an Exit op.
        /// 
        ///  For loop variables, grad is the gradient so just add an exit.
        ///  For loop invariants, we need to add an accumulator loop.
        /// </summary>
        public (object, Tensor[]) _EnterGrad(Tensor op, Tensor[] grad)
        {
            throw new NotImplementedException("_EnterGrad");
            //  graph = ops.get_default_graph()
            //# pylint: disable=protected-access
            //  grad_ctxt = graph._get_control_flow_context()
            //  # pylint: enable=protected-access
            //  if not grad_ctxt.back_prop:
            //    # Skip gradient computation, if the attribute `back_prop` is false.
            //    return grad
            //  if grad_ctxt.grad_state is None:
            //    # Pass the gradient through if we are not in a gradient while context.
            //    return grad
            //  if op.get_attr("is_constant"):
            //    # Add a gradient accumulator for each loop invariant.
            //    if isinstance(grad, ops.Tensor) :
            //      result = grad_ctxt.AddBackpropAccumulator(op, grad)
            //    elif isinstance(grad, ops.IndexedSlices) :
            //      result = grad_ctxt.AddBackpropIndexedSlicesAccumulator(op, grad)
            //    else:
            //      # TODO(yuanbyu, lukasr): Add support for SparseTensor.
            //      raise TypeError("Type %s not supported" % type(grad))
            //  else:
            //    result = exit(grad)
            //    grad_ctxt.loop_exits.append(result)
            //    grad_ctxt.ExitResult([result])
            //  return result
        }

        public (object, Tensor[]) _RefEnterGrad(Tensor op, Tensor[] grad)
        {
            return _EnterGrad(op, grad);
        }

        /// <summary>
        /// Stop backprop for the predicate of a while loop.
        /// </summary>
        public object _LoopCondGrad(object _)
        {
            return null;
        }

    }
}
