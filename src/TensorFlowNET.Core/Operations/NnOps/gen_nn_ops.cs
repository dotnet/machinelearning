using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Operations
{
    public class gen_nn_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        /// <summary>
        /// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
        /// 
        /// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
        /// and a filter / kernel tensor of shape
        /// `[filter_height, filter_width, in_channels, out_channels]`, this op
        /// performs the following:
        /// 
        /// 1. Flattens the filter to a 2-D matrix with shape
        ///    `[filter_height * filter_width * in_channels, output_channels]`.
        /// 2. Extracts image patches from the input tensor to form a *virtual*
        ///    tensor of shape `[batch, out_height, out_width,
        ///    filter_height * filter_width * in_channels]`.
        /// 3. For each patch, right-multiplies the filter matrix and the image patch
        ///    vector.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static Tensor conv2d(Conv2dParams parameters)
        {
            var _op = _op_def_lib._apply_op_helper("Conv2D", name: parameters.Name, args: new
            {
                input = parameters.Input,
                filter = parameters.Filter,
                strides = parameters.Strides,
                padding = parameters.Padding,
                use_cudnn_on_gpu = parameters.UseCudnnOnGpu,
                explicit_paddings = parameters.ExplicitPaddings,
                data_format = parameters.DataFormat,
                dilations = parameters.Dilations
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes the gradients of convolution with respect to the filter.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static Tensor conv2d_backprop_filter(Conv2dParams parameters)
        {
            var _op = _op_def_lib._apply_op_helper("Conv2DBackpropFilter", name: parameters.Name, args: new
            {
                input = parameters.Input,
                filter_sizes = parameters.FilterSizes,
                out_backprop = parameters.OutBackProp,
                strides = parameters.Strides,
                padding = parameters.Padding,
                use_cudnn_on_gpu = parameters.UseCudnnOnGpu,
                explicit_paddings = parameters.ExplicitPaddings,
                data_format = parameters.DataFormat,
                dilations = parameters.Dilations
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes the gradients of convolution with respect to the input.
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static Tensor conv2d_backprop_input(Conv2dParams parameters)
        {
            var _op = _op_def_lib._apply_op_helper("Conv2DBackpropInput", name: parameters.Name, args: new
            {
                input_sizes = parameters.InputSizes,
                filter = parameters.Filter,
                out_backprop = parameters.OutBackProp,
                strides = parameters.Strides,
                padding = parameters.Padding,
                use_cudnn_on_gpu = parameters.UseCudnnOnGpu,
                explicit_paddings = parameters.ExplicitPaddings,
                data_format = parameters.DataFormat,
                dilations = parameters.Dilations
            });

            return _op.outputs[0];
        }

        public static Tensor bias_add(Tensor value,
            Tensor bias,
            string data_format = null,
            string name = null)
        {
            if (data_format == null)
                data_format = "NHWC";

            var _op = _op_def_lib._apply_op_helper("BiasAdd", name: name, args: new
            {
                value,
                bias,
                data_format
            });

            return _op.outputs[0];
        }

        public static Tensor bias_add_grad(Tensor out_backprop,
            string data_format = "NHWC",
            string name = null)
        {
            if (data_format == null)
                data_format = "NHWC";

            var _op = _op_def_lib._apply_op_helper("BiasAddGrad", name: name, args: new
            {
                out_backprop,
                data_format
            });

            return _op.outputs[0];
        }

        public static Tensor[] _fused_batch_norm(Tensor x,
                Tensor scale,
                Tensor offset,
                Tensor mean,
                Tensor variance,
                float epsilon = 0.0001f,
                string data_format = "NHWC",
                bool is_training = true,
                string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("FusedBatchNorm", name: name, args: new
            {
                x,
                scale,
                offset,
                mean,
                variance,
                epsilon,
                data_format,
                is_training
            });

            return _op.outputs;
        }

        public static Tensor log_softmax(Tensor logits, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("LogSoftmax", name: name, args: new
            {
                logits
            });

            return _op.outputs[0];
        }

        public static Tensor max_pool(Tensor input,
            int[] ksize,
            int[] strides,
            string padding,
            string data_format = "NHWC",
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("MaxPool", name: name, args: new
            {
                input,
                ksize,
                strides,
                padding,
                data_format,
            });

            return _op.outputs[0];
        }

        public static Tensor max_pool_grad(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, 
            string data_format= "NHWC", string name= null)
        {
            var _op = _op_def_lib._apply_op_helper("MaxPoolGrad", name: name, args: new
            {
                orig_input,
                orig_output,
                grad,
                ksize,
                strides,
                padding,
                data_format
            });

            return _op.outputs[0];
        }

        public static Tensor[] top_kv2(Tensor input, int k, bool sorted = true, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("TopKV2", name: name, args: new
            {
                input,
                k,
                sorted
            });

            return _op.outputs;
        }

        public static Tensor relu_grad(Tensor gradients, Tensor features, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ReluGrad", name: name, args: new
            {
                gradients,
                features
            });

            return _op.outputs[0];
        }

        public static Tensor softmax(Tensor logits, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Softmax", name: name, args: new
            {
                logits
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features"></param>
        /// <param name="labels"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("SoftmaxCrossEntropyWithLogits", name: name, args: new
            {
                features,
                labels
            });

            return (_op.outputs[0], _op.outputs[1]);
        }

        /// <summary>
        ///    Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features">
        ///    batch_size x num_classes matrix
        /// </param>
        /// <param name="labels">
        ///    batch_size vector with values in [0, num_classes).
        ///    This is the label for the given minibatch entry.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'SparseSoftmaxCrossEntropyWithLogits'.
        /// </param>
        /// <returns>
        ///    Returns a tuple with multiple values, as follows:
        ///    loss : Per example loss (batch_size vector).
        ///    backprop : backpropagated gradients (batch_size x num_classes matrix).
        ///    The Operation can be fetched from any of the Tensorreturned in the tuple values, by fetching the Operation property.
        /// </returns>
        /// <remarks>
        ///    Unlike <c>SoftmaxCrossEntropyWithLogits</c>, this operation does not accept
        ///    a matrix of label probabilities, but rather a single label per row
        ///    of features.  This label is considered to have probability 1.0 for the
        ///    given row.
        ///    
        ///    Inputs are the logits, not probabilities.
        /// </remarks>
        public static (Tensor loss, Tensor backprop) sparse_softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string name = "SparseSoftmaxCrossEntropyWithLogits")
        {
            var op = _op_def_lib._apply_op_helper("SparseSoftmaxCrossEntropyWithLogits", name: name, args: new { features, labels });
            int _idx = 0;
            var loss = op.outputs[_idx++];
            var backprop = op.outputs[_idx++];
            return (loss, backprop);
        }

        /// <summary>
        /// Computes rectified linear: `max(features, 0)`.
        /// </summary>
        /// <param name="features">A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `features`.</returns>
        public static Tensor relu(Tensor features, string name = null)
        {

            //_ctx = _context._context
            //if _ctx is not None and _ctx._eager_context.is_eager:
            //  try:
            //    _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
            //      _ctx._context_handle, _ctx._eager_context.device_name, "Relu", name,
            //      _ctx._post_execution_callbacks, features)
            //    return _result
            //  except _core._FallbackException:
            //    try:
            //      return relu_eager_fallback(
            //          features, name=name, ctx=_ctx)
            //    except _core._SymbolicException:
            //      pass  # Add nodes to the TensorFlow graph.
            //    except (TypeError, ValueError):
            //      result = _dispatch.dispatch(
            //            relu, features=features, name=name)
            //      if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            //        return result
            //      raise
            //  except _core._NotOkStatusException as e:
            //    if name is not None:
            //      message = e.message + " name: " + name
            //    else:
            //      message = e.message
            //    _six.raise_from(_core._status_to_exception(e.code, message), None)
            //# Add nodes to the TensorFlow graph.
            //try:
            OpDefLibrary _op_def_lib = new OpDefLibrary();
            var _op = _op_def_lib._apply_op_helper("Relu", name: name, args: new { features });
            return _op.outputs[0];
            //except (TypeError, ValueError):
            //  result = _dispatch.dispatch(
            //        relu, features=features, name=name)
            //  if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            //    return result
            //  raise
            // var _result = _op.outputs.ToArray();
            //_inputs_flat = _op.inputs
            //_attrs = ("T", _op.get_attr("T"))
            //_execute.record_gradient(
            //    "Relu", _inputs_flat, _attrs, _result, name)
            //_result, = _result
            // return _result;
        }
    }
}
