using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Operations
{
    public class _WithSpaceToBatch
    {
        private _NonAtrousConvolution call;

        public _WithSpaceToBatch(TensorShape input_shape,
            int[] dilation_rate,
            string padding,
            Func<int, string, _NonAtrousConvolution> build_op,
            TensorShape filter_shape = null,
            int[] spatial_dims = null,
            string data_format = null)
        {
            var dilation_rate_tensor = ops.convert_to_tensor(dilation_rate, TF_DataType.TF_INT32, name: "dilation_rate");
            var rate_shape = dilation_rate_tensor.GetShape();
            var num_spatial_dims = rate_shape.Dimensions[0];
            int starting_spatial_dim = -1;
            if (!string.IsNullOrEmpty(data_format) && data_format.StartsWith("NC"))
                starting_spatial_dim = 2;
            else
                starting_spatial_dim = 1;

            if (spatial_dims == null)
                throw new NotImplementedException("_WithSpaceToBatch spatial_dims");

            var orig_spatial_dims = spatial_dims;
            spatial_dims = spatial_dims.OrderBy(x => x).ToArray();
            if (!Enumerable.SequenceEqual(spatial_dims, orig_spatial_dims) || spatial_dims.Any(x => x < 1))
                throw new ValueError("spatial_dims must be a montonically increasing sequence of positive integers");

            int expected_input_rank = -1;
            if (!string.IsNullOrEmpty(data_format) && data_format.StartsWith("NC"))
                expected_input_rank = spatial_dims.Last();
            else
                expected_input_rank = spatial_dims.Last() + 1;

            var const_rate = tensor_util.constant_value(dilation_rate_tensor);
            var rate_or_const_rate = dilation_rate;
            if(!(const_rate is null))
            {
                if (const_rate.Data<int>().Count(x => x == 1) == const_rate.size)
                {
                    call = build_op(num_spatial_dims, padding);
                    return;
                }
            }
        }

        public Tensor __call__(Tensor inp, RefVariable filter)
        {
            return call.__call__(inp, filter);
        }
    }
}
