using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Operations.Initializers
{
    /// <summary>
    /// Initializer capable of adapting its scale to the shape of weights tensors.
    /// </summary>
    public class VarianceScaling : IInitializer
    {
        protected float _scale;
        protected string _mode;
        protected string _distribution;
        protected int? _seed;
        protected TF_DataType _dtype;

        public VarianceScaling(float scale = 1.0f,
            string mode = "fan_in",
            string distribution = "truncated_normal",
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            if (scale < 0)
                throw new ValueError("`scale` must be positive float.");
            _scale = scale;
            _mode = mode;
            _distribution = distribution;
            _seed = seed;
            _dtype = dtype;
        }

        public Tensor call(TensorShape shape, TF_DataType dtype)
        {
            var (fan_in, fan_out) = _compute_fans(shape);
            if (_mode == "fan_in")
                _scale /= Math.Max(1, fan_in);
            else if (_mode == "fan_out")
                _scale /= Math.Max(1, fan_out);
            else
                _scale /= Math.Max(1, (fan_in + fan_out) / 2);

            if (_distribution == "normal" || _distribution == "truncated_normal")
            {
                float stddev = (float)Math.Sqrt(_scale) / .87962566103423978f;
                return random_ops.truncated_normal(shape, mean: 0.0f, stddev: stddev, dtype: dtype, seed: _seed);
            }
            else if (_distribution == "untruncated_normal")
            {
                throw new NotImplementedException("truncated_normal");
            }
            else
            {
                var limit = Math.Sqrt(3.0f * _scale);
                return random_ops.random_uniform(shape, (float)-limit, (float)limit, dtype, seed: _seed);
            }
        }

        private (int, int) _compute_fans(int[] shape)
        {
            if (shape.Length < 1)
                return (1, 1);
            if (shape.Length == 1)
                return (shape[0], shape[0]);
            if (shape.Length == 2)
                return (shape[0], shape[1]);
            else
            {
                // Assuming convolution kernels (2D, 3D, or more).
                // kernel shape: (..., input_depth, depth)
                int receptive_field_size = 1;
                foreach (var dim in shape.Take(2))
                    receptive_field_size *= dim;
                var fan_in = shape[shape.Length - 2] * receptive_field_size;
                var fan_out = shape[shape.Length - 1] * receptive_field_size;
                return (fan_in, fan_out);
            }
        }

        public virtual object get_config()
        {
            return new
            {
                scale = _scale,
                mode = _mode,
                distribution = _distribution,
                seed = _seed,
                dtype = _dtype
            };
        }
    }
}
