using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Initializers
{
    public class TruncatedNormal : IInitializer
    {
        private float mean;
        private float stddev;
        private int? seed;
        private TF_DataType dtype;

        public TruncatedNormal(float mean = 0.0f,
            float stddev = 1.0f,
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            this.mean = mean;
            this.stddev = stddev;
            this.seed = seed;
            this.dtype = dtype;
        }

        public Tensor call(TensorShape shape, TF_DataType dtype)
        {
            return random_ops.truncated_normal(shape, mean, stddev, dtype : dtype, seed: seed);
        }

        public object get_config()
        {
            return new
            {
                mean = mean,
                stddev = stddev,
                seed = seed,
                dtype = dtype.name()
            };
        }
    }
}
