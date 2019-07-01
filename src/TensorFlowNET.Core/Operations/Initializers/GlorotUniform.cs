using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Initializers
{
    public class GlorotUniform : VarianceScaling
    {
        public GlorotUniform(float scale = 1.0f,
            string mode = "fan_avg",
            string distribution = "uniform",
            int? seed = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT) : base(scale, mode, distribution, seed, dtype)
        {

        }

        public object get_config()
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
