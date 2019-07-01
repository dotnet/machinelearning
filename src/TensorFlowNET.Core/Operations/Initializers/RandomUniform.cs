using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Initializers
{
    public class RandomUniform : IInitializer
    {
        private int? seed;
        private float minval;
        private float maxval;
        private TF_DataType dtype;

        public RandomUniform()
        {

        }

        public Tensor call(TensorShape shape, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            return random_ops.random_uniform(shape, 
                minval: minval, 
                maxval: maxval, 
                dtype: dtype, 
                seed: seed);
        }

        public object get_config()
        {
            return new {
                minval,
                maxval,
                seed,
                dtype
            };
        }
    }
}
