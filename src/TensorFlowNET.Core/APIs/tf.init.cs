using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Initializers;

namespace Tensorflow
{
    public static partial class tf
    {
        public static IInitializer zeros_initializer => new Zeros();
        public static IInitializer ones_initializer => new Ones();
        public static IInitializer glorot_uniform_initializer => new GlorotUniform();
        public static IInitializer uniform_initializer => new RandomUniform();

        public static variable_scope variable_scope(string name,
               string default_name = null,
               Tensor[] values = null,
               bool? reuse = null,
               bool auxiliary_name_scope = true) => new variable_scope(name,
                   default_name,
                   values,
                   reuse: reuse,
                   auxiliary_name_scope: auxiliary_name_scope);

        public static variable_scope variable_scope(VariableScope scope,
              string default_name = null,
              Tensor[] values = null,
              bool? reuse = null,
              bool auxiliary_name_scope = true) => new variable_scope(scope,
                  default_name,
                  values,
                  reuse: reuse,
                  auxiliary_name_scope: auxiliary_name_scope);

        public static IInitializer truncated_normal_initializer(float mean = 0.0f,
            float stddev = 1.0f,
            int? seed = null,
            TF_DataType dtype = TF_DataType.DtInvalid) => new TruncatedNormal(mean: mean,
                stddev: stddev,
                seed: seed,
                dtype: dtype);
    }
}
