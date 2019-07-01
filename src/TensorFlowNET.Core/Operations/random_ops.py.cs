using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class random_ops
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor random_normal(int[] shape, 
            float mean = 0.0f, 
            float stddev = 1.0f, 
            TF_DataType dtype = TF_DataType.TF_FLOAT, 
            int? seed = null, 
            string name = null)
        {
            return with(ops.name_scope(name, "random_normal", new { shape, mean, stddev }), scope =>
            {
                var shape_tensor = _ShapeTensor(shape);
                var mean_tensor = ops.convert_to_tensor(mean, dtype: dtype, name: "mean");
                var stddev_tensor = ops.convert_to_tensor(stddev, dtype: dtype, name = "stddev");
                var (seed1, seed2) = random_seed.get_seed(seed);
                var rnd = gen_random_ops.random_standard_normal(shape_tensor, dtype: dtype, seed: seed1, seed2: seed2);
                var mul = rnd * stddev_tensor;
                var value = math_ops.add(mul, mean_tensor, name: name);
                return value;
            });
        }

        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="dtype">The type of the output</param>
        /// <param name="seed">Used to create a random seed for the distribution.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>A tensor of the specified shape filled with random uniform values.</returns>
        public static Tensor random_uniform(int[] shape, 
            float minval = 0,
            float maxval = 1,
            TF_DataType dtype = TF_DataType.TF_FLOAT, 
            int? seed = null, 
            string name = null)
        {
            return with(ops.name_scope(name, "random_uniform", new { shape, minval, maxval }), scope =>
            {
                name = scope;
                var tensorShape = _ShapeTensor(shape);
                var minTensor = ops.convert_to_tensor(minval, dtype: dtype, name: "min");
                var maxTensor = ops.convert_to_tensor(maxval, dtype: dtype, name: "max");
                var rnd = gen_random_ops.random_uniform(tensorShape, dtype);
                return math_ops.add(rnd * (maxTensor - minTensor), minTensor, name: name);
            });
        }

        public static Tensor random_uniform(Tensor shape,
            long minval = 0,
            Tensor maxval = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
        {
            return with(ops.name_scope(name, "random_uniform", new { shape, minval, maxval }), scope =>
            {
                name = scope;
                var minTensor = ops.convert_to_tensor(minval, dtype: dtype, name: "min");
                var maxTensor = ops.convert_to_tensor(maxval == null ? 1 : maxval, dtype: dtype, name: "max");
                var (seed1, seed2) = random_seed.get_seed(seed);
                if (dtype.is_integer())
                {
                    return gen_random_ops.random_uniform_int(shape, minTensor, maxTensor, seed: seed1, seed2: seed2, name: name);
                }
                else
                {
                    var rnd = gen_random_ops.random_uniform(shape, dtype);
                    return math_ops.add(rnd * (maxTensor - minTensor), minTensor, name: name);
                }
            });
        }

        public static Tensor truncated_normal(int[] shape,
            float mean = 0.0f,
            float stddev = 1.0f,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
        {
            return with(ops.name_scope(name, "truncated_normal", new { shape, mean, stddev }), scope =>
            {
                name = scope;
                var shape_tensor = _ShapeTensor(shape);
                var mean_tensor = ops.convert_to_tensor(mean, dtype: dtype, name: "mean");
                var stddev_tensor = ops.convert_to_tensor(stddev, dtype: dtype, name: "stddev");
                var (seed1, seed2) = random_seed.get_seed(seed);
                var rnd = gen_random_ops.truncated_normal(shape_tensor, dtype, seed: seed1, seed2: seed2);
                var mul = rnd * stddev_tensor;
                var value = math_ops.add(mul, mean_tensor, name: name);
                return value;
            });
        }

        private static Tensor _ShapeTensor(int[] shape)
        {
            return ops.convert_to_tensor(shape, name: "shape");
        }
    }
}

