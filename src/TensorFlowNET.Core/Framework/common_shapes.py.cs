using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework
{
    public static class common_shapes
    {
        /// <summary>
        /// Returns the broadcasted shape between `shape_x` and `shape_y
        /// </summary>
        /// <param name="shape_x"></param>
        /// <param name="shape_y"></param>
        public static Tensor broadcast_shape(Tensor shape_x, Tensor shape_y)
        {
            var return_dims = _broadcast_shape_helper(shape_x, shape_y);
            // return tensor_shape(return_dims);
            throw new NotFiniteNumberException();
        }
        /// <summary>
        /// Helper functions for is_broadcast_compatible and broadcast_shape.
        /// </summary>
        /// <param name="shape_x"> A `TensorShape`</param>
        /// <param name="shape_y"> A `TensorShape`</param>
        /// <return> Returns None if the shapes are not broadcast compatible,
        /// a list of the broadcast dimensions otherwise.
        /// </return>
        public static Tensor _broadcast_shape_helper(Tensor shape_x, Tensor shape_y)
        {
            throw new NotFiniteNumberException();
        }

        public static int? rank(Tensor tensor)
        {
            return tensor.rank;
        }

        public static bool has_fully_defined_shape(Tensor tensor)
        {
            return tensor.GetShape().is_fully_defined();
        }
    }
}
