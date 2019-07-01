using Google.Protobuf.Collections;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Represents the shape of a `Tensor`.
    /// </summary>
    public class TensorShape : Shape
    {
        public TensorShape(TensorShapeProto proto)
        {
            if (proto.UnknownRank) return;

            ReShape(proto.Dim.Select(x => (int)x.Size).ToArray());
        }

        public TensorShape(params int[] dims) : base(dims)
        {

        }

        /// <summary>
        /// Returns True iff `self` is fully defined in every dimension.
        /// </summary>
        /// <returns></returns>
        public bool is_fully_defined()
        {
            return Dimensions != null && Dimensions.Count(x => x < 1) == 0;
        }

        public bool is_compatible_with(TensorShape shape2)
        {
            throw new NotImplementedException("TensorShape is_compatible_with");
        }
    }
}
