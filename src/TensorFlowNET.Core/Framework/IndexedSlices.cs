using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework
{
    /// <summary>
    /// A sparse representation of a set of tensor slices at given indices.
    /// </summary>
    public class IndexedSlices : CompositeTensor
    {
        Tensor _values;
        public Tensor values => _values;
        Tensor _indices;
        public Tensor indices => _indices;
        Tensor _dense_shape;
        public Tensor dense_shape => _dense_shape;

        public string name => _values.name;

        public string device => _values.Device;

        public Operation op => _values.op;

        public TF_DataType dtype => _values.dtype;

        public Graph graph => _values.graph;

        public IndexedSlices(Tensor values, Tensor indices, Tensor dense_shape = null)
        {
            _values = values;
            _indices = indices;
            _dense_shape = dense_shape;

            _values.Tag = this;
        }

        public static implicit operator Tensor(IndexedSlices indexedSlices)
        {
            return indexedSlices.values;
        }

        public static implicit operator IndexedSlices(Tensor tensor)
        {
            return tensor.Tag as IndexedSlices;
        }
    }
}
