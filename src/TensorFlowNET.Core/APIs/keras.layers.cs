using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;

namespace Tensorflow
{
    public static partial class keras
    {
        public static class layers
        {
            public static Embedding Embedding(int input_dim, int output_dim,
                IInitializer embeddings_initializer = null,
                bool mask_zero = false) => new Embedding(input_dim, output_dim,
                    embeddings_initializer,
                    mask_zero);

            public static Tensor[] Input(int[] batch_shape = null,
                TF_DataType dtype = TF_DataType.DtInvalid,
                string name = null,
                bool sparse = false,
                Tensor tensor = null)
            {
                var batch_size = batch_shape[0];
                var shape = batch_shape.Skip(1).ToArray();

                var input_layer = new InputLayer(
                    input_shape: shape,
                    batch_size: batch_size,
                    name: name,
                    dtype: dtype,
                    sparse: sparse,
                    input_tensor: tensor);

                var outputs = input_layer.inbound_nodes[0].output_tensors;

                return outputs;
            }
        }
    }
}
