using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Layer to be used as an entry point into a Network (a graph of layers).
    /// </summary>
    public class InputLayer : Layer
    {
        public bool sparse;
        public int? batch_size;
        public bool is_placeholder;

        public InputLayer(int[] input_shape = null,
            int? batch_size = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string name = null,
            bool sparse = false,
            Tensor input_tensor = null)
        {
            built = true;
            this.sparse = sparse;
            this.batch_size = batch_size;
            this.supports_masking = true;

            if (input_tensor == null)
            {
                var batch_input_shape = new int[] { batch_size.HasValue ? batch_size.Value : -1, -1 };

                if (sparse)
                {
                    throw new NotImplementedException("InputLayer sparse is true");
                }
                else
                {
                    input_tensor = backend.placeholder(
                          shape: batch_input_shape,
                          dtype: dtype,
                          name: name);
                }

                is_placeholder = true;
                _batch_input_shape = batch_input_shape;
            }

            new Node(this,
                inbound_layers: new Layer[0],
                node_indices: new int[0],
                tensor_indices: new int[0],
                input_tensors: new Tensor[] { input_tensor },
                output_tensors: new Tensor[] { input_tensor });
        }
    }
}
