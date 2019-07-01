using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.Engine
{
    public class Sequential : Model, IPython
    {
        public Sequential(string name = null) 
            : base(name: name)
        {
            supports_masking = true;
            _compute_output_and_mask_jointly = true;
        }

        public void __enter__()
        {
            
        }

        /// <summary>
        /// Adds a layer instance on top of the layer stack.
        /// </summary>
        /// <param name="layer"></param>
        public void add(Layer layer)
        {
            built = false;
            var set_inputs = false;
            if(_layers.Count == 0)
            {
                var (batch_shape, dtype) = (layer._batch_input_shape, layer._dtype);
                if(batch_shape != null)
                {
                    // Instantiate an input layer.
                    var x = keras.layers.Input(
                      batch_shape: batch_shape,
                      dtype: dtype,
                      name: layer.name + "_input");

                    // This will build the current layer
                    // and create the node connecting the current layer
                    // to the input layer we just created.
                    layer.__call__(x);
                    set_inputs = true;
                }
            }
        }

        public void __exit__()
        {
            
        }

        public void Dispose()
        {

        }
    }
}
