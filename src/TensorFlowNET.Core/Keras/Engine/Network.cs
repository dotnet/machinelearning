using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.Engine
{
    public class Network : Layer
    {
        protected bool _is_compiled;
        protected bool _expects_training_arg;
        protected bool _compute_output_and_mask_jointly;
        /// <summary>
        /// All layers in order of horizontal graph traversal.
        /// Entries are unique. Includes input and output layers.
        /// </summary>
        protected List<Layer> _layers;

        public Network(string name = null) 
            : base(name: name)
        {
            _init_subclassed_network(name);
        }

        protected virtual void _init_subclassed_network(string name = null)
        {
            _base_init(name: name);
        }

        protected virtual void _base_init(string name = null)
        {
            _init_set_name(name);
            trainable = true;
            _is_compiled = false;
            _expects_training_arg = false;
            _compute_output_and_mask_jointly = false;
            supports_masking = false;
            _layers = new List<Layer>();
        }
    }
}
