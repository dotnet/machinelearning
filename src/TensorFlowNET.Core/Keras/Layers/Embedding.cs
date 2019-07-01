using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Layers
{
    public class Embedding : Layer
    {
        private int input_dim;
        private int output_dim;
        private bool mask_zero;
        public RefVariable embeddings;
        public IInitializer embeddings_initializer;

        public Embedding(int input_dim, int output_dim,
            IInitializer embeddings_initializer = null,
            bool mask_zero = false,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int[] input_shape = null) : base(dtype: dtype, input_shape: input_shape)
        {
            this.input_dim = input_dim;
            this.output_dim = output_dim;
            this.embeddings_initializer = embeddings_initializer == null ? tf.uniform_initializer : embeddings_initializer;
            this.mask_zero = mask_zero;
            supports_masking = mask_zero;
        }

        protected override void build(TensorShape input_shape)
        {
            embeddings = add_weight(shape: new int[] { input_dim, output_dim },
                initializer: embeddings_initializer,
                name: "embeddings");
            built = true;
        }
    }
}
