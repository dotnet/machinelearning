using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Tensorflow.Keras
{
    public class Sequence
    {
        /// <summary>
        /// Pads sequences to the same length.
        /// https://keras.io/preprocessing/sequence/
        /// https://faroit.github.io/keras-docs/1.2.0/preprocessing/sequence/
        /// </summary>
        /// <param name="sequences">List of lists, where each element is a sequence.</param>
        /// <param name="maxlen">Int, maximum length of all sequences.</param>
        /// <param name="dtype">Type of the output sequences.</param>
        /// <param name="padding">String, 'pre' or 'post':</param>
        /// <param name="truncating">String, 'pre' or 'post'</param>
        /// <param name="value">Float or String, padding value.</param>
        /// <returns></returns>
        public NDArray pad_sequences(NDArray sequences, 
            int? maxlen = null,
            string dtype = "int32",
            string padding = "pre",
            string truncating = "pre",
            object value = null)
        {
            int[] length = new int[sequences.size];

            if (maxlen == null)
                maxlen = length.Max();

            if (value == null)
                value = 0f;

            var nd = new NDArray(np.int32, new Shape(sequences.size, maxlen.Value));
            for (int i = 0; i < nd.shape[0]; i++)
            {
                switch(sequences[i])
                {
                    default:
                        throw new NotImplementedException("pad_sequences");
                }
            }

            return nd;
        }
    }
}
