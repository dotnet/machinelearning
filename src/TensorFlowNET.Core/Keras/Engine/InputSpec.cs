using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Specifies the ndim, dtype and shape of every input to a layer.
    /// </summary>
    public class InputSpec
    {
        public int? ndim;
        public int? min_ndim;
        Dictionary<int, int> axes;

        public InputSpec(TF_DataType dtype = TF_DataType.DtInvalid,
            int? ndim = null,
            int? min_ndim = null,
            Dictionary<int, int> axes = null)
        {
            this.ndim = ndim;
            if (axes == null)
                axes = new Dictionary<int, int>();
            this.axes = axes;
            this.min_ndim = min_ndim;
        }
    }
}
