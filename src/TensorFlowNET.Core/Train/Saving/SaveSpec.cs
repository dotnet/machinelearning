using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Class used to describe tensor slices that need to be saved.
    /// </summary>
    public class SaveSpec
    {
        private Tensor _tensor;
        public Tensor tensor => _tensor;

        private string _slice_spec;
        public string slice_spec => _slice_spec;

        private string _name;
        public string name => _name;

        private TF_DataType _dtype;
        public TF_DataType dtype => _dtype;

        public SaveSpec(Tensor tensor, string slice_spec, string name, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            _tensor = tensor;
            _slice_spec = slice_spec;
            _name = name;
            _dtype = dtype;
        }
    }
}
