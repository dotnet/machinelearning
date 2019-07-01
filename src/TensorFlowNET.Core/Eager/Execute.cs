using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class Execute
    {
        public void record_gradient(string op_name, InputList inputs, Dictionary<string, object> attrs, Tensor[] results, string name = null)
        {
            pywrap_tfe_src.RecordGradient(op_name, inputs._inputs, attrs, results, name);
        }
    }
}
