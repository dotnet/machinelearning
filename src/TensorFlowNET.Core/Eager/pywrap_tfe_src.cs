using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public class pywrap_tfe_src
    {
        public static void RecordGradient(string op_name, Tensor[] inputs, Dictionary<string, object> attrs, Tensor[] results, string name = null)
        {
            var input_ids = inputs.Select(x => x.Id).ToArray();
            var input_dtypes = inputs.Select(x => x.dtype).ToArray();

            bool should_record = false;
            foreach (var input_dtype in input_dtypes)
            {
                if (Tape.IsDtypeTrainable(input_dtype.as_datatype_enum()))
                {
                    should_record = true;
                    break;
                }
            }
            if (!should_record) return;

            var op_outputs = results;
            var op_inputs = inputs;
        }
    }
}
