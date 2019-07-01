using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_logging_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Operation _assert(Tensor condition, object[] data, int? summarize = 3, string name = null)
        {
            if (!summarize.HasValue)
                summarize = 3;

            var _op = _op_def_lib._apply_op_helper("Assert", name, args: new { condition, data, summarize });

            return _op;
        }

        public static Tensor histogram_summary(string tag, Tensor values, string name = null)
        {
            var dict = new Dictionary<string, object>();
            var op = _op_def_lib._apply_op_helper("HistogramSummary", name: name, args: new { tag, values });
            return op.output;
        }

        /// <summary>
        ///    Outputs a <c>Summary</c> protocol buffer with scalar values.
        /// </summary>
        /// <param name="tags">
        ///    Tags for the summary.
        /// </param>
        /// <param name="values">
        ///    Same shape as <c>tags.  Values for the summary.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'ScalarSummary'.
        /// </param>
        /// <returns>
        ///    Scalar.  Serialized <c>Summary</c> protocol buffer.
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    The input <c>tags</c> and <c>values</c> must have the same shape.  The generated summary
        ///    has a summary value for each tag-value pair in <c>tags</c> and <c>values</c>.
        /// </remarks>
        public static Tensor scalar_summary(string tags, Tensor values, string name = "ScalarSummary")
        {
            var dict = new Dictionary<string, object>();
            dict["tags"] = tags;
            dict["values"] = values;
            var op = _op_def_lib._apply_op_helper("ScalarSummary", name: name, keywords: dict);
            return op.output;
        }

        /// <summary>
        ///    Merges summaries.
        /// </summary>
        /// <param name="inputs">
        ///    Can be of any shape.  Each must contain serialized <c>Summary</c> protocol
        ///    buffers.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'MergeSummary'.
        /// </param>
        /// <returns>
        ///    Scalar. Serialized <c>Summary</c> protocol buffer.
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    This op creates a
        ///    [<c>Summary</c>](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
        ///    protocol buffer that contains the union of all the values in the input
        ///    summaries.
        ///    
        ///    When the Op is run, it reports an <c>InvalidArgument</c> error if multiple values
        ///    in the summaries to merge use the same tag.
        /// </remarks>
        public static Tensor merge_summary(Tensor[] inputs, string name = "MergeSummary")
        {
            var dict = new Dictionary<string, object>();
            dict["inputs"] = inputs;
            var op = _op_def_lib._apply_op_helper("MergeSummary", name: name, keywords: dict);
            return op.output;
        }
    }
}
