using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class c_api_util
    {
        public static TF_Output tf_output(IntPtr c_op, int index) => new TF_Output(c_op, index);

        public static ImportGraphDefOptions ScopedTFImportGraphDefOptions() => new ImportGraphDefOptions();

        public static Buffer tf_buffer(byte[] data) => new Buffer(data);

        public static IEnumerable<Operation> new_tf_operations(Graph graph)
        {
            foreach (var c_op in tf_operations(graph))
            {
                if (graph._get_operation_by_tf_operation(c_op) == null)
                    yield return c_op;
            }
        }

        public static IEnumerable<Operation> tf_operations(Graph graph)
        {
            uint pos = 0;
            IntPtr c_op;
            while ((c_op = c_api.TF_GraphNextOperation(graph, ref pos)) != IntPtr.Zero)
            {
                yield return c_op;
            }
        }
    }
}
