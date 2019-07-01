using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Destroy an options object.  Graph will be deleted once no more
        /// TFSession's are referencing it.
        /// </summary>
        /// <param name="graph"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteGraph(IntPtr graph);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteImportGraphDefOptions(IntPtr opts);

        /// <summary>
        /// Deletes a results object returned by TF_GraphImportGraphDefWithResults().
        /// </summary>
        /// <param name="results"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteImportGraphDefResults(IntPtr results);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphGetOpDef(IntPtr graph, string op_name, IntPtr output_op_def, IntPtr status);

        /// <summary>
        /// Returns the shape of the Tensor referenced by `output` in `graph`
        /// into `dims`. `dims` must be an array large enough to hold `num_dims`
        /// entries (e.g., the return value of TF_GraphGetTensorNumDims).
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="output"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphGetTensorShape(IntPtr graph, TF_Output output, long[] dims, int num_dims, IntPtr status);

        /// <summary>
        /// Import the graph serialized in `graph_def` into `graph`.
        /// Convenience function for when only return outputs are needed.
        ///
        /// `num_return_outputs` must be the number of return outputs added (i.e. the
        /// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
        /// `num_return_outputs` is non-zero, `return_outputs` must be of length
        /// `num_return_outputs`. Otherwise it can be null.
        /// </summary>
        /// <param name="graph">TF_Graph* graph</param>
        /// <param name="graph_def">const TF_Buffer*</param>
        /// <param name="options">const TF_ImportGraphDefOptions*</param>
        /// <param name="return_outputs">TF_Output*</param>
        /// <param name="num_return_outputs">int</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_GraphImportGraphDefWithReturnOutputs(IntPtr graph, IntPtr graph_def, IntPtr options, IntPtr return_outputs, int num_return_outputs, IntPtr status);

        /// <summary>
        /// Import the graph serialized in `graph_def` into `graph`.  Returns nullptr and
        /// a bad status on error. Otherwise, returns a populated
        /// TF_ImportGraphDefResults instance. The returned instance must be deleted via
        /// TF_DeleteImportGraphDefResults().
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="graph_def">const TF_Buffer*</param>
        /// <param name="options">const TF_ImportGraphDefOptions*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>TF_ImportGraphDefResults*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GraphImportGraphDefWithResults(IntPtr graph, IntPtr graph_def, IntPtr options, IntPtr status);

        /// <summary>
        /// Import the graph serialized in `graph_def` into `graph`.
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="graph_def">TF_Buffer*</param>
        /// <param name="options">TF_ImportGraphDefOptions*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphImportGraphDef(IntPtr graph, IntPtr graph_def, IntPtr options, IntPtr status);
        /// <summary>
        /// Iterate through the operations of a graph.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="pos"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GraphNextOperation(IntPtr graph, ref uint pos);

        /// <summary>
        /// Returns the operation in the graph with `oper_name`. Returns nullptr if
        /// no operation found.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="oper_name"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GraphOperationByName(IntPtr graph, string oper_name);

        /// <summary>
        /// Sets the shape of the Tensor referenced by `output` in `graph` to
        /// the shape described by `dims` and `num_dims`.
        /// </summary>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphSetTensorShape(IntPtr graph, TF_Output output, long[] dims, int num_dims, IntPtr status);

        /// <summary>
        /// Write out a serialized representation of `graph` (as a GraphDef protocol
        /// message) to `output_graph_def` (allocated by TF_NewBuffer()).
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="output_graph_def">TF_Buffer*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphToGraphDef(IntPtr graph, IntPtr output_graph_def, IntPtr status);
        
        /// <summary>
        /// Returns the number of dimensions of the Tensor referenced by `output`
        /// in `graph`.
        /// 
        /// If the number of dimensions in the shape is unknown, returns -1.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="output"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_GraphGetTensorNumDims(IntPtr graph, TF_Output output, IntPtr status);

        /// <summary>
        /// Cause the imported graph to have a control dependency on `oper`. `oper`
        /// should exist in the graph being imported into.
        /// </summary>
        /// <param name="opts"></param>
        /// <param name="oper"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddControlDependency(IntPtr opts, IntPtr oper);

        /// <summary>
        /// Set any imported nodes with input `src_name:src_index` to have that input
        /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
        /// `dst` references a node already existing in the graph being imported into.
        /// `src_name` is copied and has no lifetime requirements.
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        /// <param name="src_name">const char*</param>
        /// <param name="src_index">int</param>
        /// <param name="dst">TF_Output</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddInputMapping(IntPtr opts, string src_name, int src_index, TF_Output dst);

        /// <summary>
        /// Add an operation in `graph_def` to be returned via the `return_opers` output
        /// parameter of TF_GraphImportGraphDef(). `oper_name` is copied and has no
        // lifetime requirements.
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions* opts</param>
        /// <param name="oper_name">const char*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddReturnOperation(IntPtr opts, string oper_name);

        /// <summary>
        /// Add an output in `graph_def` to be returned via the `return_outputs` output
        /// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
        /// mapping, the corresponding existing tensor in `graph` will be returned.
        /// `oper_name` is copied and has no lifetime requirements.
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        /// <param name="oper_name">const char*</param>
        /// <param name="index">int</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddReturnOutput(IntPtr opts, string oper_name, int index);

        /// <summary>
        /// Returns the number of return operations added via
        /// TF_ImportGraphDefOptionsAddReturnOperation().
        /// </summary>
        /// <param name="opts"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_ImportGraphDefOptionsNumReturnOperations(IntPtr opts);

        /// <summary>
        /// Returns the number of return outputs added via
        /// TF_ImportGraphDefOptionsAddReturnOutput().
        /// </summary>
        /// <param name="opts">const TF_ImportGraphDefOptions*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_ImportGraphDefOptionsNumReturnOutputs(IntPtr opts);

        /// <summary>
        /// Set any imported nodes with control input `src_name` to have that input
        /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
        /// `dst` references an operation already existing in the graph being imported
        /// into. `src_name` is copied and has no lifetime requirements. 
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        /// <param name="src_name">const char*</param>
        /// <param name="dst">TF_Operation*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsRemapControlDependency(IntPtr opts, string src_name, IntPtr dst);

        /// <summary>
        /// Set the prefix to be prepended to the names of nodes in `graph_def` that will
        /// be imported into `graph`. `prefix` is copied and has no lifetime
        /// requirements.
        /// </summary>
        /// <param name="ops"></param>
        /// <param name="prefix"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsSetPrefix(IntPtr ops, string prefix);

        /// <summary>
        /// Set whether to uniquify imported operation names. If true, imported operation
        /// names will be modified if their name already exists in the graph. If false,
        /// conflicting names will be treated as an error. Note that this option has no
        /// effect if a prefix is set, since the prefix will guarantee all names are
        /// unique. Defaults to false.
        /// </summary>
        /// <param name="ops">TF_ImportGraphDefOptions*</param>
        /// <param name="uniquify_prefix">unsigned char</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsSetUniquifyNames(IntPtr ops, char uniquify_prefix);

        /// <summary>
        /// Fetches the return operations requested via
        /// TF_ImportGraphDefOptionsAddReturnOperation(). The number of fetched
        /// operations is returned in `num_opers`. The array of return operations is
        /// returned in `opers`. `*opers` is owned by and has the lifetime of `results`.
        /// </summary>
        /// <param name="results">TF_ImportGraphDefResults*</param>
        /// <param name="num_opers">int*</param>
        /// <param name="opers">TF_Operation***</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefResultsReturnOperations(IntPtr results, ref int num_opers, ref TF_Operation opers);

        /// <summary>
        /// Fetches the return outputs requested via
        /// TF_ImportGraphDefOptionsAddReturnOutput(). The number of fetched outputs is
        /// returned in `num_outputs`. The array of return outputs is returned in
        /// `outputs`. `*outputs` is owned by and has the lifetime of `results`.
        /// </summary>
        /// <param name="results">TF_ImportGraphDefResults* results</param>
        /// <param name="num_outputs">int*</param>
        /// <param name="outputs">TF_Output**</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefResultsReturnOutputs(IntPtr results, ref int num_outputs, ref IntPtr outputs);

        /// <summary>
        /// This function creates a new TF_Session (which is created on success) using
        /// `session_options`, and then initializes state (restoring tensors and other
        /// assets) using `run_options`.
        /// </summary>
        /// <param name="session_options">const TF_SessionOptions*</param>
        /// <param name="run_options">const TF_Buffer*</param>
        /// <param name="export_dir">const char*</param>
        /// <param name="tags">const char* const*</param>
        /// <param name="tags_len">int</param>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="meta_graph_def">TF_Buffer*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_LoadSessionFromSavedModel(IntPtr session_options, IntPtr run_options,
            string export_dir, string[] tags, int tags_len,
            IntPtr graph, ref TF_Buffer meta_graph_def, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewGraph();

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewImportGraphDefOptions();

        /// <summary>
        /// Updates 'dst' to consume 'new_src'.
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="new_src"></param>
        /// <param name="dst"></param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        
        public static extern void UpdateEdge(IntPtr graph, TF_Output new_src, TF_Input dst, IntPtr status);
    }
}
