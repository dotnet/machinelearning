using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Request that `desc` be co-located on the device where `op`
        /// is placed.
        ///
        /// Use of this is discouraged since the implementation of device placement is
        /// subject to change. Primarily intended for internal libraries 
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="op"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ColocateWith(IntPtr desc, IntPtr op);

        /// <summary>
        /// Get the OpList of all OpDefs defined in this address space.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GetAllOpList();

        /// <summary>
        /// For inputs that take a single tensor.
        /// </summary>
        /// <param name="desc">TF_OperationDescription*</param>
        /// <param name="input">TF_Output</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_AddInput(IntPtr desc, TF_Output input);

        /// <summary>
        /// Call once per control input to `desc`.
        /// </summary>
        /// <param name="desc">TF_OperationDescription*</param>
        /// <param name="input">TF_Operation*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_AddControlInput(IntPtr desc, IntPtr input);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="op">TF_Operation*</param>
        /// <param name="input">TF_Operation*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void AddControlInput(IntPtr graph, IntPtr op, IntPtr input);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="op">TF_Operation*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void RemoveAllControlInputs(IntPtr graph, IntPtr op);

        /// <summary>
        /// For inputs that take a list of tensors.
        /// inputs must point to TF_Output[num_inputs].
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="inputs"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_AddInputList(IntPtr desc, TF_Output[] inputs, int num_inputs);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_FinishOperation(IntPtr desc, IntPtr status);

        /// <summary>
        /// Operation will only be added to *graph when TF_FinishOperation() is
        /// called (assuming TF_FinishOperation() does not return an error).
        /// *graph must not be deleted until after TF_FinishOperation() is
        /// called.
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="opType">const char*</param>
        /// <param name="oper_name">const char*</param>
        /// <returns>TF_OperationDescription*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewOperation(IntPtr graph, string opType, string oper_name);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_OperationDevice(IntPtr oper);

        /// <summary>
        /// Get list of all control inputs to an operation.  `control_inputs` must
        /// point to an array of length `max_control_inputs` (ideally set to
        /// TF_OperationNumControlInputs(oper)).  Returns the number of control
        /// inputs (should match TF_OperationNumControlInputs(oper)).
        /// </summary>
        /// <param name="oper">TF_Operation*</param>
        /// <param name="control_inputs">TF_Operation**</param>
        /// <param name="max_control_inputs"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationGetControlInputs(IntPtr oper, IntPtr control_inputs, int max_control_inputs);

        /// <summary>
        /// Get the list of operations that have `*oper` as a control input.
        /// `control_outputs` must point to an array of length at least
        /// `max_control_outputs` (ideally set to
        /// TF_OperationNumControlOutputs(oper)). Beware that a concurrent
        /// modification of the graph can increase the number of control
        /// outputs.  Returns the number of control outputs (should match
        /// TF_OperationNumControlOutputs(oper)).
        /// </summary>
        /// <param name="oper">TF_Operation*</param>
        /// <param name="control_outputs">TF_Operation**</param>
        /// <param name="max_control_outputs"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationGetControlOutputs(IntPtr oper, IntPtr control_outputs, int max_control_outputs);

        /// <summary>
        /// TF_Output producer = TF_OperationInput(consumer);
        /// There is an edge from producer.oper's output (given by
        /// producer.index) to consumer.oper's input (given by consumer.index).
        /// </summary>
        /// <param name="oper_in"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_Output TF_OperationInput(TF_Input oper_in);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationInputListLength(IntPtr oper, string arg_name, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern TF_DataType TF_OperationInputType(TF_Input oper_in);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_OperationName(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumInputs(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_OperationOpType(IntPtr oper);

        /// <summary>
        /// Get the number of control inputs to an operation.
        /// </summary>
        /// <param name="oper"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumControlInputs(IntPtr oper);

        /// <summary>
        /// Get the number of operations that have `*oper` as a control input.
        /// </summary>
        /// <param name="oper"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumControlOutputs(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumOutputs(IntPtr oper);

        /// <summary>
        /// Get the number of current consumers of a specific output of an
        /// operation.  Note that this number can change when new operations
        /// are added to the graph.
        /// </summary>
        /// <param name="oper_out"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationOutputNumConsumers(TF_Output oper_out);

        /// <summary>
        /// Get list of all current consumers of a specific output of an
        /// operation.  `consumers` must point to an array of length at least
        /// `max_consumers` (ideally set to
        /// TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
        /// modification of the graph can increase the number of consumers of
        /// an operation.  Returns the number of output consumers (should match
        /// TF_OperationOutputNumConsumers(oper_out)).
        /// </summary>
        /// <param name="oper_out">TF_Output</param>
        /// <param name="consumers">TF_Input*</param>
        /// <param name="max_consumers">int</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe int TF_OperationOutputConsumers(TF_Output oper_out, IntPtr consumers, int max_consumers);

        [DllImport(TensorFlowLibName)]
        public static extern TF_DataType TF_OperationOutputType(TF_Output oper_out);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationToNodeDef(IntPtr oper, IntPtr buffer, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationOutputListLength(IntPtr oper, string arg_name, IntPtr status);
    }
}
