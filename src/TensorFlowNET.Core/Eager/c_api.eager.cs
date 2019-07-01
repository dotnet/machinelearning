using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Return a new options object.
        /// </summary>
        /// <returns>TFE_ContextOptions*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewContextOptions();

        /// <summary>
        /// Destroy an options object.
        /// </summary>
        /// <param name="options">TFE_ContextOptions*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteContextOptions(IntPtr options);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="opts">const TFE_ContextOptions*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>TFE_Context*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewContext(IntPtr opts, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteContext(IntPtr ctx);

        /// <summary>
        /// Execute the operation defined by 'op' and return handles to computed
        /// tensors in `retvals`.
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="retvals">TFE_TensorHandle**</param>
        /// <param name="num_retvals">int*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_Execute(IntPtr op, IntPtr retvals, int[] num_retvals, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        /// <param name="op_or_function_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewOp(IntPtr ctx, string op_or_function_name, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteOp(IntPtr op);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">TF_DataType</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrType(IntPtr op, string attr_name, TF_DataType value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="dims">const int64_t*</param>
        /// <param name="num_dims">const int</param>
        /// <param name="out_status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrShape(IntPtr op, string attr_name, long[] dims, int num_dims, Status out_status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">const void*</param>
        /// <param name="length">size_t</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrString(IntPtr op, string attr_name, string value, uint length);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpAddInput(IntPtr op, IntPtr h, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t">const tensorflow::Tensor&</param>
        /// <returns>TFE_TensorHandle*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewTensorHandle(IntPtr t);
    }
}
