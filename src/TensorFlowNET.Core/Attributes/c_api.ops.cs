using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Fills in `value` with the value of the attribute `attr_name`.  `value` must
        /// point to an array of length at least `max_length` (ideally set to
        /// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
        /// attr_name)).
        /// </summary>
        /// <param name="oper">TF_Operation*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_AttrMetadata TF_OperationGetAttrMetadata(IntPtr oper, string attr_name, IntPtr status);

        /// <summary>
        /// Fills in `value` with the value of the attribute `attr_name`.  `value` must
        /// point to an array of length at least `max_length` (ideally set to
        /// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
        /// attr_name)). 
        /// </summary>
        /// <param name="oper">TF_Operation*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">void* </param>
        /// <param name="max_length">size_t</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationGetAttrString(IntPtr oper, string attr_name, IntPtr value, uint max_length, IntPtr status);
        
        /// <summary>
        /// Sets `output_attr_value` to the binary-serialized AttrValue proto
        /// representation of the value of the `attr_name` attr of `oper`.
        /// </summary>
        /// <param name="oper"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationGetAttrValueProto(IntPtr oper, string attr_name, IntPtr output_attr_value, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrBool(IntPtr desc, string attr_name, bool value);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrValueProto(IntPtr desc, string attr_name, IntPtr proto, uint proto_len, IntPtr status);

        /// <summary>
        /// Set `num_dims` to -1 to represent "unknown rank".
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="attr_name"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrShape(IntPtr desc, string attr_name, long[] dims, int num_dims);

        /// <summary>
        /// Call some TF_SetAttr*() function for every attr that is not
        /// inferred from an input and doesn't have a default value you wish to
        /// keep.
        /// 
        /// `value` must point to a string of length `length` bytes.
        /// </summary>
        /// <param name="desc">TF_OperationDescription*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">const void*</param>
        /// <param name="length">size_t</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrString(IntPtr desc, string attr_name, string value, uint length);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="attr_name"></param>
        /// <param name="values"></param>
        /// <param name="lengths"></param>
        /// <param name="num_values"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrStringList(IntPtr desc, string attr_name, IntPtr[] values, uint[] lengths, int num_values);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrTensor(IntPtr desc, string attr_name, IntPtr value, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrType(IntPtr desc, string attr_name, TF_DataType value);
    }
}
