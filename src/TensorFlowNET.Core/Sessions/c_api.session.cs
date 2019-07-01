using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Destroy a session object.
        ///
        /// Even if error information is recorded in *status, this call discards all
        /// local resources associated with the session.  The session may not be used
        /// during or after this call (and the session drops its reference to the
        /// corresponding graph). 
        /// </summary>
        /// <param name="session">TF_Session*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteSession(IntPtr session, IntPtr status);

        /// <summary>
        /// Destroy an options object.
        /// </summary>
        /// <param name="opts">TF_SessionOptions*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteSessionOptions(IntPtr opts);

        /// <summary>
        /// Return a new execution session with the associated graph, or NULL on
        /// error. Does not take ownership of any input parameters.
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="opts">const TF_SessionOptions*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>TF_Session*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewSession(IntPtr graph, IntPtr opts, IntPtr status);

        /// <summary>
        /// Return a new options object.
        /// </summary>
        /// <returns>TF_SessionOptions*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe IntPtr TF_NewSessionOptions();

        /// <summary>
        /// Run the graph associated with the session starting with the supplied inputs
        /// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
        ///
        /// Any NULL and non-NULL value combinations for (`run_options`,
        /// `run_metadata`) are valid.
        ///
        ///    - `run_options` may be NULL, in which case it will be ignored; or
        ///      non-NULL, in which case it must point to a `TF_Buffer` containing the
        ///      serialized representation of a `RunOptions` protocol buffer.
        ///    - `run_metadata` may be NULL, in which case it will be ignored; or
        ///      non-NULL, in which case it must point to an empty, freshly allocated
        ///      `TF_Buffer` that may be updated to contain the serialized representation
        ///      of a `RunMetadata` protocol buffer.
        ///
        /// The caller retains ownership of `input_values` (which can be deleted using
        /// TF_DeleteTensor). The caller also retains ownership of `run_options` and/or
        /// `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on
        /// them.
        ///
        /// On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
        /// output_values[]. Ownership of the elements of output_values[] is transferred
        /// to the caller, which must eventually call TF_DeleteTensor on them.
        ///
        /// On failure, output_values[] contains NULLs.
        /// </summary>
        /// <param name="session">TF_Session*</param>
        /// <param name="run_options">const TF_Buffer*</param>
        /// <param name="inputs">const TF_Output*</param>
        /// <param name="input_values">TF_Tensor* const*</param>
        /// <param name="ninputs">int</param>
        /// <param name="outputs">const TF_Output*</param>
        /// <param name="output_values">TF_Tensor**</param>
        /// <param name="noutputs">int</param>
        /// <param name="target_opers">const TF_Operation* const*</param>
        /// <param name="ntargets">int</param>
        /// <param name="run_metadata">TF_Buffer*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SessionRun(IntPtr session, TF_Buffer* run_options,
                   TF_Output[] inputs, IntPtr[] input_values, int ninputs,
                   TF_Output[] outputs, IntPtr[] output_values, int noutputs,
                   IntPtr[] target_opers, int ntargets,
                   IntPtr run_metadata,
                   IntPtr status);

        /// <summary>
        /// Set the config in TF_SessionOptions.options.
        /// config should be a serialized tensorflow.ConfigProto proto.
        /// If config was not parsed successfully as a ConfigProto, record the
        /// error information in *status.
        /// </summary>
        /// <param name="options">TF_SessionOptions*</param>
        /// <param name="proto">const void*</param>
        /// <param name="proto_len">size_t</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetConfig(IntPtr options, IntPtr proto, ulong proto_len, IntPtr status);
    }
}
