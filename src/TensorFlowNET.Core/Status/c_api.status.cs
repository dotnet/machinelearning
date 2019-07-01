using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Delete a previously created status object.
        /// </summary>
        /// <param name="s"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteStatus(IntPtr s);

        /// <summary>
        /// Return the code record in *s.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_Code TF_GetCode(IntPtr s);

        /// <summary>
        /// Return a pointer to the (null-terminated) error message in *s.
        /// The return value points to memory that is only usable until the next
        /// mutation to *s.  Always returns an empty string if TF_GetCode(s) is TF_OK.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_Message(IntPtr s);

        /// <summary>
        /// Return a new status object.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_NewStatus();

        /// <summary>
        /// Record <code, msg> in *s.  Any previous information is lost.
        /// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
        /// </summary>
        /// <param name="s"></param>
        /// <param name="code"></param>
        /// <param name="msg"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetStatus(IntPtr s, TF_Code code, string msg);
    }
}
