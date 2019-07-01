using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// TF_Status holds error information. It either has an OK code, or
    /// else an error code with an associated error message.
    /// </summary>
    public class Status : IDisposable
    {
        protected readonly IntPtr _handle;

        /// <summary>
        /// Error message
        /// </summary>
        public string Message => c_api.StringPiece(c_api.TF_Message(_handle));

        /// <summary>
        /// Error code
        /// </summary>
        public TF_Code Code => c_api.TF_GetCode(_handle);

        public Status()
        {
            _handle = c_api.TF_NewStatus();
        }

        public void SetStatus(TF_Code code, string msg)
        {
            c_api.TF_SetStatus(_handle, code, msg);
        }

        /// <summary>
        /// Check status 
        /// Throw exception with error message if code != TF_OK
        /// </summary>
        public void Check(bool throwException = false)
        {
            if(Code != TF_Code.TF_OK)
            {
                Console.WriteLine(Message);
                if (throwException)
                {
                    throw new Exception(Message);
                }
            }
        }

        public static implicit operator IntPtr(Status status)
        {
            return status._handle;
        }

        public void Dispose()
        {
            c_api.TF_DeleteStatus(_handle);
        }
    }
}
