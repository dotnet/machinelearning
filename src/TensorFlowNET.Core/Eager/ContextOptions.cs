using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class ContextOptions : IDisposable
    {
        private IntPtr _handle;

        public ContextOptions()
        {
            _handle = c_api.TFE_NewContextOptions();
        }

        public void Dispose()
        {
            c_api.TFE_DeleteContextOptions(_handle);
        }

        public static implicit operator IntPtr(ContextOptions opts)
        {
            return opts._handle;
        }
    }
}
