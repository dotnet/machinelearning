using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class Context : IDisposable
    {
        private IntPtr _handle;

        public static int GRAPH_MODE = 0;
        public static int EAGER_MODE = 1;

        public int default_execution_mode;

        public Context(ContextOptions opts, Status status)
        {
            _handle = c_api.TFE_NewContext(opts, status);
            status.Check(true);
        }

        public void Dispose()
        {
            c_api.TFE_DeleteContext(_handle);
        }

        public bool executing_eagerly()
        {
            return false;
        }

        public static implicit operator IntPtr(Context ctx)
        {
            return ctx._handle;
        }
    }
}
