using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class SessionOptions : IDisposable
    {
        private IntPtr _handle;
        private Status _status;

        public unsafe SessionOptions()
        {
            var opts = c_api.TF_NewSessionOptions();
            _handle = opts;
            _status = new Status();
        }

        public unsafe SessionOptions(IntPtr handle)
        {
            _handle = handle;
        }

        public void Dispose()
        {
            c_api.TF_DeleteSessionOptions(_handle);
            _status.Dispose();
        }

        public Status SetConfig(ConfigProto config)
        {
            var bytes = config.ToByteArray();
            var proto = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, proto, bytes.Length);
            c_api.TF_SetConfig(_handle, proto, (ulong)bytes.Length, _status);
            _status.Check(false);
            return _status;
        }

        public static implicit operator IntPtr(SessionOptions opts) => opts._handle;
        public static implicit operator SessionOptions(IntPtr handle) => new SessionOptions(handle);
    }
}
