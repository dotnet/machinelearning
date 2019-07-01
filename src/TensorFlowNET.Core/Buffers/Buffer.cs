using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class Buffer : IDisposable
    {
        private IntPtr _handle;

        private TF_Buffer buffer => Marshal.PtrToStructure<TF_Buffer>(_handle);

        public byte[] Data 
        {
            get 
            {
                var data = new byte[buffer.length];
                if (buffer.length > 0)
                    Marshal.Copy(buffer.data, data, 0, (int)buffer.length);
                return data;
            }
        }

        public int Length => (int)buffer.length;

        public Buffer()
        {
            _handle = c_api.TF_NewBuffer();
        }

        public Buffer(IntPtr handle)
        {
            _handle = handle;
        }

        public Buffer(byte[] data)
        {
            var dst = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, dst, data.Length);

            _handle = c_api.TF_NewBufferFromString(dst, (ulong)data.Length);
        }

        public static implicit operator IntPtr(Buffer buffer)
        {
            return buffer._handle;
        }

        public static implicit operator byte[](Buffer buffer)
        {
            return buffer.Data;
        }

        public void Dispose()
        {
            c_api.TF_DeleteBuffer(_handle);
        }
    }
}
