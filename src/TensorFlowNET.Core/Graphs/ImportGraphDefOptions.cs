using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class ImportGraphDefOptions : IDisposable
    {
        private IntPtr _handle;

        public int NumReturnOutputs => c_api.TF_ImportGraphDefOptionsNumReturnOutputs(_handle);

        public ImportGraphDefOptions()
        {
            _handle = c_api.TF_NewImportGraphDefOptions();
        }

        public ImportGraphDefOptions(IntPtr handle)
        {
            _handle = handle;
        }

        public void AddReturnOutput(string name, int index)
        {
            c_api.TF_ImportGraphDefOptionsAddReturnOutput(_handle, name, index);
        }

        public void Dispose()
        {
            c_api.TF_DeleteImportGraphDefOptions(_handle);
        }

        public static implicit operator IntPtr(ImportGraphDefOptions opts) => opts._handle;
        public static implicit operator ImportGraphDefOptions(IntPtr handle) => new ImportGraphDefOptions(handle);
    }
}
