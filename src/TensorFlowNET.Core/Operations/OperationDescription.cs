using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class OperationDescription
    {
        private IntPtr _handle;
        public IntPtr op => _handle;

        public OperationDescription(Graph graph, string opType, string opName)
        {
            _handle = c_api.TF_NewOperation(graph, opType, opName);
        }

        public OperationDescription(IntPtr handle)
        {
            _handle = handle;
        }

        public void AddInputList(params TF_Output[] inputs)
        {
            c_api.TF_AddInputList(_handle, inputs, inputs.Length);
        }

        public void SetAttrType(string attr_name, TF_DataType value)
        {
            c_api.TF_SetAttrType(_handle, attr_name, value);
        }

        public void SetAttrShape(string attr_name, long[] dims)
        {
            c_api.TF_SetAttrShape(_handle, attr_name, dims, dims.Length);
        }

        public Operation FinishOperation(Status status)
        {
            return c_api.TF_FinishOperation(_handle, status);
        }

        public static implicit operator OperationDescription(IntPtr handle)
        {
            return new OperationDescription(handle);
        }

        public static implicit operator IntPtr(OperationDescription desc)
        {
            return desc._handle;
        }
    }
}
