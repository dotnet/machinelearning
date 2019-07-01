using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Convert to other datatype implicitly
    /// </summary>
    public partial class Operation
    {
        public static implicit operator Operation(IntPtr handle) => new Operation(handle);
        public static implicit operator IntPtr(Operation op) => op._handle;
        public static implicit operator Tensor(Operation op) => op.output;

        public override string ToString()
        {
            return _handle == IntPtr.Zero ? "tf.Operation Undefined" : $"tf.Operation '{name}' type={OpType}";
        }

        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case IntPtr val:
                    return val == _handle;
                case Operation val:
                    return val._handle == _handle;
            }

            return base.Equals(obj);
        }
    }
}
