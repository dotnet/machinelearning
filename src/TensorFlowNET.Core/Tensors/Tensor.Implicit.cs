using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        /// <summary>
        /// Issue unresolved, will cause name_scope problem.
        /// </summary>
        /// <param name="scalar"></param>
        /*public static implicit operator Tensor(double scalar)
        {
            return constant_op.constant(scalar);
        }*/

        /*public static implicit operator Tensor(int scalar)
        {
            return constant_op.constant(scalar);
        }*/

        public static implicit operator int(Tensor tensor)
        {
            return tensor.Data<int>()[0];
        }

        public static implicit operator IntPtr(Tensor tensor)
        {
            if (tensor._handle == IntPtr.Zero)
                Console.WriteLine("tensor is not allocated.");
            return tensor._handle;
        }

        public static implicit operator Operation(Tensor tensor)
        {
            return tensor.op;
        }

        public static implicit operator Tensor(IntPtr handle)
        {
            return new Tensor(handle);
        }
    }
}
