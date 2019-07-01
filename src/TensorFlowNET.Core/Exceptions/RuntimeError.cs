using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class RuntimeError : Exception
    {
        public RuntimeError() : base()
        {

        }

        public RuntimeError(string message) : base(message)
        {

        }
    }
}
