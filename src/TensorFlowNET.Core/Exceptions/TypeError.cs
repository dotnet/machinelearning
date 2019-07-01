using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class TypeError : Exception
    {
        public TypeError() : base()
        {

        }

        public TypeError(string message) : base(message)
        {

        }
    }
}
