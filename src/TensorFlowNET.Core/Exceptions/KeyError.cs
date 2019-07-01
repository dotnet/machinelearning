using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class KeyError : Exception
    {
        public KeyError() : base()
        {

        }

        public KeyError(string message) : base(message)
        {

        }
    }
}
