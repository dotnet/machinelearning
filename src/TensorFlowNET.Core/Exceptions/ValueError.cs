using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class ValueError : Exception
    {
        public ValueError() : base()
        {

        }

        public ValueError(string message) : base(message)
        {

        }
    }
}
