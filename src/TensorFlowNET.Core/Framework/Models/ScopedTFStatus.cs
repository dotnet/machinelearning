using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework.Models
{
    public class ScopedTFStatus : Status
    {
        public ScopedTFStatus() : base()
        {
        }

        ~ScopedTFStatus()
        {
            base.Dispose();
        }
    }
}
