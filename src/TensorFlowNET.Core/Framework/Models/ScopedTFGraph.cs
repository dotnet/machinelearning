using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework.Models
{
    public class ScopedTFGraph : Graph
    {
        public ScopedTFGraph() : base()
        {

        }

        ~ScopedTFGraph()
        {
            base.Dispose();
        }
    }
}
