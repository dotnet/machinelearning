using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.IO;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Summaries.Summary summary = new Summaries.Summary();

        public static Tensor scalar(string name, Tensor tensor) 
            => summary.scalar(name, tensor);
    }
}
