using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class AggregationMethod
    {
        public static int ADD_N = 0;
        public static int DEFAULT = ADD_N;
        // The following are experimental and may not be supported in future releases.
        public static int EXPERIMENTAL_TREE = 1;
        public static int EXPERIMENTAL_ACCUMULATE_N = 2;
    }
}
