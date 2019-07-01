using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class random_seed
    {
        private static int DEFAULT_GRAPH_SEED = 87654321;

        public static (int?, int?) get_seed(int? op_seed = null)
        {
            if (op_seed.HasValue)
                return (DEFAULT_GRAPH_SEED, 0);
            else
                return (null, null);
        }
    }
}
