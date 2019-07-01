using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class distributions
        {
            public static Normal Normal(Tensor loc,
                Tensor scale,
                bool validate_args = false,
                bool allow_nan_stats = true,
                string name = "Normal") => new Normal(loc, scale, validate_args = false, allow_nan_stats = true, "Normal");
        }
    }
}
