using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Mode for variable access within a variable scope.
    /// </summary>
    public enum _ReuseMode
    {
        // Indicates that variables are to be fetched if they already exist or
        // otherwise created.
        AUTO_REUSE = 1
    }
}
