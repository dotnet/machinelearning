using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Data
{
    /*
     * This class is should be used as the base type for all Attributes which define a Type.
     * The DataViewManager relies upon this type to determine if the Attribute is relevant.
     */
    public abstract class TypeAttribute : Attribute
    {
    }
}
