using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.Interface
{
    internal interface ICSharpProjectGenerator
    {
        ICSharpProject ToProject();
    }
}
