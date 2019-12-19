using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    internal interface IProjectFileGenerator
    {
        ICSharpFile ToProjectFile();
    }
}
