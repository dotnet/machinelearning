using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CodeGenerator.CodeGenerator;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.Templates.Console
{
    internal partial class ModelInputClass : IProjectFileGenerator
    {
        public ICSharpFile ToProjectFile()
        {
            return new CSharpCodeFile()
            {
                File = Utilities.Utils.FormatCode(TransformText()),
                Name = "ModelInput.cs",
            };
        }
    }
}
