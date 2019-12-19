using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CodeGenerator.CodeGenerator;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.Templates.Console
{
    internal partial class ModelProject : IProjectFileGenerator
    {
        public string OutputName { get; set; }

        public ICSharpFile ToProjectFile()
        {
            return new CSharpCodeFile()
            {
                File = TransformText(),
                Name = $"{OutputName}.Model.csproj",
            };
        }
    }
}
