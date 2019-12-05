using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CodeGenerator.CodeGenerator;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.Templates.Console
{
    internal partial class ConsumeModel : IProjectFileGenerator
    {
        public IProjectFile ToProjectFile()
        {
            return new ProjectFile()
            {
                Data = Utilities.Utils.FormatCode(TransformText()),
                Name = "ConsumeModel.cs",
            };
        }
    }
}
