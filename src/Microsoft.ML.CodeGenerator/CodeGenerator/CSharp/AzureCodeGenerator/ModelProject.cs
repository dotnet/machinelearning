using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.Templates.Console
{
    internal partial class ModelProject : IProjectFile
    {
        public IProjectFile ToProjectFile()
        {
            return new ProjectFile()
            {
                Data = TransformText(),
            };
        }

        public void WriteToDisk(string path)
        {
            throw new NotImplementedException();
        }
    }
}
