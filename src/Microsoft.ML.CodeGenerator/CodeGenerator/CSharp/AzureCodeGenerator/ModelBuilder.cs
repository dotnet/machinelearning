using System.Collections.Generic;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.Console;

namespace Microsoft.ML.CodeGenerator.Templates.Console
{
    internal partial class ModelBuilder : IProjectFile
    {
        public ModelBuilder()
        {
            Kfolds = 5;
            HasOnnxModel = false;
        }

        public IProjectFile ToProjectFile()
        {
            return new ProjectFile()
            {
                Data = Utilities.Utils.FormatCode(TransformText()),
            };
        }

        public void WriteToDisk(string location)
        {
            throw new System.NotImplementedException();
        }
    }
}