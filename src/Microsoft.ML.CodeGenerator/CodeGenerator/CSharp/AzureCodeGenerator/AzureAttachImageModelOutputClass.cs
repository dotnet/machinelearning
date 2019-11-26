using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.Templates.AzureImageClassification.Model
{
    internal partial class AzureAttachImageModelOutputClass : IProjectFile
    {
        public IProjectFile ToProjectFile()
        {
            return new ProjectFile()
            {
                Data = Utilities.Utils.FormatCode(TransformText()),
            };
        }

        public void WriteToDisk(string path)
        {
            throw new NotImplementedException();
        }
    }
}
