using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    internal class Project : List<IProjectFile>, IProject
    {
        public string Name { get; set; }

        public void WriteToDisk(string folder)
        {
            foreach ( var file in this)
            {
                file.WriteToDisk(Path.Combine(folder, file.Name));
            }
        }
    }
}
