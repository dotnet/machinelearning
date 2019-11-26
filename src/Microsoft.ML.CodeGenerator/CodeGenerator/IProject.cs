using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    internal interface IProject : IWritable, IEnumerable<IProjectFile>
    {
        public string Name { get; set; }
    }
}
