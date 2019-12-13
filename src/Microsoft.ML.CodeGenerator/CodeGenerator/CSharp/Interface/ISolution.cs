using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    internal interface ISolution : IWritable, IEnumerable<IProject>
    {
        public string Name { get; set; }
    }
}
