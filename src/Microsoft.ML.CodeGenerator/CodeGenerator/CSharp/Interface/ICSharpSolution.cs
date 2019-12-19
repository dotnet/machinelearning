using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    internal interface ICSharpSolution : IWritable, IEnumerable<ICSharpProject>
    {
        public string Name { get; set; }
    }
}
