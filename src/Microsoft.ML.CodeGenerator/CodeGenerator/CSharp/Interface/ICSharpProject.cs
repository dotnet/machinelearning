using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    internal interface ICSharpProject : IWritable, IEnumerable<ICSharpFile>
    {
        public string Name { get; set; }
    }
}
