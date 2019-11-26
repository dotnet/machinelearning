using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    public interface IWritable
    {
        void WriteToDisk(string path);
    }
}
