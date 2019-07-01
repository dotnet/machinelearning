using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework.Models
{
    public class ScopedTFImportGraphDefResults : ImportGraphDefOptions
    {
        public ScopedTFImportGraphDefResults() : base()
        {
            
        }

        public ScopedTFImportGraphDefResults(IntPtr results) : base(results)
        {

        }

        ~ScopedTFImportGraphDefResults()
        {
            base.Dispose();
        }
    }
}
