using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework.Models
{
    public class ScopedTFImportGraphDefOptions : ImportGraphDefOptions
    {
        public ScopedTFImportGraphDefOptions() : base()
        {

        }

        ~ScopedTFImportGraphDefOptions()
        {
            base.Dispose();
        }
    }
}
