using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Samples
{
    public abstract class SampleBase
    {
        public static string GetAbsolutePath<T>(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(T).Assembly.Location);
            string fullPath = Path.Combine(_dataRoot.Directory.FullName, relativePath);

            return fullPath;
        }
    }
}
