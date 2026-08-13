using System;
using System.Diagnostics;

namespace Microsoft.ML.TorchSharp.Tests.QATests
{
    public class ResourceLeakDetector
    {
        public bool CheckForResourceLeaks()
        {
            // Check for resource leaks
            if (GC.GetTotalMemory(true) > 100 * 1024 * 1024)
            {
                return true;
            }
            return false;
        }
    }
}