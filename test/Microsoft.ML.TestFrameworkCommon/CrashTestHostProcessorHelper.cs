using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class CrashTestHostProcessorHelper
    {
        public static void CrashTestHostProcess()
        {
            Environment.FailFast("Crash on purpose here to take dump!");
        }
    }
}
