using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

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
