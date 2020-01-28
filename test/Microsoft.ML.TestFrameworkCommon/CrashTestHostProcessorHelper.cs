using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class CrashTestHostProcessorHelper
    {
        public static void CrashTestHostProcess()
        {
            ThreadPool.QueueUserWorkItem(new WaitCallback(ignored =>
            {
                throw new Exception();
            }));
        }
    }
}
