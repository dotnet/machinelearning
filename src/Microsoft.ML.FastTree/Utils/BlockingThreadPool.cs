// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// This class wraps the standard .NET ThreadPool and adds the following functionality:
    /// 1) the user explicitly defines a maximum of concurrently running threads
    /// 2) if the maximum is k, and k work items are already running, a call to RunOrBlock will block until a
    ///    thread is available
    /// 3) a work item can be any function with 6 or less arguments
    /// 4) a work item knows the index of the thread it is running on - this can be used if the threads share \
    ///    common resources
    /// </summary>
    public static class BlockingThreadPool
    {
        private static int _numThreads;

        public static int NumThreads
        {
            get { return _numThreads; }
        }

        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="numThreads">the maximal number of concurrent threads</param>
        public static void Initialize(int numThreads)
        {
            if (_numThreads == numThreads)
                return;

            _numThreads = numThreads;
        }
    }
}
