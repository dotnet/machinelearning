// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

#if NET6_0_OR_GREATER
using Executor = Microsoft.DotNet.RemoteExecutor.RemoteExecutor;
#else
using Executor = Microsoft.ML.TestFramework.RemoteExecutor;
#endif

namespace Microsoft.ML.CpuMath.UnitTests
{

    internal static class RemoteExecutor
    {
        public const int SuccessExitCode = 42;

        public static void RemoteInvoke(
            Func<string, string, string, string, int> method,
            string arg1, string arg2, string arg3, string arg4,
#if NETFRAMEWORK
            Microsoft.ML.TestFramework.RemoteInvokeOptions options = null)
#else
            Microsoft.DotNet.RemoteExecutor.RemoteInvokeOptions options = null)
#endif
        {
#if NETFRAMEWORK
            Executor.RemoteInvoke(method, arg1, arg2, arg3, arg4, options);
#else
            Executor.Invoke(method, arg1, arg2, arg3, arg4, options).Dispose();
#endif
        }

        public static void RemoteInvoke(
            Func<string, string, string, int> method,
            string arg1, string arg2, string arg3,
#if NETFRAMEWORK
            Microsoft.ML.TestFramework.RemoteInvokeOptions options = null)
#else
            Microsoft.DotNet.RemoteExecutor.RemoteInvokeOptions options = null)
#endif
        {
#if NETFRAMEWORK
            Executor.RemoteInvoke(method, arg1, arg2, arg3, options);
#else
            Executor.Invoke(method, arg1, arg2, arg3, options).Dispose();
#endif
        }

        public static void RemoteInvoke(
            Func<string, string, int> method,
            string arg1, string arg2,
#if NETFRAMEWORK
            Microsoft.ML.TestFramework.RemoteInvokeOptions options = null)
#else
            Microsoft.DotNet.RemoteExecutor.RemoteInvokeOptions options = null)
#endif
        {
#if NETFRAMEWORK
            Executor.RemoteInvoke(method, arg1, arg2, options);
#else
            Executor.Invoke(method, arg1, arg2, options).Dispose();
#endif
        }
    }
}
