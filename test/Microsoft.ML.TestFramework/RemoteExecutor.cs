// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Xunit;
using Xunit.Sdk;

namespace Microsoft.ML.TestFramework
{
    /// <summary>
    /// Base class used for all tests that need to spawn a remote process.
    /// Most of the code has been taken from RemoteExecutorTestBase class in the corefx repo.
    /// </summary>
    public static class RemoteExecutor
    {
        /// <summary>The name of the test console app.</summary>
        public static readonly string TestConsoleApp = Path.GetFullPath(@"RemoteExecutorConsoleApp.dll");
#if NETFRAMEWORK
        public static readonly string HostRunner = Path.GetFullPath(@"RemoteExecutorConsoleApp.exe");
        private static readonly string _extraParameter = "";
#else
        public static readonly string HostRunner = Process.GetCurrentProcess().MainModule.FileName;
        private static readonly string _extraParameter = TestConsoleApp;
#endif
        /// <summary>A timeout (milliseconds) after which a wait on a remote operation should be considered a failure.</summary>
        public const int FailWaitTimeoutMilliseconds = 60 * 1000;

        /// <summary>The exit code returned when the test process exits successfully.</summary>
        public const int SuccessExitCode = 42;

        /// <summary>Invokes the method from this assembly in another process using the specified arguments.</summary>
        /// <param name="method">The method to invoke.</param>
        /// <param name="arg1">The first argument to pass to the method.</param>
        /// <param name="arg2">The second argument to pass to the method.</param>
        /// <param name="options">Options to use for the invocation.</param>
        public static void RemoteInvoke(
            Func<string, string, int> method,
            string arg1, string arg2,
            RemoteInvokeOptions options = null)
        {
            RemoteInvoke(GetMethodInfo(method), new[] { arg1, arg2 }, options);
        }

        /// <summary>Invokes the method from this assembly in another process using the specified arguments.</summary>
        /// <param name="method">The method to invoke.</param>
        /// <param name="arg1">The first argument to pass to the method.</param>
        /// <param name="arg2">The second argument to pass to the method.</param>
        /// <param name="arg3">The third argument to pass to the method.</param>
        /// <param name="options">Options to use for the invocation.</param>
        public static void RemoteInvoke(
            Func<string, string, string, int> method,
            string arg1, string arg2, string arg3,
            RemoteInvokeOptions options = null)
        {
            RemoteInvoke(GetMethodInfo(method), new[] { arg1, arg2, arg3 }, options);
        }

        /// <summary>Invokes the method from this assembly in another process using the specified arguments.</summary>
        /// <param name="method">The method to invoke.</param>
        /// <param name="arg1">The first argument to pass to the method.</param>
        /// <param name="arg2">The second argument to pass to the method.</param>
        /// <param name="arg3">The third argument to pass to the method.</param>
        /// <param name="arg4">The fourth argument to pass to the method.</param>
        /// <param name="options">Options to use for the invocation.</param>
        public static void RemoteInvoke(
            Func<string, string, string, string, int> method,
            string arg1, string arg2, string arg3, string arg4,
            RemoteInvokeOptions options = null)
        {
            RemoteInvoke(GetMethodInfo(method), new[] { arg1, arg2, arg3, arg4 }, options);
        }

        /// <summary>Invokes the method from this assembly in another process using the specified arguments.</summary>
        /// <param name="method">The method to invoke.</param>
        /// <param name="args">The arguments to pass to the method.</param>
        /// <param name="options">Options to use for the invocation.</param>
        /// <param name="pasteArguments">true if this function should paste the arguments (e.g. surrounding with quotes); false if that responsibility is left up to the caller.</param>
        private static void RemoteInvoke(MethodInfo method, string[] args, RemoteInvokeOptions options, bool pasteArguments = true)
        {
            options = options ?? new RemoteInvokeOptions();

            // Verify the specified method returns an int (the exit code) or nothing,
            // and that if it accepts any arguments, they're all strings.
            Assert.True(method.ReturnType == typeof(void) || method.ReturnType == typeof(int) || method.ReturnType == typeof(Task<int>));
            Assert.All(method.GetParameters(), pi => Assert.Equal(typeof(string), pi.ParameterType));

            // And make sure it's in this assembly.  This isn't critical, but it helps with deployment to know
            // that the method to invoke is available because we're already running in this assembly.
            Type t = method.DeclaringType;
            Assembly a = t.GetTypeInfo().Assembly;

            // Start the other process and return a wrapper for it to handle its lifetime and exit checking.
            ProcessStartInfo psi = options.StartInfo;
            psi.UseShellExecute = false;

            // If we need the host (if it exists), use it, otherwise target the console app directly.
            string metadataArgs = PasteArguments.Paste(new string[] { a.FullName, t.FullName, method.Name, options.ExceptionFile }, pasteFirstArgumentUsingArgV0Rules: false);
            string passedArgs = pasteArguments ? PasteArguments.Paste(args, pasteFirstArgumentUsingArgV0Rules: false) : string.Join(" ", args);
            string testConsoleAppArgs = _extraParameter + " " + metadataArgs + " " + passedArgs;

            psi.FileName = HostRunner;
            psi.Arguments = testConsoleAppArgs;

            // Return the handle to the process, which may or not be started
            CheckProcess(Process.Start(psi), options);
        }

        private static void CheckProcess(Process process, RemoteInvokeOptions options)
        {
            if (process != null)
            {
                // A bit unorthodox to do throwing operations in a Dispose, but by doing it here we avoid
                // needing to do this in every derived test and keep each test much simpler.
                try
                {
                    Assert.True(process.WaitForExit(options.TimeOut),
                        $"Timed out after {options.TimeOut}ms waiting for remote process {process.Id}");

                    if (File.Exists(options.ExceptionFile))
                    {
                        throw new RemoteExecutionException(File.ReadAllText(options.ExceptionFile));
                    }

                    if (options.CheckExitCode)
                    {
                        int expected = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? options.ExpectedExitCode : unchecked((sbyte)options.ExpectedExitCode);
                        int actual = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? process.ExitCode : unchecked((sbyte)process.ExitCode);

                        Assert.True(expected == actual, $"Exit code was {process.ExitCode} but it should have been {options.ExpectedExitCode}");
                    }
                }
                finally
                {
                    if (File.Exists(options.ExceptionFile))
                    {
                        File.Delete(options.ExceptionFile);
                    }

                    // Cleanup
                    try { process.Kill(); }
                    catch { } // ignore all cleanup errors

                    process.Dispose();
                    process = null;
                }
            }
        }

        private sealed class RemoteExecutionException : XunitException
        {
            internal RemoteExecutionException(string stackTrace) : base("Remote process failed with an unhandled exception.", stackTrace) { }
        }

        private static MethodInfo GetMethodInfo(Delegate d)
        {
            // RemoteInvoke doesn't support marshaling state on classes associated with
            // the delegate supplied (often a display class of a lambda).  If such fields
            // are used, odd errors result, e.g. NullReferenceExceptions during the remote
            // execution.  Try to ward off the common cases by proactively failing early
            // if it looks like such fields are needed.
            if (d.Target != null)
            {
                // The only fields on the type should be compiler-defined (any fields of the compiler's own
                // making generally include '<' and '>', as those are invalid in C# source).  Note that this logic
                // may need to be revised in the future as the compiler changes, as this relies on the specifics of
                // actually how the compiler handles lifted fields for lambdas.
                Type targetType = d.Target.GetType();
                Assert.All(
                    targetType.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly),
                    fi => Assert.True(fi.Name.IndexOf('<') != -1, $"Field marshaling is not supported by {nameof(RemoteInvoke)}: {fi.Name}"));
            }

            return d.GetMethodInfo();
        }
    }

    /// <summary>Options used with RemoteInvoke.</summary>
    public sealed class RemoteInvokeOptions
    {
        public RemoteInvokeOptions(Dictionary<string, string> environmentVariables = null)
        {
            if (environmentVariables != null)
            {
                foreach (var item in environmentVariables)
                {
                    StartInfo.EnvironmentVariables.Add(item.Key, item.Value);
                }
            }
        }

        public ProcessStartInfo StartInfo { get; set; } = new ProcessStartInfo();
        public bool CheckExitCode { get; set; } = true;
        public int TimeOut { get; set; } = RemoteExecutor.FailWaitTimeoutMilliseconds;
        public int ExpectedExitCode { get; set; } = RemoteExecutor.SuccessExitCode;
        public string ExceptionFile { get; } = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
    }
}
