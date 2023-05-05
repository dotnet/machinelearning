// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Reflection;
using System.Threading;
using Microsoft.ML.Internal.Internallearn.Test;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.TestFramework
{
    public class BaseTestClass : IDisposable
    {
        public string TestName { get; set; }
        public string FullTestName { get; set; }

        public ChannelMessageKind MessageKindToLog;

        static BaseTestClass()
        {
            // specific to use tls 1.2 as https://aka.ms/mlnet-resources/ only accpets tls 1.2 or newer
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

            AppDomain.CurrentDomain.UnhandledException += (sender, e) =>
            {
                // Write to stdout because stderr does not show up in the test output
                Console.WriteLine($"Unhandled exception: {e.ExceptionObject}");
            };

            GlobalBase.AssemblyInit();
            RootDir = TestCommon.GetRepoRoot();
            DataDir = Path.Combine(RootDir, "test", "data");
        }

        public BaseTestClass(ITestOutputHelper output)
        {
            //This locale is currently set for tests only so that the produced output
            //files can be compared on systems with other locales to give set of known
            //correct results that are on en-US locale.
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");

#if NETFRAMEWORK
            string codeBaseUri = typeof(BaseTestClass).Assembly.CodeBase;
            string path = new Uri(codeBaseUri).AbsolutePath;
            var currentAssemblyLocation = new FileInfo(Directory.GetParent(path).FullName);
#else
            // There is an extra folder in the netfx path representing the runtime identifier.
            var currentAssemblyLocation = new FileInfo(typeof(BaseTestClass).Assembly.Location);
#endif
            OutDir = Path.Combine(currentAssemblyLocation.Directory.FullName, "TestOutput");
            Directory.CreateDirectory(OutDir);
            Output = output;

            ITest test = (ITest)output.GetType().GetField("test", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(output);
            FullTestName = test.TestCase.TestMethod.TestClass.Class.Name + "." + test.TestCase.TestMethod.Method.Name;
            TestName = test.TestCase.TestMethod.Method.Name;

            MessageKindToLog = ChannelMessageKind.Error;
            var attributes = test.TestCase.TestMethod.Method.GetCustomAttributes(typeof(LogMessageKind));
            foreach (var attrib in attributes)
            {
                MessageKindToLog = attrib.GetNamedArgument<ChannelMessageKind>("MessageKind");
            }

            // write to the console when a test starts and stops so we can identify any test hangs/deadlocks in CI
            Console.WriteLine($"Starting test: {FullTestName}");
            Initialize();
        }

        void IDisposable.Dispose()
        {
            Cleanup();
            Process proc = Process.GetCurrentProcess();
            Console.WriteLine($"Finished test: {FullTestName} " +
                $"with memory usage {proc.WorkingSet64.ToString("N", CultureInfo.InvariantCulture)}");
        }

        protected virtual void Initialize()
        {
        }

        protected virtual void Cleanup()
        {
        }

        protected static string RootDir { get; }
        protected string OutDir { get; }
        protected static string DataDir { get; }

        protected ITestOutputHelper Output { get; }

        public static string GetDataPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(DataDir, name));
        }
        public static string GetDataPath(string subDir, string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(DataDir, subDir, name));
        }

        protected string GetOutputPath(string name)
        {
            return TestCommon.GetOutputPath(OutDir, name);
        }
        protected string GetOutputPath(string subDir, string name)
        {
            return TestCommon.GetOutputPath(OutDir, subDir, name);
        }
        protected string DeleteOutputPath(string subDir, string name)
        {
            return TestCommon.DeleteOutputPath(OutDir, subDir, name);
        }
        protected string DeleteOutputPath(string name)
        {
            return TestCommon.DeleteOutputPath(OutDir, name);
        }
    }
}
