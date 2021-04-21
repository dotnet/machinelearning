// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit.Abstractions;

namespace Microsoft.ML.IntegrationTests
{
    public class IntegrationTestBaseClass : IDisposable
    {
        static IntegrationTestBaseClass()
        {
            RootDir = TestCommon.GetRepoRoot();
            DataDir = Path.Combine(RootDir, "test", "data");
        }

        public string TestName { get; set; }
        public string FullTestName { get; set; }
        public string OutDir { get; }
        public ChannelMessageKind MessageKindToLog;

        protected static string RootDir { get; }
        protected static string DataDir { get; }
        protected ITestOutputHelper Output { get; }

        public IntegrationTestBaseClass(ITestOutputHelper output)
        {
            //This locale is currently set for tests only so that the produced output
            //files can be compared on systems with other locales to give set of known
            //correct results that are on en-US locale.
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");

#if NETFRAMEWORK
            string codeBaseUri = typeof(IntegrationTestBaseClass).Assembly.CodeBase;
            string path = new Uri(codeBaseUri).AbsolutePath;
            var currentAssemblyLocation = new FileInfo(Directory.GetParent(path).FullName);
#else
            // There is an extra folder in the netfx path representing the runtime identifier.
            var currentAssemblyLocation = new FileInfo(typeof(IntegrationTestBaseClass).Assembly.Location);
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

        public void LogTestOutput(object sender, LoggingEventArgs e)
        {
            if (e.Kind >= MessageKindToLog)
                Output.WriteLine(e.Message);
        }

        void IDisposable.Dispose()
        {
            Cleanup();
            Console.WriteLine($"Finished test: {FullTestName}");
        }

        protected virtual void Initialize()
        {
        }

        protected virtual void Cleanup()
        {
        }
    }
}
