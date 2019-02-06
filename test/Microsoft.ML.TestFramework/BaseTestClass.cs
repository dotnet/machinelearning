// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Threading;
using Microsoft.ML.Internal.Internallearn.Test;
using Xunit.Abstractions;

namespace Microsoft.ML.TestFramework
{
    public class BaseTestClass : IDisposable
    {
        public string TestName { get; set; }
        public string FullTestName { get; set; }

        static BaseTestClass()
        {
            GlobalBase.AssemblyInit();
            RootDir = GetRepoRoot();
            DataDir = Path.Combine(RootDir, "test", "data");
        }

        private static string GetRepoRoot()
        {
#if NETFRAMEWORK
            string directory = AppDomain.CurrentDomain.BaseDirectory;
#else
            string directory = AppContext.BaseDirectory;
#endif

            while (!Directory.Exists(Path.Combine(directory, ".git")) && directory != null)
            {
                directory = Directory.GetParent(directory).FullName;
            }

            if (directory == null)
            {
                return null;
            }
            return directory;
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

            // write to the console when a test starts and stops so we can identify any test hangs/deadlocks in CI
            Console.WriteLine($"Starting test: {FullTestName}");
            Initialize();
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

        protected void EnsureOutputDir(string subDir)
        {
            Directory.CreateDirectory(Path.Combine(OutDir, subDir));
        }
        protected string GetOutputPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(OutDir, name);
        }
        protected string GetOutputPath(string subDir, string name)
        {
            if (string.IsNullOrWhiteSpace(subDir))
                return GetOutputPath(name);
            EnsureOutputDir(subDir);
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(OutDir, subDir, name); // REVIEW: put the path in in braces in case the path has spaces
        }
        protected string DeleteOutputPath(string subDir, string name)
        {
            string path = GetOutputPath(subDir, name);
            if (!string.IsNullOrWhiteSpace(path))
                File.Delete(path);
            return path;
        }
        protected string DeleteOutputPath(string name)
        {
            string path = GetOutputPath(name);
            if (!string.IsNullOrWhiteSpace(path))
                File.Delete(path);
            return path;
        }
    }
}
