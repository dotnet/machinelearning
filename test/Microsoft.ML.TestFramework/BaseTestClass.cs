// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Internallearn.Test;
using System;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Threading;
using Xunit.Abstractions;

namespace Microsoft.ML.TestFramework
{
    public class BaseTestClass : IDisposable
    {
        private readonly string _rootDir;
        private readonly string _outDir;
        private readonly string _dataRoot;

        public string TestName { get; set; }
        public string FullTestName { get; set; }

        static BaseTestClass() => GlobalBase.AssemblyInit();

        public BaseTestClass(ITestOutputHelper output)
        {
            //This locale is currently set for tests only so that the produced output
            //files can be compared on systems with other locales to give set of known
            //correct results that are on en-US locale.
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");

            var currentAssemblyLocation = new FileInfo(typeof(BaseTestClass).Assembly.Location);
            _rootDir = currentAssemblyLocation.Directory.Parent.Parent.Parent.Parent.FullName;
            _outDir = Path.Combine(currentAssemblyLocation.Directory.FullName, "TestOutput");
            Directory.CreateDirectory(_outDir);
            _dataRoot = Path.Combine(_rootDir, "test", "data");
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

        protected string RootDir => _rootDir;
        protected string OutDir => _outDir;

        protected ITestOutputHelper Output { get; }

        protected string GetDataPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(_dataRoot, name));
        }
        protected string GetDataPath(string subDir, string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(_dataRoot, subDir, name));
        }

        protected void EnsureOutputDir(string subDir)
        {
            Directory.CreateDirectory(Path.Combine(_outDir, subDir));
        }
        protected string GetOutputPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(_outDir, name);
        }
        protected string GetOutputPath(string subDir, string name)
        {
            if (string.IsNullOrWhiteSpace(subDir))
                return GetOutputPath(name);
            EnsureOutputDir(subDir);
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.Combine(_outDir, subDir, name); // REVIEW: put the path in in braces in case the path has spaces
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
