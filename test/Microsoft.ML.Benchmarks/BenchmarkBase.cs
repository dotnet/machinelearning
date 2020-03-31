// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.TestFrameworkCommon;

namespace Microsoft.ML.Benchmarks
{
    public class BenchmarkBase
    {
        // Make sure DataDir is initialized before benchmark running.
        static BenchmarkBase()
        {
            RootDir = TestCommon.GetRepoRoot();
            DataDir = Path.Combine(RootDir, "test", "data");
        }

        protected static string RootDir { get; }
        protected static string DataDir { get; }

        // Don't use BaseTestClass's GetDataPath method instead for benchmark.
        // BaseTestClass's static constructor is not guaranteed to be called before
        // benchmark running (depending on CLR version this has different behaviour).
        // The problem with executing BaseTestClass's static constructor when benchmark
        // is running is it sometime cause process hanging when the constructor trying 
        // to load MKL, this is related to below issue:
        // https://github.com/dotnet/machinelearning/issues/1073
        public static string GetBenchmarkDataPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(DataDir, name));
        }
    }
}
