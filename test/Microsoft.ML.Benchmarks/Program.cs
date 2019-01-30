// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Globalization;
using System.Threading;
using BenchmarkDotNet.Running;

namespace Microsoft.ML.Benchmarks
{
    class Program
    {
        /// <summary>
        /// execute dotnet run -c Release and choose the benchmarks you want to run
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // enforce Neutral Language as "en-us" because the input data files use dot as decimal separator (and it fails for cultures with ",")
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            BenchmarkSwitcher
               .FromAssembly(typeof(Program).Assembly)
               .Run(args, new RecommendedConfig());
        }
    }
}
