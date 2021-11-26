// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A theory for BenchmarkDotNet tests.
    /// </summary>
    public sealed class BenchmarkTheoryAttribute : EnvironmentSpecificTheoryAttribute
    {
#if DEBUG
        private const string SkipMessage = "BenchmarkDotNet does not allow running the benchmarks in Debug, so this test is disabled for DEBUG";
        private readonly bool _isEnvironmentSupported = false;
#elif NET461
        private const string SkipMessage = "We are currently not running Benchmarks for FullFramework";
        private readonly bool _isEnvironmentSupported = false;
#else
        private const string SkipMessage = "We don't support 32 bit yet";
        private readonly bool _isEnvironmentSupported = System.Environment.Is64BitProcess;
#endif

        public BenchmarkTheoryAttribute() : base(SkipMessage)
        {
        }

        protected override bool IsEnvironmentSupported() => _isEnvironmentSupported;
    }
}
