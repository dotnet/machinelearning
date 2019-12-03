// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;
using Xunit.Sdk;

namespace Microsoft.ML.TestFrameworkCommon.Attributes
{
    /// <summary>
    /// ML.NET facts that will retry several flaky test cases, use default timeout settings
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.MLNETFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public class MLNETFactAttribute : FactAttribute
    {
        /// <summary>
        /// Number of retries allowed for a failed test. If unset (or set less than 1), will
        /// default to 3 attempts.
        /// </summary>
        public int MaxRetries { get; set; }
    }
}