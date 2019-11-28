// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;
using Xunit.Sdk;

namespace Microsoft.ML.TestFrameworkCommon.Attributes
{
    /// <summary>
    /// Works just like Fact Attribute except that failures are retried (by default, 3 times).
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public class RetryFactAttribute : FactAttribute
    {
        /// <summary>
        /// Number of retries allowed for a failed test. If unset (or set less than 1), will
        /// default to 3 attempts.
        /// </summary>
        public int MaxRetries { get; set; }
    }
}