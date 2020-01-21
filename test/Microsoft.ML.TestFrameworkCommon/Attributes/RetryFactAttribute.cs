// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Xunit.Sdk;

namespace Microsoft.ML.TestFrameworkCommon.Attributes
{
    /// <summary>
    /// ML.NET facts that will retry several flaky test cases, use default timeout settings
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public class RetryFactAttribute : FactAttribute
    {
        /// <summary>
        /// Number of retries allowed for a failed test. If unset (or set less than 1), will
        /// default to 2 attempts.
        /// </summary>
        public int MaxRetries { get; set; }
    }


    /// <summary>
    /// ML.NET facts that will retry several flaky test cases, use default timeout settings
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public class RetryLessThanNetCore30OrNotNetCoreFactAttribute : EnvironmentSpecificFactAttribute
    {
        public RetryLessThanNetCore30OrNotNetCoreFactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return AppDomain.CurrentDomain.GetData("FX_PRODUCT_VERSION") == null;
        }
        /// <summary>
        /// Number of retries allowed for a failed test. If unset (or set less than 1), will
        /// default to 2 attempts.
        /// </summary>
        public int MaxRetries { get; set; }
    }

    /// <summary>
    /// A fact for tests requiring X64 environment.
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public sealed class RetryX64FactAttribute : EnvironmentSpecificFactAttribute
    {
        public RetryX64FactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return Environment.Is64BitProcess;
        }

        public int MaxRetries { get; set; }
    }

    /// <summary>
    /// A fact for tests requiring TensorFlow.
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public sealed class RetryTensorFlowFactAttribute : EnvironmentSpecificFactAttribute
    {
        public RetryTensorFlowFactAttribute() : base("TensorFlow is 64-bit only")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return Environment.Is64BitProcess;
        }

        public int MaxRetries { get; set; }
    }

    /// <summary>
    /// A fact for tests requiring matrix factorization.
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public sealed class RetryMatrixFactorizationFactAttribute : EnvironmentSpecificFactAttribute
    {
        public RetryMatrixFactorizationFactAttribute() : base("")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return true;
        }

        public int MaxRetries { get; set; }
    }

    /// <summary>
    /// A fact for tests requiring x64 environment and either .NET Core version lower than 3.0 or framework other than .NET Core.
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public sealed class RetryLessThanNetCore30OrNotNetCoreAndX64FactAttribute : EnvironmentSpecificFactAttribute
    {
        public RetryLessThanNetCore30OrNotNetCoreAndX64FactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return Environment.Is64BitProcess && AppDomain.CurrentDomain.GetData("FX_PRODUCT_VERSION") == null;
        }

        public int MaxRetries { get; set; }
    }

    /// <summary>
    /// A fact for tests requiring LightGBM.
    /// </summary>
    [XunitTestCaseDiscoverer("Microsoft.ML.TestFrameworkCommon.RetryFactDiscoverer", "Microsoft.ML.TestFrameworkCommon")]
    public sealed class RetryLightGBMFactAttribute : EnvironmentSpecificFactAttribute
    {
        public RetryLightGBMFactAttribute() : base("LightGBM is 64-bit only")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return Environment.Is64BitProcess;
        }

        public int MaxRetries { get; set; }
    }
}

