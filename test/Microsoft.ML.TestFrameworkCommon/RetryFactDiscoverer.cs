// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace Microsoft.ML.TestFrameworkCommon
{
    public class RetryFactDiscoverer : IXunitTestCaseDiscoverer
    {
        readonly IMessageSink _diagnosticMessageSink;

        public RetryFactDiscoverer(IMessageSink diagnosticMessageSink)
        {
            this._diagnosticMessageSink = diagnosticMessageSink;
        }

        public IEnumerable<IXunitTestCase> Discover(ITestFrameworkDiscoveryOptions discoveryOptions, 
            ITestMethod testMethod, IAttributeInfo factAttribute)
        {
            //by default, retry failed tests at max 2 times
            var maxRetries = factAttribute.GetNamedArgument<int>("MaxRetries");
            if (maxRetries < 1)
                maxRetries = 2;

            yield return new RetryTestCase(_diagnosticMessageSink, discoveryOptions.MethodDisplayOrDefault(), testMethod, maxRetries);
        }
    }
}

