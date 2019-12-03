// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.ComponentModel;
using System.Threading;
using System.Threading.Tasks;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace Microsoft.ML.TestFrameworkCommon
{
    [Serializable]
    public class MLNETTestCase : XunitTestCase
    {
        private int maxRetries;

        [EditorBrowsable(EditorBrowsableState.Never)]
        [Obsolete("Called by the de-serializer", true)]
        public MLNETTestCase() { }

        public MLNETTestCase(IMessageSink diagnosticMessageSink, TestMethodDisplay testMethodDisplay, ITestMethod testMethod, 
            int maxRetries, int timeOut)
            : base(diagnosticMessageSink, testMethodDisplay, TestMethodDisplayOptions.None, testMethod, testMethodArguments: null)
        {
            this.maxRetries = maxRetries;
            Timeout = timeOut;
        }

        // This method is called by the xUnit test framework classes to run the test case. We will do the
        // loop here, forwarding on to the implementation in XunitTestCase to do the heavy lifting. We will
        // continue to re-run the test until the aggregator has an error (meaning that some internal error
        // condition happened), or the test runs without failure, or we've hit the maximum number of tries.
        public override async Task<RunSummary> RunAsync(IMessageSink diagnosticMessageSink,
                                                        IMessageBus messageBus,
                                                        object[] constructorArguments,
                                                        ExceptionAggregator aggregator,
                                                        CancellationTokenSource cancellationTokenSource)
        {
            var runCount = 0;

            while (true)
            {
                // This is really the only tricky bit: we need to capture and delay messages (since those will
                // contain run status) until we know we've decided to accept the final result;
                var delayedMessageBus = new DelayedMessageBus(messageBus);
                RunSummary summary = null;

                try
                {
                    // write to the console when a test starts and stops so we can identify any test hangs/deadlocks in CI
                    Console.WriteLine($"[{DateTime.Now}] Starting test: {DisplayName}.");
                    summary = await base.RunAsync(diagnosticMessageSink, delayedMessageBus, constructorArguments, aggregator, cancellationTokenSource);
                    Console.WriteLine($"[{DateTime.Now}] Finished test: {DisplayName}.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Test {DisplayName} has unhandled exception {ex.StackTrace}.");
                    Console.WriteLine($"Call stack is : {new System.Diagnostics.StackTrace().ToString()}");
                }

                if (aggregator.HasExceptions || summary.Failed > 0)
                {
                    string os = Environment.OSVersion.ToString();
                    string architecture = Environment.Is64BitProcess ? "x64" : "x86";
                    string framework = "";
                    string configuration = "";

#if DEBUG
                    configuration = "Debug";
#else
                    configuration = "Release";
#endif

#if NETFRAMEWORK
                    framework = "NetFx461";
#else
                    framework = AppDomain.CurrentDomain.GetData("FX_PRODUCT_VERSION") == null ? "NetCoreApp21" : "NetCoreApp30";
#endif

                    var errorMessage = $"Execution of '{DisplayName}' at {os}_{architecture}_{framework}_{configuration} failed (attempt #{runCount}).";
                    diagnosticMessageSink.OnMessage(new DiagnosticMessage(errorMessage));
                    Console.WriteLine(errorMessage);
                }

                if (aggregator.HasExceptions || summary.Failed == 0 || ++runCount >= maxRetries)
                {
                    delayedMessageBus.Dispose();  // Sends all the delayed messages
                    return summary;
                }
            }
        }

        public override void Serialize(IXunitSerializationInfo data)
        {
            base.Serialize(data);

            data.AddValue("MaxRetries", maxRetries);
        }

        public override void Deserialize(IXunitSerializationInfo data)
        {
            base.Deserialize(data);

            maxRetries = data.GetValue<int>("MaxRetries");
        }
    }
}
