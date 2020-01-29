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
    public class RetryTestCase : XunitTestCase
    {
        private int maxRetries;

        [EditorBrowsable(EditorBrowsableState.Never)]
        [Obsolete("Called by the de-serializer", true)]
        public RetryTestCase() { }

        public RetryTestCase(IMessageSink diagnosticMessageSink, TestMethodDisplay testMethodDisplay, 
            ITestMethod testMethod, int maxRetries)
            : base(diagnosticMessageSink, testMethodDisplay, TestMethodDisplayOptions.None, testMethod, testMethodArguments: null)
        {
            this.maxRetries = maxRetries;
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

                RunSummary summary = await base.RunAsync(diagnosticMessageSink, delayedMessageBus, constructorArguments, aggregator, cancellationTokenSource);
                if (aggregator.HasExceptions || summary.Failed > 0)
                {
                    var details = ExtractTestFailDetailsFromMessageBus(delayedMessageBus);
                    var errorMessage = $"Execution of '{DisplayName}' failed (attempt #{runCount + 1}) with details {details}.";

                    diagnosticMessageSink.OnMessage(new DiagnosticMessage(errorMessage));
                    Console.WriteLine(errorMessage);
                }

                if (summary.Failed == 0 || ++runCount >= maxRetries)
                {
                    delayedMessageBus.Dispose();  // Sends all the delayed messages
                    return summary;
                }
            }
        }

        private static string ExtractTestFailDetailsFromMessageBus(DelayedMessageBus delayedMessageBus)
        {
            string details = "";

            foreach (var message in delayedMessageBus.messages)
            {
                if (message.ToString() == "Xunit.Sdk.TestFailed")
                {
                    try
                    {
                        var messages = (string[])message.GetType().GetProperty("Messages").GetValue(message);
                        var exceptionTypes = (string[])message.GetType().GetProperty("ExceptionTypes").GetValue(message);
                        var stackTraces = (string[])message.GetType().GetProperty("StackTraces").GetValue(message);

                        if (messages != null && messages.Length > 0)
                        {
                            details += "Messages: " + string.Join(";", messages) + ". ";
                        }

                        if (exceptionTypes != null && exceptionTypes.Length > 0)
                        {
                            details += "ExceptionTypes: " + string.Join(";", exceptionTypes) + ". ";
                        }

                        if (stackTraces != null && stackTraces.Length > 0)
                        {
                            details += "StackTraces: " + string.Join(";", stackTraces) + ".";
                        }
                    }
                    catch
                    {
                        Console.WriteLine($"Fail to read test fail message from message bus.");
                    }
                }
            }

            return details;
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

