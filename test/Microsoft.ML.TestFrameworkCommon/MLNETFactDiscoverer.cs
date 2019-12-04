// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.ComponentModel;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace Microsoft.ML.TestFrameworkCommon
{
    public class MLNETFactDiscoverer : IXunitTestCaseDiscoverer
    {
        readonly IMessageSink diagnosticMessageSink;
        readonly IList<string> flakyTestLists;

        public MLNETFactDiscoverer(IMessageSink diagnosticMessageSink)
        {
            this.diagnosticMessageSink = diagnosticMessageSink;
            flakyTestLists = new List<string> {
                "Microsoft.ML.Tests.TimeSeries.SsaForecast",
                "Microsoft.ML.RunTests.TestPredictors.MulticlassTreeFeaturizedLRTest",
                "Microsoft.ML.Tests.Scenarios.Api.CookbookSamples.CookbookSamplesDynamicApi.CrossValidationIris",
                "Microsoft.ML.Scenarios.TensorFlowScenariosTests.TensorFlowImageClassification",
                "Microsoft.ML.RunTests.TestPredictors.MulticlassLRTest",
                "Microsoft.ML.RunTests.TestDataPipe.SavePipeDraculaKeyLabel",
                "Microsoft.ML.RunTests.TestEntryPoints.EntryPointLogisticRegressionMulticlass",
                "Microsoft.ML.RunTests.TestEntryPoints.TestCrossValidationMacroWithStratification"
            };
        }

        public IEnumerable<IXunitTestCase> Discover(ITestFrameworkDiscoveryOptions discoveryOptions, ITestMethod testMethod, IAttributeInfo factAttribute)
        {
            var displayName = factAttribute.GetNamedArgument<string>("DisplayName");

            var maxRetries = factAttribute.GetNamedArgument<int>("MaxRetries");
            if (maxRetries < 1)
                maxRetries = flakyTestLists.Contains(displayName) ? 3 : 1;

            var timeOut = factAttribute.GetNamedArgument<int>("Timeout");
            if (timeOut < 1)
                timeOut = 5 * 60 * 1000;


            yield return new MLNETTestCase(diagnosticMessageSink, discoveryOptions.MethodDisplayOrDefault(), testMethod, maxRetries, timeOut);
        }
    }
}
