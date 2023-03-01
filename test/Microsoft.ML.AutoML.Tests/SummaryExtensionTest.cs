// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;
using Microsoft.ML.AutoML.Utils;
using FluentAssertions;

namespace Microsoft.ML.AutoML.Test
{
    public class SummaryExtensionTest : BaseTestClass
    {
        public SummaryExtensionTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void TestLbfgsMaximumEntropyMulticlassTrainerSummary()
        {
            var context = new MLContext();
            var trainer = context.MulticlassClassification.Trainers.LbfgsMaximumEntropy();
            var summary = trainer.Summary();
            summary.Should().Be("shit");
        }
    }
}
