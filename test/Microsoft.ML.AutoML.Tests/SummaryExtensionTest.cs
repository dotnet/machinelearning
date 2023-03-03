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
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;

namespace Microsoft.ML.AutoML.Test
{
    public class SummaryExtensionTest : BaseTestClass
    {
        public SummaryExtensionTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void TestEstimatorSummary()
        {
            var context = new MLContext();
            var trainers = new List<IEstimator<ITransformer>>();
            trainers.Add(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy());
            trainers.Add(context.BinaryClassification.Trainers.LbfgsLogisticRegression());
            trainers.Add(context.Regression.Trainers.LbfgsPoissonRegression());
            trainers.Add(context.BinaryClassification.Trainers.SdcaLogisticRegression());
            trainers.Add(context.BinaryClassification.Trainers.SdcaNonCalibrated());
            trainers.Add(context.MulticlassClassification.Trainers.SdcaNonCalibrated());
            trainers.Add(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(l1Regularization: 1, l2Regularization: 0.1f));
            trainers.Add(context.Regression.Trainers.Sdca());
            trainers.Add(context.BinaryClassification.Trainers.SgdCalibrated());
            trainers.Add(context.BinaryClassification.Trainers.SgdNonCalibrated());
            trainers.Add(context.BinaryClassification.Trainers.AveragedPerceptron());
            trainers.Add(context.BinaryClassification.Trainers.LinearSvm());
            trainers.Add(context.BinaryClassification.Trainers.LdSvm());
            trainers.Add(context.MulticlassClassification.Trainers.OneVersusAll(context.BinaryClassification.Trainers.LdSvm()));
            trainers.Add(context.MulticlassClassification.Trainers.PairwiseCoupling(context.BinaryClassification.Trainers.LdSvm()));
            trainers.Add(context.Regression.Trainers.OnlineGradientDescent());
            trainers.Add(context.MulticlassClassification.Trainers.NaiveBayes());
            var summaries = trainers.Select(t => t.Summary());

            Approvals.VerifyAll(summaries, label: "");
        }
    }
}
