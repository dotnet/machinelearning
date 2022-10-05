// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class TrialResultManagerTest : BaseTestClass
    {
        public TrialResultManagerTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact(Skip = "failing part of ci test, need investigation")]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void CsvTrialResultManager_end_to_end_test()
        {
            var lgbmSearchSpace = new SearchSpace.SearchSpace<LgbmOption>();
            var tuner = new RandomSearchTuner(lgbmSearchSpace, 0);
            var trialResults = Enumerable.Range(0, 10)
                        .Select((i) =>
                        {
                            var trialSettings = new TrialSettings
                            {
                                TrialId = i,
                            };
                            var parameter = tuner.Propose(trialSettings);
                            trialSettings.Parameter = parameter;
                            return new TrialResult
                            {
                                TrialSettings = trialSettings,
                                DurationInMilliseconds = 10.123,
                                Loss = i * 0.99,
                                PeakCpu = i * 0.98,
                                PeakMemoryInMegaByte = i * 0.97,
                            };
                        });
            var tempFilePath = Path.Combine(OutDir, Path.GetRandomFileName() + ".txt");
            var csvTrialResultManager = new CsvTrialResultManager(tempFilePath, lgbmSearchSpace);

            // the tempFile is empty, so GetAllTrialResults should be 0;
            csvTrialResultManager.GetAllTrialResults().Count().Should().Be(0);

            // Add trialResults to csvTrialResultManager, the # of trials should be 10.
            foreach (var trialResult in trialResults)
            {
                csvTrialResultManager.AddOrUpdateTrialResult(trialResult);
            }
            csvTrialResultManager.GetAllTrialResults().Count().Should().Be(10);

            // if repeated trial added, csvTrialResultManager should update the existing trial.
            foreach (var trialResult in trialResults)
            {
                csvTrialResultManager.AddOrUpdateTrialResult(trialResult);
            }
            csvTrialResultManager.GetAllTrialResults().Count().Should().Be(10);

            // save as csv test
            csvTrialResultManager.Save();

            // reload test.
            csvTrialResultManager = new CsvTrialResultManager(tempFilePath, lgbmSearchSpace);
            csvTrialResultManager.GetAllTrialResults().Count().Should().Be(10);

            var fileContent = File.ReadAllText(tempFilePath);

            // replace line break to \r
            fileContent = fileContent.Replace(Environment.NewLine, "\r");
            Output.WriteLine(fileContent);
            File.Delete(tempFilePath);
            Approvals.Verify(fileContent);
        }
    }
}
