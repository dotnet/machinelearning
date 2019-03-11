// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data.IO;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class PcaTests : TestDataPipeBase
    {
        private readonly string _dataSource;
        private readonly TextSaver _saver;

        public PcaTests(ITestOutputHelper helper)
            : base(helper)
        {
            _dataSource = GetDataPath("generated_regression_dataset.csv");
            _saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
        }

        [Fact]
        public void PcaWorkout()
        {
            var data = TextLoaderStatic.CreateLoader(_env,
                c => (label: c.LoadFloat(11), weight: c.LoadFloat(0), features: c.LoadFloat(1, 10)),
                separator: ';', hasHeader: true)
                .Load(_dataSource);

            var invalidData = TextLoaderStatic.CreateLoader(_env,
                c => (label: c.LoadFloat(11), weight: c.LoadFloat(0), features: c.LoadText(1, 10)),
                separator: ';', hasHeader: true)
                .Load(_dataSource);

            var est = ML.Transforms.ProjectToPrincipalComponents("pca", "features", rank: 4, seed: 10);
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var estNonDefaultArgs = ML.Transforms.ProjectToPrincipalComponents("pca", "features", rank: 3, exampleWeightColumnName: "weight", overSampling: 2, ensureZeroMean: false);
            TestEstimatorCore(estNonDefaultArgs, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            Done();
        }

        [Fact]
        public void TestPcaEstimator()
        {
            var data = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(_dataSource);

            var est = ML.Transforms.ProjectToPrincipalComponents("pca", "features", rank: 5, seed: 1);
            var outputPath = GetOutputPath("PCA", "pca.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
            savedData = ML.Transforms.SelectColumns("pca").Fit(savedData).Transform(savedData);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);

            CheckEquality("PCA", "pca.tsv", digitsOfPrecision: 4);
            Done();
        }
    }
}
