// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.RunTests;
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
            var data = ML.Data.LoadFromTextFile(_dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("weight", DataKind.Single, 0),
                new TextLoader.Column("features", DataKind.Single, 1, 10)
            }, hasHeader: true, separatorChar: ';');

            var invalidData = ML.Data.LoadFromTextFile(_dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("weight", DataKind.Single, 0),
                new TextLoader.Column("features", DataKind.String, 1, 10)
            }, hasHeader: true, separatorChar: ';');

            var est = ML.Transforms.ProjectToPrincipalComponents("pca", "features", rank: 4, seed: 10);
            TestEstimatorCore(est, data, invalidInput: invalidData);

            var estNonDefaultArgs = ML.Transforms.ProjectToPrincipalComponents("pca", "features", rank: 3, exampleWeightColumnName: "weight", overSampling: 2, ensureZeroMean: false);
            TestEstimatorCore(estNonDefaultArgs, data, invalidInput: invalidData);

            Done();
        }

        [Fact]
        public void TestPcaEstimator()
        {
            var data = ML.Data.LoadFromTextFile(_dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var est = ML.Transforms.ProjectToPrincipalComponents("pca", "features", rank: 5, seed: 1);
            var outputPath = GetOutputPath("PCA", "pca.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
            savedData = ML.Transforms.SelectColumns("pca").Fit(savedData).Transform(savedData);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);

            CheckEquality("PCA", "pca.tsv", digitsOfPrecision: 4);
            Done();
        }
    }
}
