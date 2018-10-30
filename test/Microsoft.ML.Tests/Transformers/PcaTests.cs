// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.PCA;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class PcaTests : TestDataPipeBase
    {
        private readonly ConsoleEnvironment _env;
        private readonly string _dataSource;
        private readonly TextSaver _saver;

        public PcaTests(ITestOutputHelper helper)
            : base(helper)
        {
            _env = new ConsoleEnvironment(seed: 1);
            _dataSource = GetDataPath("generated_regression_dataset.csv");
            _saver = new TextSaver(_env, new TextSaver.Arguments { Silent = true, OutputHeader = false });
        }

        [Fact]
        public void PcaWorkout()
        {
            var data = TextLoader.CreateReader(_env,
                c => (label: c.LoadFloat(11), weight: c.LoadFloat(0), features: c.LoadFloat(1, 10)),
                separator: ';', hasHeader: true)
                .Read(_dataSource);

            var invalidData = TextLoader.CreateReader(_env,
                c => (label: c.LoadFloat(11), weight: c.LoadFloat(0), features: c.LoadText(1, 10)),
                separator: ';', hasHeader: true)
                .Read(_dataSource);

            var est = new PcaEstimator(_env, "features", "pca", rank: 4, seed: 10);
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var estNonDefaultArgs = new PcaEstimator(_env, "features", "pca", rank: 3, weightColumn: "weight", overSampling: 2, center: false);
            TestEstimatorCore(estNonDefaultArgs, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            Done();
        }

        [Fact]
        public void TestPcaEstimator()
        {
            var data = TextLoader.CreateReader(_env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(_dataSource);

            var est = new PcaEstimator(_env, "features", "pca", rank: 5, seed: 1);
            var outputPath = GetOutputPath("PCA", "pca.tsv");
            using (var ch = _env.Start("save"))
            {
                IDataView savedData = TakeFilter.Create(_env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = new ChooseColumnsTransform(_env, savedData, "pca");

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, _saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("PCA", "pca.tsv", digitsOfPrecision: 4);
            Done();
        }
    }
}
