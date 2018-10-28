// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class TestTimeSeries : TestDataPipeBase
    {
        protected override void Initialize()
        {
            base.Initialize();
            Env.ComponentCatalog.RegisterAssembly(typeof(ExponentialAverageTransform).Assembly);
        }

        public TestTimeSeries(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact(Skip = "Fails due to inability to run in parallel cursors. Temporarily disabling this until stateful prediction engine is implemented.")]
        public void SavePipeIidSpike()
        {
            TestCore(GetDataPath(Path.Combine("Timeseries", "real_1.csv")),
                true,
                    new[]{"loader=TextLoader{sep=, col=Features:R4:1 header=+}",
                    "xf=IidSpikeDetector{src=Features name=Anomaly cnf=99.5 wnd=200 side=Positive}",
                    "xf=Convert{col=fAnomaly:R4:Anomaly}",
                    "xf=IidSpikeDetector{src=Features name=Anomaly2 cnf=99.5 wnd=200 side=Negative}",
                    "xf=Convert{col=fAnomaly2:R4:Anomaly2}",
                    "xf=Select{keepcol=Features keepcol=fAnomaly keepcol=fAnomaly2}" });

            Done();
        }

        [Fact(Skip = "Fails due to inability to run in parallel cursors. Temporarily disabling this until stateful prediction engine is implemented.")]
        public void SavePipeIidChangePoint()
        {
            TestCore(GetDataPath(Path.Combine("Timeseries", "real_11.csv")),
                true,
                    new[]{"loader=TextLoader{sep=, col=Features:R4:1 header=+}",
                    @"xf=IidChangePointDetector{src=Features name=Anomaly cnf=83 wnd=100 mart=Power eps=0.1}",
                    "xf=Convert{col=fAnomaly:R4:Anomaly}",
                    "xf=IidChangePointDetector{src=Features name=Anomaly2 cnf=83 wnd=100 mart=Mixture}",
                    "xf=Convert{col=fAnomaly2:R4:Anomaly2}",
                    "xf=Select{keepcol=Features keepcol=fAnomaly keepcol=fAnomaly2}" });

            Done();
        }

        [Fact(Skip = "Randomly generated dataset causes asserts to fire. Temporarily disabling this test until we find a real TS dataset.")]
        public void SavePipeSsaSpike()
        {
            TestCore(GetDataPath(Path.Combine("Timeseries", "A4Benchmark-TS2.csv")),
                true,
                    new[]{"loader=TextLoader{sep=, col=Features:R4:1 header=+}",
                    @"xf=SsaSpikeDetector{src=Features name=Anomaly twnd=500 swnd=50 err=SignedDifference cnf=99.5 wnd=100 side=Negative}",
                    "xf=Convert{col=fAnomaly:R4:Anomaly}",
                    "xf=SsaSpikeDetector{src=Features name=Anomaly2 twnd=500 swnd=50 err=SignedDifference cnf=99.5 wnd=100 side=Positive}",
                    "xf=Convert{col=fAnomaly2:R4:Anomaly2}",
                    "xf=SsaSpikeDetector{src=Features name=Anomaly3 twnd=500 swnd=50 err=SignedDifference cnf=99.5 wnd=100}",
                    "xf=Convert{col=fAnomaly3:R4:Anomaly3}",
                    "xf=Select{keepcol=Features keepcol=fAnomaly keepcol=fAnomaly2 keepcol=fAnomaly3}" });

            Done();
        }

        [Fact]
        public void SavePipeSsaSpikeNoData()
        {
            string pathData = DeleteOutputPath("SavePipe", "SsaSpikeNoData.txt");
            File.WriteAllLines(pathData, Enumerable.Repeat("0", 50));

            // The rank should not be equivalent to window size if input data is all zeros. This is a regression test.
            TestCore(pathData,
                true,
                    new[]{"loader=TextLoader{col=Features:R4:0}",
                    "xf=SsaSpikeDetector{src=Features name=Anomaly twnd=50 swnd=5 err=SignedDifference cnf=99.5 wnd=10}" });

            Done();
        }

        [Fact]
        public void SavePipeExponentialAverage()
        {
            TestCore(null, true,
                    new[]{"loader=Text{col=Input:R4:1}",
                    "xf=ExpAvg{src=Input name=Output d=0.9}" });

            Done();
        }

        [Fact]
        public void SavePipeSlidingWindow()
        {
            TestCore(null, true,
                    new[]{"loader=Text{col=Input:R4:1}",
                    "xf=SlideWinTransform{src=Input name=Output wnd=3 l=0}" });

            Done();
        }

        [Fact]
        public void SavePipeSlidingWindowW1L1()
        {
            TestCore(null, true,
                new[]{"loader=Text{col=Input:R4:1}",
                "xf=SlideWinTransform{src=Input name=Output wnd=1 l=1}" });

            Done();
        }

        [Fact]
        public void SavePipeSlidingWindowW2L1()
        {
            TestCore(null, true,
                    new[]{"loader=Text{col=Input:R4:1}",
                    "xf=SlideWinTransform{src=Input name=Output wnd=2 l=1}" });

            Done();
        }

        [Fact]
        public void SavePipeSlidingWindowW1L2()
        {
            TestCore(null, true,
                    new[]{"loader=Text{col=Input:R4:1}",
                    "xf=SlideWinTransform{src=Input name=Output wnd=1 l=2}" });

            Done();
        }


        [Fact]
        public void SavePipePValue()
        {
            TestCore(null, true,
                    new[]{"loader=Text{col=Input:R4:1}",
                    "xf=PVal{src=Input name=Output wnd=10}"});

            Done();
        }

        [Fact]
        public void SavePipePercentileThreshold()
        {
            TestCore(null, true,
                    new[]{"loader=Text{col=Input:R4:1}",
                    "xf=TopPcnt{src=Input name=Output wnd=10 pcnt=10}" });

            Done();
        }

        [Fact]
        public void SavePipeMovingAverageUniform()
        {
            TestCore(null, true,
                    new[]{"loader=Text{col=Input:R4:1}",
                    "xf=MoAv{src=Input name=Output wnd=2 l=0}" });

            Done();
        }

        [Fact]
        public void SavePipeMovingAverageNonUniform()
        {
            TestCore(null, true,
                new[]{"loader=Text{col=Input:R4:1}",
                    "xf=MoAv{src=Input name=Output wnd=3 weights=1,1.5,2 l=0}" });

            Done();
        }

    }
}