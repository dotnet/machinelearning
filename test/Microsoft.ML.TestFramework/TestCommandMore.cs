// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed partial class TestCommandMore : TestSteppedDmCommandBase
    {
        public TestCommandMore(ITestOutputHelper helper) : base(helper)
        {
        }

        [Fact]
        [TestCategory(Cat)]
        public void CommandTrainScorePriorTrainer()
        {
            // First run a training on any dataset.
            string pathData = GetDataPath("breast-cancer-withheader.txt");
            OutputPath trainModel = ModelPath();
            TestCore("train", pathData, "loader=text{header+}", "lab=Label feat=Features tr=priorpredictor{}");

            // Then, run the score.
            _step++;
            OutputPath scorePath = CreateOutputPath("data.idv");
            OutputPath scoreModel = CreateOutputPath("score-datamodel.zip");
            string extraScore = string.Format("all=+ feat=Features dout={{{0}}} out={{{1}}}",
                scorePath.Path, scoreModel.Path);
            TestInCore("score", pathData, trainModel, extraScore);
            TestPipeFromModel(pathData, scoreModel);

            // Transform the score output to txt for baseline
            _step++;
            OutputPath scorePathTxt = CreateOutputPath("data.txt");
            TestReloadedCore("savedata", scorePath.Path, "loader=binary", "saver=text", "", scorePathTxt.Arg("dout"));

            Done();
        }

        [TestCategory(Cat), TestCategory("LinearClassification"), TestCategory("SDCA")]
        [Fact]
        public void CommandCVLinearClassification()
        {
            const string floatLabelLoaderCmdline = @"loader=Text{col=Label:Num:0 col=Features:Num:~ header+}";
            const string doubleLabelLoaderCmdline = @"loader=Text{col=Label:R8:0 col=Features:Num:~ header+}";
            const string singleLabelLoaderCmdline = @"loader=Text{col=Label:R4:0 col=Features:Num:~ header+}";
            const string keyLabelLoaderCmdline = @"loader=Text{col=Label:U4[0-1]:0 col=Features:Num:~ header+}";
            const string dvBoolLabelLoaderCmdline = @"loader=Text{col=Label:BL:0 col=Features:Num:~ header+}";
            const string sdcaTrainingCmdline = @"lab=Label feat=Features tr=sdca{l2=1e-06 l1=0 iter=10 checkFreq=-1 nt=1} seed=1 useThreads-";
            const string sgdTrainingCmdline = @"lab=Label feat=Features tr=sgd{nt=1 checkFreq=-1} seed=1 useThreads-";

            string pathData = GetDataPath(@"breast-cancer-withheader.txt");

            TestCore("CV", pathData, floatLabelLoaderCmdline, sdcaTrainingCmdline);

            _step++;
            TestCore("CV", pathData, doubleLabelLoaderCmdline, sdcaTrainingCmdline);

            _step++;
            TestCore("CV", pathData, singleLabelLoaderCmdline, sdcaTrainingCmdline);

            _step++;
            TestCore("CV", pathData, keyLabelLoaderCmdline, sdcaTrainingCmdline);

            _step++;
            TestCore("CV", pathData, dvBoolLabelLoaderCmdline, sdcaTrainingCmdline);

            _step++;
            TestCore("CV", pathData, floatLabelLoaderCmdline, sgdTrainingCmdline);

            _step++;
            TestCore("CV", pathData, keyLabelLoaderCmdline, sgdTrainingCmdline);

            _step++;
            TestCore("CV", pathData, dvBoolLabelLoaderCmdline, sgdTrainingCmdline);

            _step++;
            TestCore("CV", pathData, doubleLabelLoaderCmdline, sgdTrainingCmdline);

            _step++;
            TestCore("CV", pathData, singleLabelLoaderCmdline, sgdTrainingCmdline);

            Done();
        }

        [Fact]
        [TestCategory(Cat)]
        public void CommandVersionTypeCheck()
        {
            using (var writer = new StringWriter())
            {
                using (var env = new TlcEnvironment(7, outWriter: writer))
                {
                    MainForTest(env, writer, "Version");
                }
                writer.Flush();
                string commandOutput = writer.ToString();
                commandOutput = commandOutput.Replace(writer.NewLine, "");
                Version ver;
                Assert.True(Version.TryParse(commandOutput, out ver));
                Done();
            }
        }
    }
}
