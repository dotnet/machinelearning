//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace RunTestsMore
{
    public class PermutationFeatureImportanceCommandTests : TestSteppedDmCommandBase
    {
        public PermutationFeatureImportanceCommandTests(ITestOutputHelper output) 
            : base(output)
        {
        }

        [Fact]
        public void CommandPermutationFeatureImportance()
        {
            // Prepare model.zip for pfi.
            var modelPath = ModelPath();
            string pathData = GetDataPath("breast-cancer-withheader.txt");
            string cmd = string.Format("train loader=text{{header+ col=Label:0 col=Features:1-9}} tr=lr{{nt=1}} xf[norm]=MinMax{{col=Features}} data={{{0}}} out={{{1}}}", pathData, modelPath.Path);
            MainForTest(cmd);

            // Test pfi command.
            OutputPath reportPath = CreateOutputPath("report.txt");
            TestInCore("pfi", pathData, modelPath, "seed=42", reportPath.Arg("rout"));
            Done();
        }

        [Fact]
        public void CommandPermutationFeatureImportanceWithScoresTop()
        {
            // Prepare model.zip for pfi.
            var modelPath = ModelPath();
            string pathData = GetDataPath(@"../UCI", "adult.train");
            string cmd = string.Format("train tr=AP{{iter=2}} " +
                "loader=TextLoader{{sep=, col=Features:R4:0,2,4,10-12 col=workclass:TX:1 col=education:TX:3 col=marital_status:TX:5 col=occupation:TX:6 col=relationship:TX:7 col=race:TX:8 col=sex:TX:9 col=native_country:TX:13 col=label_IsOver50K_:R4:14 header=+}} " +
                "xf=CopyColumns{{col=Label:label_IsOver50K_}} " +
                "xf=CategoricalTransform{{col=workclass col=education col=marital_status col=occupation col=relationship col=race col=sex col=native_country}} " +
                "xf=Concat{{col=Features:Features,workclass,education,marital_status,occupation,relationship,race,sex,native_country}} " +
                "xf=Keep{{col=Label col=Features}}  data={{{0}}} out={{{1}}} seed=42", pathData, modelPath.Path);
            MainForTest(cmd);

            // Test pfi command.
            OutputPath reportPath = CreateOutputPath("report.txt");
            TestInCore("pfi", pathData, modelPath, "top=100 seed=42", reportPath.Arg("rout"));
            Done();
        }
    }
}
