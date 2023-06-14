// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.ML.AutoML.Test;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class TrainValidaionDatasetManagerTest : BaseTestClass
    {
        public TrainValidaionDatasetManagerTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TrainValidationDatasetManagerSubSamplingTest()
        {
            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);

            var trainDataLength = DatasetDimensionsUtil.CountRows(trainData, ulong.MaxValue);
            trainDataLength.Should().Be(500);

            var trainValidationDatasetManager = new TrainValidateDatasetManager(trainData, trainData, "SubSampleKey");

            var parameter = Parameter.CreateNestedParameter();
            parameter[nameof(TrainValidateDatasetManager)] = Parameter.CreateNestedParameter();
            parameter[nameof(TrainValidateDatasetManager)][trainValidationDatasetManager.SubSamplingKey] = Parameter.FromDouble(0.3);
            var setting = new TrialSettings
            {
                Parameter = parameter,
            };

            var subSampleTrainData = trainValidationDatasetManager.LoadTrainDataset(context, setting);
            var subSampleTrainDataLength = DatasetDimensionsUtil.CountRows(subSampleTrainData, ulong.MaxValue);
            subSampleTrainDataLength.Should().Be(150);
        }
    }
}
