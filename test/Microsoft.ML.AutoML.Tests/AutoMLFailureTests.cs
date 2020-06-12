// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class AutoMLFailureTests : BaseTestClass
    {
        public AutoMLFailureTests(ITestOutputHelper output) : base(output)
        {
        }

        public class ModelInput
        {
            [ColumnName("Label"), LoadColumn(0)]
            public int Label { get; set; }


            [ColumnName("ProblematicColumn"), LoadColumn(1)]
            public string ProblematicColumn { get; set; }

        }

        [Fact]
        public void CrossValidationOverflowTest()
        {
            // This test is introduced for https://github.com/dotnet/machinelearning/issues/5211
            // that provides users an informational exception message
            MLContext mlContext = new MLContext(1);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: GetDataPath("cross_validation_overflow_dataset.txt"),
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            IDataView testDataView = mlContext.Data.BootstrapSample(trainingDataView);

            ExperimentResult<MulticlassClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(60)
                .Execute(trainingDataView, labelColumnName: "Label");
            RunDetail<MulticlassClassificationMetrics> bestRun = experimentResult.BestRun;
            IDataView testDataViewWithBestScore = bestRun.Model.Transform(testDataView);

            try
            {
                var testMetrics = mlContext.MulticlassClassification.CrossValidate(
                                      testDataViewWithBestScore,
                                      bestRun.Estimator,
                                      numberOfFolds: 5,
                                      labelColumnName: "Label");
                Assert.True(false);
            }
            catch (System.Exception ex)
            {
                Assert.Contains("Arithmetic operation resulted in an overflow. Related column: ProblematicColumn", ex.Message);
                return;
            }
            
        }
    }
}


