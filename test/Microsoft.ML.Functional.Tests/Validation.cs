// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers.HalLearners;
using Xunit;

namespace Microsoft.ML.Functional.Tests
{
    public partial class ValidationScenarios
    {
        /// <summary>
        /// Cross-validation: Have a mechanism to do cross validation, that is, you come up with
        /// a data source (optionally with stratification column), come up with an instantiable transform
        /// and trainer pipeline, and it will handle (1) splitting up the data, (2) training the separate
        /// pipelines on in-fold data, (3) scoring on the out-fold data, (4) returning the set of
        /// evaluations and optionally trained pipes. (People always want metrics out of xfold,
        /// they sometimes want the actual models too.)
        /// </summary>
        [Fact]
        void CrossValidation()
        {
            var mlContext = new MLContext(seed: 789);

            // Get the dataset, create a train and test
            var data = DatasetUtils.LoadHousingRegressionDataset(mlContext);

            // Create a pipeline to train on the sentiment data
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] {
                    "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                    "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"})
                .Append(mlContext.Transforms.CopyColumns("Label", "MedianHomeValue"))
                .Append(mlContext.Regression.Trainers.OrdinaryLeastSquares());

            // Compute the CV result
            var cvResult = mlContext.Regression.CrossValidate(data, pipeline, numFolds: 5);

            // Check that the results are valid
            Assert.IsType<RegressionMetrics>(cvResult[0].metrics);
            Assert.IsType<TransformerChain<RegressionPredictionTransformer<OlsLinearRegressionModelParameters>>>(cvResult[0].model);
            Assert.True(cvResult[0].scoredTestData is IDataView);
            Assert.Equal(5, cvResult.Length);
        }
    }
}
