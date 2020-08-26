// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tests;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TestRegressionScenario()
        {
            var context = new MLContext();

            string taxiDataPath = GetDataPath("taxi-fare-train.csv");

            var taxiData =
                context.Data.LoadFromTextFile<FeatureContributionTests.TaxiTrip>(taxiDataPath, hasHeader: true,
                    separatorChar: ',');

            var splitData = context.Data.TrainTestSplit(taxiData, testFraction: 0.1);

            IDataView trainingDataView = context.Data.FilterRowsByColumn(splitData.TrainSet, "FareAmount", lowerBound: 1, upperBound: 150);

            var dataProcessPipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                                        .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                                        .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                                        .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                                        .Append(context.Transforms.NormalizeMeanVariance(outputColumnName: "PassengerCount"))
                                        .Append(context.Transforms.NormalizeMeanVariance(outputColumnName: "TripTime"))
                                        .Append(context.Transforms.NormalizeMeanVariance(outputColumnName: "TripDistance"))
                                        .Append(context.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", "PassengerCount",
                                            "TripTime", "TripDistance"));

            var trainer = context.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var model = trainingPipeline.Fit(trainingDataView);

            var predictions = model.Transform(splitData.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Assert.True(metrics.RSquared > .8);
            Assert.True(metrics.RootMeanSquaredError > 2);
        }
    }
}
