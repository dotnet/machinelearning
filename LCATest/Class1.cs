using System;

namespace LCATest
{
    public class Class1 : BaseTestClass
    {

        private class RootCauseLocalizationData
        {
            [RootCauseLocalizationInputType]
            public RootCauseLocalizationInput Input { get; set; }

            public RootCauseLocalizationData()
            {
                Input = null;
            }

            public RootCauseLocalizationData(DateTime anomalyTimestamp, Dictionary<string, string> anomalyDimensions, List<MetricSlice> slices, DTRootCauseLocalizationEstimator.AggregateType aggregateteType, string aggregateSymbol)
            {
                Input = new RootCauseLocalizationInput(anomalyTimestamp, anomalyDimensions, slices, aggregateteType, aggregateSymbol);
            }
        }

        private class RootCauseLocalizationTransformedData
        {
            [RootCauseType()]
            public RootCause RootCause { get; set; }

            public RootCauseLocalizationTransformedData()
            {
                RootCause = null;
            }
        }

        [Fact]
        public void RootCauseLocalizationWithDT()
        {
            // Create an root cause localizatiom input list.
            var rootCauseLocalizationData = new List<RootCauseLocalizationData>() { new RootCauseLocalizationData(new DateTime(), new Dictionary<String, String>(), new List<MetricSlice>() { new MetricSlice(new DateTime(), new List<Microsoft.ML.TimeSeries.Point>()) }, DTRootCauseLocalizationEstimator.AggregateType.Sum, "SUM"), new RootCauseLocalizationData(new DateTime(), new Dictionary<String, String>(), new List<MetricSlice>() { new MetricSlice(new DateTime(), new List<Microsoft.ML.TimeSeries.Point>()) }, DTRootCauseLocalizationEstimator.AggregateType.Avg, "AVG") };

            var ml = new MLContext(1);
            // Convert the list of root cause data to an IDataView object, which is consumable by ML.NET API.
            var data = ml.Data.LoadFromEnumerable(rootCauseLocalizationData);

            // Create pipeline to localize root cause by decision tree.
            var pipeline = ml.Transforms.LocalizeRootCauseByDT(nameof(RootCauseLocalizationTransformedData.RootCause), nameof(RootCauseLocalizationData.Input));

            // Fit the model.
            var model = pipeline.Fit(data);

            // Test path:  input list -> IDataView -> Enumerable of RootCauseLocalizationInputs.
            var transformedData = model.Transform(data);

            // Load input list in DataView back to Enumerable.
            var transformedDataPoints = ml.Data.CreateEnumerable<RootCauseLocalizationTransformedData>(transformedData, false);

            foreach (var dataPoint in transformedDataPoints)
            {
                var rootCause = dataPoint.RootCause;

                Assert.NotNull(rootCause);
            }

            var engine = ml.Model.CreatePredictionEngine<RootCauseLocalizationData, RootCauseLocalizationTransformedData>(model);
            var newRootCauseInput = new RootCauseLocalizationData(new DateTime(), new Dictionary<String, String>(), new List<MetricSlice>() { new MetricSlice(new DateTime(), new List<Microsoft.ML.TimeSeries.Point>()) }, DTRootCauseLocalizationEstimator.AggregateType.Sum, "SUM");
            var transformedRootCause = engine.Predict(newRootCauseInput);

            Assert.NotNull(transformedRootCause);
            //todo - will add more tests here when onboarding mock data
        }
    }
}
