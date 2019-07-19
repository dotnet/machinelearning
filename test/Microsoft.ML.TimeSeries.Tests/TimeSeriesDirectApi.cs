// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Transforms.TimeSeries;
using Xunit;

namespace Microsoft.ML.Tests
{
    public sealed class TimeSeries
    {

        private sealed class Prediction
        {
#pragma warning disable CS0649
            [VectorType(4)]
            public double[] Change;
#pragma warning restore CS0649
        }

        private sealed class ForecastPrediction
        {
#pragma warning disable CS0649
            [VectorType(4)]
            public float[] Forecast;
            [VectorType(4)]
            public float[] MinCnf;
            [VectorType(4)]
            public float[] MaxCnf;
#pragma warning restore CS0649
        }

        public class Prediction1
        {
            public float Random;
        }

        private sealed class Data
        {
            public string Text;
            public float Random;
            public float Value;

            public Data(float value)
            {
                Text = "random123value";
                Random = -1;
                Value = value;
            }
        }

        class ForecastResult
        {
#pragma warning disable CS0649
            public float Forecast;
#pragma warning restore CS0649
        }

        class ForecastResultArray
        {
#pragma warning disable CS0649
            public float[] Forecast;
            public float[] ConfidenceLowerBound;
            public float[] ConfidenceUpperBound;
#pragma warning restore CS0649
        }

        private sealed class TimeSeriesData
        {
            public float Value;

            public TimeSeriesData(float value)
            {
                Value = value;
            }
        }

        private sealed class SrCnnAnomalyDetection
        {
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }

        [Fact]
        public void ChangeDetection()
        {
            var env = new MLContext();
            const int size = 10;
            List<Data> data = new List<Data>(size);
            var dataView = env.Data.LoadFromEnumerable(data);
            for (int i = 0; i < size / 2; i++)
                data.Add(new Data(5));

            for (int i = 0; i < size / 2; i++)
                data.Add(new Data((float)(5 + i * 1.1)));

            var args = new IidChangePointDetector.Options()
            {
                Confidence = 80,
                Source = "Value",
                Name = "Change",
                ChangeHistoryLength = size
            };
            // Train
            var detector = new IidChangePointEstimator(env, args).Fit(dataView);
            // Transform
            var output = detector.Transform(dataView);
            // Get predictions
            var enumerator = env.Data.CreateEnumerable<Prediction>(output, true).GetEnumerator();
            Prediction row = null;
            List<double> expectedValues = new List<double>() { 0, 5, 0.5, 5.1200000000000114E-08, 0, 5, 0.4999999995, 5.1200000046080209E-08, 0, 5, 0.4999999995, 5.1200000092160303E-08,
                0, 5, 0.4999999995, 5.12000001382404E-08};
            int index = 0;
            while (enumerator.MoveNext() && index < expectedValues.Count)
            {
                row = enumerator.Current;

                Assert.Equal(expectedValues[index++], row.Change[0]);
                Assert.Equal(expectedValues[index++], row.Change[1]);
                Assert.Equal(expectedValues[index++], row.Change[2]);
                Assert.Equal(expectedValues[index++], row.Change[3]);
            }
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void ChangePointDetectionWithSeasonality()
        {
            var env = new MLContext();
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();
            var dataView = env.Data.LoadFromEnumerable(data);

            var args = new SsaChangePointDetector.Options()
            {
                Confidence = 95,
                Source = "Value",
                Name = "Change",
                ChangeHistoryLength = ChangeHistorySize,
                TrainingWindowSize = MaxTrainingSize,
                SeasonalWindowSize = SeasonalitySize
            };

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));

            // Train
            var detector = new SsaChangePointEstimator(env, args).Fit(dataView);
            // Transform
            var output = detector.Transform(dataView);
            // Get predictions
            var enumerator = env.Data.CreateEnumerable<Prediction>(output, true).GetEnumerator();
            Prediction row = null;
            List<double> expectedValues = new List<double>() { 0, -3.31410598754883, 0.5, 5.12000000000001E-08, 0, 1.5700820684432983, 5.2001145245395008E-07,
                    0.012414560443710681, 0, 1.2854313254356384, 0.28810801662678009, 0.02038940454467935, 0, -1.0950627326965332, 0.36663890634019225, 0.026956459625565483};

            int index = 0;
            while (enumerator.MoveNext() && index < expectedValues.Count)
            {
                row = enumerator.Current;
                Assert.Equal(expectedValues[index++], row.Change[0], precision: 7);  // Alert
                Assert.Equal(expectedValues[index++], row.Change[1], precision: 7);  // Raw score
                Assert.Equal(expectedValues[index++], row.Change[2], precision: 7);  // P-Value score
                Assert.Equal(expectedValues[index++], row.Change[3], precision: 7);  // Martingale score
            }
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void ChangePointDetectionWithSeasonalityPredictionEngineNoColumn()
        {
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1);
            var dataView = ml.Data.LoadFromEnumerable(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));


            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Text_Featurized", "Text")
                .Append(new SsaChangePointEstimator(ml, new SsaChangePointDetector.Options()
                {
                    Confidence = 95,
                    Source = "Value",
                    Name = "Change",
                    ChangeHistoryLength = ChangeHistorySize,
                    TrainingWindowSize = MaxTrainingSize,
                    SeasonalWindowSize = SeasonalitySize
                }));

            // Train.
            var model = pipeline.Fit(dataView);

            //Create prediction function.
            var engine = model.CreateTimeSeriesEngine<Data, Prediction1>(ml);

            //Checkpoint with no inputs passed at prediction.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            //Load time series model and we will use this to pass two inputs and compare the raw score 
            //with "engine".
            ITransformer model2 = null;
            using (var file = File.OpenRead(modelPath))
                model2 = ml.Model.Load(file, out var schema);

            //Raw score after state gets updated with two inputs.
            var engine2 = model2.CreateTimeSeriesEngine<Data, Prediction>(ml);
            var prediction2 = engine2.Predict(new Data(1));
            //Raw score after first input.
            Assert.Equal(1.1661833524703979, prediction2.Change[1], precision: 5); // Raw score
            prediction2 = engine2.Predict(new Data(1));
            //Raw score after second input.
            Assert.Equal(0.12216401100158691, prediction2.Change[1], precision: 5); // Raw score

            //Even though time series column is not requested it will 
            // pass the observation through time series transform and update the state with the first input.
            var prediction = engine.Predict(new Data(1));
            Assert.Equal(-1, prediction.Random);

            //Save the model with state updated with just one input.
            engine.CheckPoint(ml, modelPath + 1);
            ITransformer model3 = null;
            using (var file = File.OpenRead(modelPath + 1))
                model3 = ml.Model.Load(file, out var schema);

            //Load the model with state updated with just one input, then pass in the second input
            //and raw score should match the raw score obtained by passing the two input in the first model.
            var engine3 = model3.CreateTimeSeriesEngine<Data, Prediction>(ml);
            var prediction3 = engine3.Predict(new Data(1));
            Assert.Equal(0.12216401100158691, prediction2.Change[1], precision: 5); // Raw score
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void ChangePointDetectionWithSeasonalityPredictionEngine()
        {
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1);
            var dataView = ml.Data.LoadFromEnumerable(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));


            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Text_Featurized", "Text")
                .Append(ml.Transforms.Conversion.ConvertType("Value", "Value", DataKind.Single))
                .Append(new SsaChangePointEstimator(ml, new SsaChangePointDetector.Options()
                {
                    Confidence = 95,
                    Source = "Value",
                    Name = "Change",
                    ChangeHistoryLength = ChangeHistorySize,
                    TrainingWindowSize = MaxTrainingSize,
                    SeasonalWindowSize = SeasonalitySize
                }));

            // Train.
            var model = pipeline.Fit(dataView);

            //Model 1: Prediction #1.
            var engine = model.CreateTimeSeriesEngine<Data, Prediction>(ml);
            var prediction = engine.Predict(new Data(1));
            Assert.Equal(0, prediction.Change[0], precision: 7); // Alert
            Assert.Equal(1.1661833524703979, prediction.Change[1], precision: 5); // Raw score
            Assert.Equal(0.5, prediction.Change[2], precision: 7); // P-Value score
            Assert.Equal(5.1200000000000114E-08, prediction.Change[3], precision: 7); // Martingale score

            //Model 1: Checkpoint.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            //Model 1: Prediction #2
            prediction = engine.Predict(new Data(1));
            Assert.Equal(0, prediction.Change[0], precision: 7); // Alert
            Assert.Equal(0.12216401100158691, prediction.Change[1], precision: 5); // Raw score
            Assert.Equal(0.14823824685192111, prediction.Change[2], precision: 5); // P-Value score
            Assert.Equal(1.5292508189989167E-07, prediction.Change[3], precision: 7); // Martingale score

            // Load Model 1.
            ITransformer model2 = null;
            using (var file = File.OpenRead(modelPath))
                model2 = ml.Model.Load(file, out var schema);

            //Predict and expect the same result after checkpointing(Prediction #2).
            engine = model2.CreateTimeSeriesEngine<Data, Prediction>(ml);
            prediction = engine.Predict(new Data(1));
            Assert.Equal(0, prediction.Change[0], precision: 7); // Alert
            Assert.Equal(0.12216401100158691, prediction.Change[1], precision: 5); // Raw score
            Assert.Equal(0.14823824685192111, prediction.Change[2], precision: 5); // P-Value score
            Assert.Equal(1.5292508189989167E-07, prediction.Change[3], precision: 5); // Martingale score
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void SsaForecast()
        {
            var env = new MLContext();
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;

            List<Data> data = new List<Data>();
            var dataView = env.Data.LoadFromEnumerable(data);

            var args = new SsaForecastingTransformer.Options()
            {
                ConfidenceLevel = 0.95f,
                Source = "Value",
                Name = "Forecast",
                ConfidenceLowerBoundColumn = "MinCnf",
                ConfidenceUpperBoundColumn = "MaxCnf",
                WindowSize = 10,
                SeriesLength = 11,
                TrainSize = 22,
                Horizon = 4,
                IsAdaptive = true
            };

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));

            // Train
            var detector = new SsaForecastingEstimator(env, args).Fit(dataView);
            // Transform
            var output = detector.Transform(dataView);
            // Get predictions
            var enumerator = env.Data.CreateEnumerable<ForecastPrediction>(output, true).GetEnumerator();
            ForecastPrediction row = null;
            List<float> expectedForecast = new List<float>() { 0.191491723f, 2.53994083f, 5.26454258f, 7.37313938f };
            List<float> minCnf = new List<float>() { -3.9741993f, -2.36872721f, 0.09407653f, 2.18899345f };
            List<float> maxCnf = new List<float>() { 4.3571825f, 7.448609f, 10.435009f, 12.5572853f };
            enumerator.MoveNext();
            row = enumerator.Current;

            for (int localIndex = 0; localIndex < 4; localIndex++)
            {
                Assert.Equal(expectedForecast[localIndex], row.Forecast[localIndex], precision: 7);
                Assert.Equal(minCnf[localIndex], row.MinCnf[localIndex], precision: 7);
                Assert.Equal(maxCnf[localIndex], row.MaxCnf[localIndex], precision: 7);
            }

        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void SsaForecastPredictionEngine()
        {
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1);
            var dataView = ml.Data.LoadFromEnumerable(data);

            var args = new SsaForecastingTransformer.Options()
            {
                ConfidenceLevel = 0.95f,
                Source = "Value",
                Name = "Forecast",
                WindowSize = 10,
                SeriesLength = 11,
                TrainSize = 22,
                Horizon = 4,
                ConfidenceLowerBoundColumn = "ConfidenceLowerBound",
                ConfidenceUpperBoundColumn = "ConfidenceUpperBound",
                VariableHorizon = true
            };

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));

            // Train
            var model = ml.Transforms.Text.FeaturizeText("Text_Featurized", "Text")
                .Append(ml.Transforms.Conversion.ConvertType("Value", "Value", DataKind.Single))
                .Append(ml.Forecasting.ForecastBySsa("Forecast", "Value", 10, 11, 22, 4,
                    confidenceLowerBoundColumn: "ConfidenceLowerBound",
                    confidenceUpperBoundColumn: "ConfidenceUpperBound", variableHorizon: true))
                .Append(ml.Transforms.Concatenate("Forecast", "Forecast", "ConfidenceLowerBound", "ConfidenceUpperBound"))
                .Fit(dataView);

            //Prediction engine.
            var engine = model.CreateTimeSeriesEngine<Data, ForecastResultArray>(ml);
            ForecastResultArray result = new ForecastResultArray();

            // Forecast and change the horizon to 5.
            engine.Predict(null, ref result, horizon: 5);
            // [Forecast, ConfidenceLowerBound, ConfidenceUpperBound]
            Assert.Equal(result.Forecast, new float[] { -1.02245092f, 0.08333081f, 2.60737085f, 5.397319f, 7.500832f, -5.188142f, -4.82533741f,
                -2.563095f, 0.213172823f, 2.29317045f, 3.14324f, 4.991999f, 7.777837f, 10.5814648f, 12.7084932f });

            // Update the forecasting model.
            engine.Predict(new Data(2));

            // Update the model and then forecast.
            engine.Predict(new Data(2), ref result);

            engine.CheckPoint(ml, "model.zip");
            // [Forecast, ConfidenceLowerBound, ConfidenceUpperBound]
            Assert.Equal(result.Forecast, new float[] { 4.310587f, 6.39716768f, 7.73934f, 8.029469f, 0.144895911f,
                1.48849952f, 2.568874f, 2.84532261f, 8.476278f, 11.3058357f, 12.9098063f, 13.2136145f });

            // Checkpoint the model.
            ITransformer modelCopy;
            using (var file = File.OpenRead("model.zip"))
                modelCopy = ml.Model.Load(file, out DataViewSchema schema);

            // We must create a new prediction engine from the persisted model.
            var forecastEngineCopy = modelCopy.CreateTimeSeriesEngine<Data, ForecastResultArray>(ml);
            ForecastResultArray resultCopy = new ForecastResultArray();

            // Update both the models.
            engine.Predict(new Data(3));
            forecastEngineCopy.Predict(new Data(3));

            // Forecast values with the original and check-pointed model.
            forecastEngineCopy.Predict(null, ref resultCopy, horizon: 5);
            engine.Predict(null, ref result, horizon: 5);
            // [Forecast, ConfidenceLowerBound, ConfidenceUpperBound]
            Assert.Equal(result.Forecast, new float[] { 6.00658846f, 7.506871f, 7.96424866f, 7.17514229f,
                5.02655172f, 1.84089744f, 2.59820318f, 2.79378271f, 1.99099624f,
                -0.181109816f, 10.1722794f, 12.41554f, 13.1347151f, 12.3592882f, 10.2342129f});

            // The forecasted results should be the same because the state of the models
            // is the same.
            Assert.Equal(result.Forecast, resultCopy.Forecast);
            
        }

        [Fact]
        public void AnomalyDetectionWithSrCnn()
        {
            var ml = new MLContext();

            // Generate sample series data with an anomaly
            var data = new List<TimeSeriesData>();
            for (int index = 0; index < 20; index++)
            {
                data.Add(new TimeSeriesData(5));
            }
            data.Add(new TimeSeriesData(10));
            for (int index = 0; index < 5; index++)
            {
                data.Add(new TimeSeriesData(5));
            }

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup the estimator arguments
            string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
            string inputColumnName = nameof(TimeSeriesData.Value);

            // The transformed data.
            var transformedData = ml.Transforms.DetectAnomalyBySrCnn(outputColumnName, inputColumnName, 16, 5, 5, 3, 8, 0.35).Fit(dataView).Transform(dataView);

            // Getting the data of the newly created column as an IEnumerable of SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(transformedData, reuseRowObject: false);

            int k = 0;
            foreach (var prediction in predictionColumn)
            {
                if (k == 20)
                    Assert.Equal(1, prediction.Prediction[0]);
                else
                    Assert.Equal(0, prediction.Prediction[0]);
                k += 1;
            }
        }
    }
}
