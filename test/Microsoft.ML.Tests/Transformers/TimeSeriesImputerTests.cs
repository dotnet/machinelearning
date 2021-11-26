// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing.Printing;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class TimeSeriesImputerTests : TestDataPipeBase
    {
        public TimeSeriesImputerTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TimeSeriesTwoGrainInput
        {
            public long date;
            public string grainA;
            public string grainB;
            public float data;
        }

        private class TimeSeriesOneGrainInput
        {
            public long date;
            public string grainA;
            public int dataA;
            public float dataB;
            public uint dataC;
        }

        private class TimeSeriesOneGrainFloatInput
        {
            public long date;
            public string grainA;
            public float dataA;
        }

        private class TimeSeriesOneGrainStringInput
        {
            public long date;
            public string grainA;
            public string dataA;
        }

        [FeaturizersFact]
        public void NotImputeOneColumn()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new TimeSeriesOneGrainInput() { date = 25, grainA = "A", dataA = 1, dataB = 2.0f, dataC = 5 },
                new TimeSeriesOneGrainInput() { date = 26, grainA = "A", dataA = 1, dataB = 2.0f, dataC = 5 },
                new TimeSeriesOneGrainInput() { date = 28, grainA = "A", dataA = 1, dataB = 2.0f, dataC = 5 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" }, new string[] { "dataB" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // We always output the same column as the input, plus adding a column saying whether the row was imputed or not.
            Assert.Equal(6, schema.Count);
            Assert.Equal("date", schema[0].Name);
            Assert.Equal("grainA", schema[1].Name);
            Assert.Equal("dataA", schema[2].Name);
            Assert.Equal("dataB", schema[3].Name);
            Assert.Equal("dataC", schema[4].Name);
            Assert.Equal("IsRowImputed", schema[5].Name);

            // We are imputing 1 row, so total rows should be 4.
            var preview = output.Preview();
            Assert.Equal(4, preview.RowView.Length);

            // Row that was imputed should have date of 27
            Assert.Equal(27L, preview.ColumnView[0].Values[2]);

            // Since we are not imputing data on one column and a row is getting imputed, its value should be default(T)
            Assert.Equal(default(float), preview.ColumnView[3].Values[2]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void ImputeOnlyOneColumn()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new TimeSeriesOneGrainInput() { date = 25, grainA = "A", dataA = 1, dataB = 2.0f, dataC = 5 },
                new TimeSeriesOneGrainInput() { date = 26, grainA = "A", dataA = 1, dataB = 2.0f, dataC = 5 },
                new TimeSeriesOneGrainInput() { date = 28, grainA = "A", dataA = 1, dataB = 2.0f, dataC = 5 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" }, new string[] { "dataB" }, TimeSeriesImputerEstimator.FilterMode.Include);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // We always output the same column as the input, plus adding a column saying whether the row was imputed or not.
            Assert.Equal(6, schema.Count);
            Assert.Equal("date", schema[0].Name);
            Assert.Equal("grainA", schema[1].Name);
            Assert.Equal("dataA", schema[2].Name);
            Assert.Equal("dataB", schema[3].Name);
            Assert.Equal("dataC", schema[4].Name);
            Assert.Equal("IsRowImputed", schema[5].Name);

            // We are imputing 1 row, so total rows should be 4.
            var preview = output.Preview();
            Assert.Equal(4, preview.RowView.Length);

            // Row that was imputed should have date of 27
            Assert.Equal(27L, preview.ColumnView[0].Values[2]);

            // Since we are not imputing data on two columns and a row is getting imputed, its value should be default(T)
            Assert.Equal(default(int), preview.ColumnView[2].Values[2]);
            Assert.Equal(default(uint), preview.ColumnView[4].Values[2]);

            // Column that was imputed should have value of 2.0f
            Assert.Equal(2.0f, preview.ColumnView[3].Values[2]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void Forwardfill()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new TimeSeriesOneGrainFloatInput() { date = 0, grainA = "A", dataA = 2.0f },
                new TimeSeriesOneGrainFloatInput() { date = 1, grainA = "A", dataA = float.NaN },
                new TimeSeriesOneGrainFloatInput() { date = 3, grainA = "A", dataA = 5.0f },
                new TimeSeriesOneGrainFloatInput() { date = 5, grainA = "A", dataA = float.NaN },
                new TimeSeriesOneGrainFloatInput() { date = 7, grainA = "A", dataA = float.NaN }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var prev = output.Preview();

            // Should have 3 original columns + 1 more for IsRowImputed
            Assert.Equal(4, output.Schema.Count);

            // Imputing rows with dates 2,4,6, so should have length of 8
            Assert.Equal(8, prev.RowView.Length);

            // Check that imputed rows have the correct dates
            Assert.Equal(2L, prev.ColumnView[0].Values[2]);
            Assert.Equal(4L, prev.ColumnView[0].Values[4]);
            Assert.Equal(6L, prev.ColumnView[0].Values[6]);

            // Make sure grain was propagated correctly
            Assert.Equal("A", prev.ColumnView[1].Values[2].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[4].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[6].ToString());

            // Make sure forward fill is working as expected. All NA's should be replaced, and imputed rows should have correct values too
            Assert.Equal(2.0f, prev.ColumnView[2].Values[1]);
            Assert.Equal(2.0f, prev.ColumnView[2].Values[2]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[4]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[5]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[6]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[7]);

            // Make sure IsRowImputed is true for row 2, 4,6 , false for the rest
            Assert.Equal(false, prev.ColumnView[3].Values[0]);
            Assert.Equal(false, prev.ColumnView[3].Values[1]);
            Assert.Equal(true, prev.ColumnView[3].Values[2]);
            Assert.Equal(false, prev.ColumnView[3].Values[3]);
            Assert.Equal(true, prev.ColumnView[3].Values[4]);
            Assert.Equal(false, prev.ColumnView[3].Values[5]);
            Assert.Equal(true, prev.ColumnView[3].Values[6]);
            Assert.Equal(false, prev.ColumnView[3].Values[7]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void DateTimeSupportForwardfill()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { date = new DateTime(1970, 1, 1), grainA = "A", dataA = 2.0f },
                new { date = new DateTime(1970, 1, 3), grainA = "A", dataA = float.NaN },
                new { date = new DateTime(1970, 1, 5), grainA = "A", dataA = 5.0f },
                new { date = new DateTime(1970, 1, 7), grainA = "A", dataA = float.NaN },
                new { date = new DateTime(1970, 1, 8), grainA = "A", dataA = float.NaN }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var prev = output.Preview();

            // Should have 3 original columns + 1 more for IsRowImputed
            Assert.Equal(4, output.Schema.Count);

            // Imputing rows with days for 2,4,6, so should have length of 8
            Assert.Equal(8, prev.RowView.Length);

            // Check that imputed rows have the correct dates
            Assert.Equal(new DateTime(1970, 1, 2), prev.ColumnView[0].Values[1]);
            Assert.Equal(new DateTime(1970, 1, 4), prev.ColumnView[0].Values[3]);
            Assert.Equal(new DateTime(1970, 1, 6), prev.ColumnView[0].Values[5]);

            // Make sure grain was propagated correctly
            Assert.Equal("A", prev.ColumnView[1].Values[1].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[3].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[5].ToString());

            // Make sure forward fill is working as expected. All NA's should be replaced, and imputed rows should have correct values too
            Assert.Equal(2.0f, prev.ColumnView[2].Values[1]);
            Assert.Equal(2.0f, prev.ColumnView[2].Values[2]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[4]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[5]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[6]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[7]);

            // Make sure IsRowImputed is true for row 1, 3, 5, false for the rest
            Assert.Equal(false, prev.ColumnView[3].Values[0]);
            Assert.Equal(true, prev.ColumnView[3].Values[1]);
            Assert.Equal(false, prev.ColumnView[3].Values[2]);
            Assert.Equal(true, prev.ColumnView[3].Values[3]);
            Assert.Equal(false, prev.ColumnView[3].Values[4]);
            Assert.Equal(true, prev.ColumnView[3].Values[5]);
            Assert.Equal(false, prev.ColumnView[3].Values[6]);
            Assert.Equal(false, prev.ColumnView[3].Values[7]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void EntryPoint()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { ts = 1L, grain = 1970, c3 = 10, c4 = 19},
                new { ts = 2L, grain = 1970, c3 = 13, c4 = 12},
                new { ts = 3L, grain = 1970, c3 = 15, c4 = 16},
                new { ts = 5L, grain = 1970, c3 = 20, c4 = 19}
            };

            var data = mlContext.Data.LoadFromEnumerable(dataList);
            TimeSeriesImputerEstimator.Options options = new TimeSeriesImputerEstimator.Options()
            {
                TimeSeriesColumn = "ts",
                GrainColumns = new[] { "grain" },
                FilterColumns = new[] { "c3", "c4" },
                FilterMode = TimeSeriesImputerEstimator.FilterMode.Include,
                ImputeMode = TimeSeriesImputerEstimator.ImputationStrategy.ForwardFill,
                Data = data
            };

            var entryOutput = TimeSeriesTransformerEntrypoint.TimeSeriesImputer(mlContext.Transforms.GetEnvironment(), options);
            // Build the pipeline, fit, and transform it.
            var output = entryOutput.OutputData;

            // Get the data from the first row and make sure it matches expected
            var prev = output.Preview();

            // Should have 4 original columns + 1 more for IsRowImputed
            Assert.Equal(5, output.Schema.Count);

            // Imputing rows with date 4 so should have length of 5
            Assert.Equal(5, prev.RowView.Length);

            // Check that imputed rows have the correct dates
            Assert.Equal(4L, prev.ColumnView[0].Values[3]);

            // Make sure grain was propagated correctly
            Assert.Equal(1970, prev.ColumnView[1].Values[2]);

            // Make sure forward fill is working as expected. All NA's should be replaced, and imputed rows should have correct values too
            Assert.Equal(15, prev.ColumnView[2].Values[3]);
            Assert.Equal(16, prev.ColumnView[3].Values[3]);

            // Make sure IsRowImputed is true for row 4, false for the rest
            Assert.Equal(false, prev.ColumnView[4].Values[0]);
            Assert.Equal(false, prev.ColumnView[4].Values[1]);
            Assert.Equal(false, prev.ColumnView[4].Values[2]);
            Assert.Equal(true, prev.ColumnView[4].Values[3]);
            Assert.Equal(false, prev.ColumnView[4].Values[4]);

            Done();
        }

        [FeaturizersFact]
        public void Median()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new TimeSeriesOneGrainFloatInput() { date = 0, grainA = "A", dataA = 2.0f },
                new TimeSeriesOneGrainFloatInput() { date = 1, grainA = "A", dataA = float.NaN },
                new TimeSeriesOneGrainFloatInput() { date = 3, grainA = "A", dataA = 5.0f },
                new TimeSeriesOneGrainFloatInput() { date = 5, grainA = "A", dataA = float.NaN },
                new TimeSeriesOneGrainFloatInput() { date = 7, grainA = "A", dataA = float.NaN }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" }, imputeMode: TimeSeriesImputerEstimator.ImputationStrategy.Median, filterColumns: null, suppressTypeErrors: true);
            var model = pipeline.Fit(data);

            var output = model.Transform(data);

            var prev = output.Preview();

            // Should have 3 original columns + 1 more for IsRowImputed
            Assert.Equal(4, output.Schema.Count);

            // Imputing rows with dates 2,4,6, so should have length of 8
            Assert.Equal(8, prev.RowView.Length);

            // Check that imputed rows have the correct dates
            Assert.Equal(2L, prev.ColumnView[0].Values[2]);
            Assert.Equal(4L, prev.ColumnView[0].Values[4]);
            Assert.Equal(6L, prev.ColumnView[0].Values[6]);

            // Make sure grain was propagated correctly
            Assert.Equal("A", prev.ColumnView[1].Values[2].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[4].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[6].ToString());

            // Make sure Median is working as expected. All NA's should be replaced, and imputed rows should have correct values too
            Assert.Equal(3.5f, prev.ColumnView[2].Values[1]);
            Assert.Equal(3.5f, prev.ColumnView[2].Values[2]);
            Assert.Equal(3.5f, prev.ColumnView[2].Values[4]);
            Assert.Equal(3.5f, prev.ColumnView[2].Values[5]);
            Assert.Equal(3.5f, prev.ColumnView[2].Values[6]);
            Assert.Equal(3.5f, prev.ColumnView[2].Values[7]);

            // Make sure IsRowImputed is true for row 2, 4,6 , false for the rest
            Assert.Equal(false, prev.ColumnView[3].Values[0]);
            Assert.Equal(false, prev.ColumnView[3].Values[1]);
            Assert.Equal(true, prev.ColumnView[3].Values[2]);
            Assert.Equal(false, prev.ColumnView[3].Values[3]);
            Assert.Equal(true, prev.ColumnView[3].Values[4]);
            Assert.Equal(false, prev.ColumnView[3].Values[5]);
            Assert.Equal(true, prev.ColumnView[3].Values[6]);
            Assert.Equal(false, prev.ColumnView[3].Values[7]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void DateTimeTypeSupportMedian()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { date = new DateTime(1970,1,1), grainA = "A", dataA = 2.0f },
                new { date = new DateTime(1970,1,2), grainA = "A", dataA = float.NaN },
                new { date = new DateTime(1970,1,4), grainA = "A", dataA = 5.0f }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" }, imputeMode: TimeSeriesImputerEstimator.ImputationStrategy.Median, filterColumns: null, suppressTypeErrors: true);
            var model = pipeline.Fit(data);

            var output = model.Transform(data);

            var prev = output.Preview();

            // Should have 3 original columns + 1 more for IsRowImputed
            Assert.Equal(4, output.Schema.Count);

            // Imputing one row, so should have length of 4
            Assert.Equal(4, prev.RowView.Length);

            // Check that all rows have the correct dates
            Assert.Equal(new DateTime(1970, 1, 1), prev.ColumnView[0].Values[0]);
            Assert.Equal(new DateTime(1970, 1, 2), prev.ColumnView[0].Values[1]);
            Assert.Equal(new DateTime(1970, 1, 3), prev.ColumnView[0].Values[2]);
            Assert.Equal(new DateTime(1970, 1, 4), prev.ColumnView[0].Values[3]);

            // Make sure Median is working as expected. All NA's should be replaced, and imputed rows should have correct values too
            Assert.Equal(3.5f, prev.ColumnView[2].Values[1]);
            Assert.Equal(3.5f, prev.ColumnView[2].Values[2]);

            // Make sure IsRowImputed is true for imputed row, false for others.
            Assert.Equal(false, prev.ColumnView[3].Values[0]);
            Assert.Equal(false, prev.ColumnView[3].Values[1]);
            Assert.Equal(true, prev.ColumnView[3].Values[2]);
            Assert.Equal(false, prev.ColumnView[3].Values[3]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void Backfill()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new TimeSeriesOneGrainFloatInput() { date = 0, grainA = "A", dataA = float.NaN },
                new TimeSeriesOneGrainFloatInput() { date = 1, grainA = "A", dataA = float.NaN },
                new TimeSeriesOneGrainFloatInput() { date = 3, grainA = "A", dataA = 5.0f },
                new TimeSeriesOneGrainFloatInput() { date = 5, grainA = "A", dataA = float.NaN },
                new TimeSeriesOneGrainFloatInput() { date = 7, grainA = "A", dataA = 2.0f }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" }, TimeSeriesImputerEstimator.ImputationStrategy.BackFill);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var prev = output.Preview();

            // Should have 3 original columns + 1 more for IsRowImputed
            Assert.Equal(4, output.Schema.Count);

            // Imputing rows with dates 2,4,6, so should have length of 8
            Assert.Equal(8, prev.RowView.Length);

            // Check that imputed rows have the correct dates
            Assert.Equal(2L, prev.ColumnView[0].Values[2]);
            Assert.Equal(4L, prev.ColumnView[0].Values[4]);
            Assert.Equal(6L, prev.ColumnView[0].Values[6]);

            // Make sure grain was propagated correctly
            Assert.Equal("A", prev.ColumnView[1].Values[2].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[4].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[6].ToString());

            // Make sure backfill is working as expected. All NA's should be replaced, and imputed rows should have correct values too
            Assert.Equal(5.0f, prev.ColumnView[2].Values[0]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[1]);
            Assert.Equal(5.0f, prev.ColumnView[2].Values[2]);
            Assert.Equal(2.0f, prev.ColumnView[2].Values[4]);
            Assert.Equal(2.0f, prev.ColumnView[2].Values[5]);
            Assert.Equal(2.0f, prev.ColumnView[2].Values[6]);

            // Make sure IsRowImputed is true for row 2, 4,6 , false for the rest
            Assert.Equal(false, prev.ColumnView[3].Values[0]);
            Assert.Equal(false, prev.ColumnView[3].Values[1]);
            Assert.Equal(true, prev.ColumnView[3].Values[2]);
            Assert.Equal(false, prev.ColumnView[3].Values[3]);
            Assert.Equal(true, prev.ColumnView[3].Values[4]);
            Assert.Equal(false, prev.ColumnView[3].Values[5]);
            Assert.Equal(true, prev.ColumnView[3].Values[6]);
            Assert.Equal(false, prev.ColumnView[3].Values[7]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void BackfillTwoGrain()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new TimeSeriesTwoGrainInput() { date = 0, grainA = "A", grainB = "A", data = float.NaN},
                new TimeSeriesTwoGrainInput() { date = 1, grainA = "A", grainB = "A", data = 0.0f},
                new TimeSeriesTwoGrainInput() { date = 3, grainA = "A", grainB = "B", data = 1.0f},
                new TimeSeriesTwoGrainInput() { date = 5, grainA = "A", grainB = "B", data = float.NaN},
                new TimeSeriesTwoGrainInput() { date = 7, grainA = "A", grainB = "B", data = 2.0f }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA", "grainB" }, TimeSeriesImputerEstimator.ImputationStrategy.BackFill);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var prev = output.Preview();

            // Should have 4 original columns + 1 more for IsRowImputed
            Assert.Equal(5, output.Schema.Count);

            // Imputing rows with dates 4,6, so should have length of 8
            Assert.Equal(7, prev.RowView.Length);

            // Check that imputed rows have the correct dates
            Assert.Equal(4L, prev.ColumnView[0].Values[3]);
            Assert.Equal(6L, prev.ColumnView[0].Values[5]);

            // Make sure grain was propagated correctly
            Assert.Equal("A", prev.ColumnView[1].Values[3].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[5].ToString());
            Assert.Equal("B", prev.ColumnView[2].Values[3].ToString());
            Assert.Equal("B", prev.ColumnView[2].Values[5].ToString());

            // Make sure backfill is working as expected. All NA's should be replaced, and imputed rows should have correct values too
            Assert.Equal(0.0f, prev.ColumnView[3].Values[0]);
            Assert.Equal(2.0f, prev.ColumnView[3].Values[3]);
            Assert.Equal(2.0f, prev.ColumnView[3].Values[4]);
            Assert.Equal(2.0f, prev.ColumnView[3].Values[5]);

            // Make sure IsRowImputed is true for row 4,6 false for the rest
            Assert.Equal(false, prev.ColumnView[4].Values[0]);
            Assert.Equal(false, prev.ColumnView[4].Values[1]);
            Assert.Equal(false, prev.ColumnView[4].Values[2]);
            Assert.Equal(true, prev.ColumnView[4].Values[3]);
            Assert.Equal(false, prev.ColumnView[4].Values[4]);
            Assert.Equal(true, prev.ColumnView[4].Values[5]);
            Assert.Equal(false, prev.ColumnView[4].Values[6]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [FeaturizersFact]
        public void InvalidTypeForImputationStrategy()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new TimeSeriesOneGrainStringInput(){ date = 0L, grainA = "A", dataA = "zero" },
                new TimeSeriesOneGrainStringInput(){ date = 1L, grainA = "A", dataA = "one" },
                new TimeSeriesOneGrainStringInput(){ date = 3L, grainA = "A", dataA = "three" },
                new TimeSeriesOneGrainStringInput(){ date = 5L, grainA = "A", dataA = "five" },
                new TimeSeriesOneGrainStringInput(){ date = 7L, grainA = "A", dataA = "seven" }};
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // When suppressTypeErrors is set to false this will throw an error.
            var pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" }, imputeMode: TimeSeriesImputerEstimator.ImputationStrategy.Median, filterColumns: null, suppressTypeErrors: false);
            var ex = Assert.Throws<System.Exception>(() => pipeline.Fit(data));
            Assert.Equal("Only Numeric type columns are supported for ImputationStrategy median. (use suppressError flag to skip imputing non-numeric types)", ex.Message);

            // When suppressTypeErrors is set to true then the default value will be used.
            pipeline = mlContext.Transforms.ReplaceMissingTimeSeriesValues("date", new string[] { "grainA" }, imputeMode: TimeSeriesImputerEstimator.ImputationStrategy.Median, filterColumns: null, suppressTypeErrors: true);
            var model = pipeline.Fit(data);

            var output = model.Transform(data);
            var prev = output.Preview();

            // Should have 3 original columns + 1 more for IsRowImputed
            Assert.Equal(4, output.Schema.Count);

            // Imputing rows with dates 2,4,6, so should have length of 8
            Assert.Equal(8, prev.RowView.Length);

            // Check that imputed rows have the default value
            Assert.Equal("", prev.ColumnView[2].Values[2].ToString());
            Assert.Equal("", prev.ColumnView[2].Values[4].ToString());
            Assert.Equal("", prev.ColumnView[2].Values[6].ToString());

            // Make sure grain was propagated correctly
            Assert.Equal("A", prev.ColumnView[1].Values[2].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[4].ToString());
            Assert.Equal("A", prev.ColumnView[1].Values[6].ToString());

            // Make sure original values stayed the same
            Assert.Equal("zero", prev.ColumnView[2].Values[0].ToString());
            Assert.Equal("one", prev.ColumnView[2].Values[1].ToString());
            Assert.Equal("three", prev.ColumnView[2].Values[3].ToString());
            Assert.Equal("five", prev.ColumnView[2].Values[5].ToString());
            Assert.Equal("seven", prev.ColumnView[2].Values[7].ToString());

            // Make sure IsRowImputed is true for row 2, 4,6 , false for the rest
            Assert.Equal(false, prev.ColumnView[3].Values[0]);
            Assert.Equal(false, prev.ColumnView[3].Values[1]);
            Assert.Equal(true, prev.ColumnView[3].Values[2]);
            Assert.Equal(false, prev.ColumnView[3].Values[3]);
            Assert.Equal(true, prev.ColumnView[3].Values[4]);
            Assert.Equal(false, prev.ColumnView[3].Values[5]);
            Assert.Equal(true, prev.ColumnView[3].Values[6]);
            Assert.Equal(false, prev.ColumnView[3].Values[7]);

            TestEstimatorCore(pipeline, data);

            Done();
        }
    }
}
