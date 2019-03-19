// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Xunit;
using Xunit.Sdk;

namespace Microsoft.ML.Functional.Tests
{
    internal static class Common
    {
        /// <summary>
        /// Asssert that an <see cref="IDataView"/> rows are of <see cref="TypeTestData"/>.
        /// </summary>
        /// <param name="testTypeDataset">An <see cref="IDataView"/>.</param>
        public static void AssertTypeTestDataset(IDataView testTypeDataset)
        {
            var toyClassProperties = typeof(TypeTestData).GetProperties();

            // Check that the schema is of the right size.
            Assert.Equal(toyClassProperties.Length, testTypeDataset.Schema.Count);

            // Create a lookup table for the types and counts of all properties.
            var types = new Dictionary<string, Type>();
            var counts = new Dictionary<string, int>();
            foreach (var property in toyClassProperties)
            {
                if (!property.PropertyType.IsArray)
                    types[property.Name] = property.PropertyType;
                else
                {
                    // Construct a VBuffer type for the array.
                    var vBufferType = typeof(VBuffer<>);
                    Type[] typeArgs = { property.PropertyType.GetElementType() };
                    Activator.CreateInstance(property.PropertyType.GetElementType());
                    types[property.Name] = vBufferType.MakeGenericType(typeArgs);
                }

                counts[property.Name] = 0;
            }

            foreach (var column in testTypeDataset.Schema)
            {
                Assert.True(types.ContainsKey(column.Name));
                Assert.Equal(1, ++counts[column.Name]);
                Assert.Equal(types[column.Name], column.Type.RawType);
            }

            // Make sure we didn't miss any columns.
            foreach (var value in counts.Values)
                Assert.Equal(1, value);
        }

        /// <summary>
        /// Assert than two <see cref="TypeTestData"/> datasets are equal.
        /// </summary>
        /// <param name="mlContext">The ML Context.</param>
        /// <param name="data1">A <see cref="IDataView"/> of <see cref="TypeTestData"/></param>
        /// <param name="data2">A <see cref="IDataView"/> of <see cref="TypeTestData"/></param>
        public static void AssertTestTypeDatasetsAreEqual(MLContext mlContext, IDataView data1, IDataView data2)
        {
            // Confirm that they are both of the propery row type.
            AssertTypeTestDataset(data1);
            AssertTypeTestDataset(data2);

            // Validate that the two Schemas are the same.
            Common.AssertEqual(data1.Schema, data2.Schema);

            // Define how to serialize the IDataView to objects.
            var enumerable1 = mlContext.Data.CreateEnumerable<TypeTestData>(data1, true);
            var enumerable2 = mlContext.Data.CreateEnumerable<TypeTestData>(data2, true);

            AssertEqual(enumerable1, enumerable2);
        }

        /// <summary>
        /// Assert that two float arrays are equal.
        /// </summary>
        /// <param name="array1">An array of floats.</param>
        /// <param name="array2">An array of floats.</param>
        public static void AssertEqual(float[] array1, float[] array2, int precision = 6)
        {
            Assert.NotNull(array1);
            Assert.NotNull(array2);
            Assert.Equal(array1.Length, array2.Length);

            for (int i = 0; i < array1.Length; i++)
                Assert.Equal(array1[i], array2[i], precision: precision);
        }

        /// <summary>
        ///  Assert that two <see cref="DataViewSchema"/> objects are equal.
        /// </summary>
        /// <param name="schema1">A <see cref="DataViewSchema"/> object.</param>
        /// <param name="schema2">A <see cref="DataViewSchema"/> object.</param>
        public static void AssertEqual(DataViewSchema schema1, DataViewSchema schema2)
        {
            Assert.NotNull(schema1);
            Assert.NotNull(schema2);

            Assert.Equal(schema1.Count(), schema2.Count());

            foreach (var schemaPair in schema1.Zip(schema2, Tuple.Create))
            {
                Assert.Equal(schemaPair.Item1.Name, schemaPair.Item2.Name);
                Assert.Equal(schemaPair.Item1.Index, schemaPair.Item2.Index);
                Assert.Equal(schemaPair.Item1.IsHidden, schemaPair.Item2.IsHidden);
                // Can probably do a better comparison of Metadata.
                AssertEqual(schemaPair.Item1.Annotations.Schema, schemaPair.Item1.Annotations.Schema);
                Assert.True((schemaPair.Item1.Type == schemaPair.Item2.Type) ||
                    (schemaPair.Item1.Type.RawType == schemaPair.Item2.Type.RawType));
            }
        }

        /// <summary>
        /// Assert than two <see cref="TypeTestData"/> enumerables are equal.
        /// </summary>
        /// <param name="data1">An enumerable of <see cref="TypeTestData"/></param>
        /// <param name="data2">An enumerable of <see cref="TypeTestData"/></param>
        public static void AssertEqual(IEnumerable<TypeTestData> data1, IEnumerable<TypeTestData> data2)
        {
            Assert.NotNull(data1);
            Assert.NotNull(data2);
            Assert.Equal(data1.Count(), data2.Count());

            foreach (var rowPair in data1.Zip(data2, Tuple.Create))
            {
                AssertEqual(rowPair.Item1, rowPair.Item2);
            }
        }

        /// <summary>
        /// Assert that two TypeTest datasets are equal.
        /// </summary>
        /// <param name="testType1">An <see cref="TypeTestData"/>.</param>
        /// <param name="testType2">An <see cref="TypeTestData"/>.</param>
        public static void AssertEqual(TypeTestData testType1, TypeTestData testType2)
        {
            Assert.Equal(testType1.Label, testType2.Label);
            Common.AssertEqual(testType1.Features, testType2.Features);
            Assert.Equal(testType1.I1, testType2.I1);
            Assert.Equal(testType1.U1, testType2.U1);
            Assert.Equal(testType1.I2, testType2.I2);
            Assert.Equal(testType1.U2, testType2.U2);
            Assert.Equal(testType1.I4, testType2.I4);
            Assert.Equal(testType1.U4, testType2.U4);
            Assert.Equal(testType1.I8, testType2.I8);
            Assert.Equal(testType1.U8, testType2.U8);
            Assert.Equal(testType1.R4, testType2.R4);
            Assert.Equal(testType1.R8, testType2.R8);
            Assert.Equal(testType1.Tx.ToString(), testType2.Tx.ToString());
            Assert.True(testType1.Ts.Equals(testType2.Ts));
            Assert.True(testType1.Dt.Equals(testType2.Dt));
            Assert.True(testType1.Dz.Equals(testType2.Dz));
        }

        /// <summary>
        /// Check that a <see cref="AnomalyDetectionMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetrics(AnomalyDetectionMetrics metrics)
        {
            Assert.InRange(metrics.AreaUnderRocCurve, 0, 1);
            Assert.InRange(metrics.DetectionRateAtKFalsePositives, 0, 1);
        }

        /// <summary>
        /// Check that a <see cref="BinaryClassificationMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetrics(BinaryClassificationMetrics metrics)
        {
            Assert.InRange(metrics.Accuracy, 0, 1);
            Assert.InRange(metrics.AreaUnderRocCurve, 0, 1);
            Assert.InRange(metrics.AreaUnderPrecisionRecallCurve, 0, 1);
            Assert.InRange(metrics.F1Score, 0, 1);
            Assert.InRange(metrics.NegativePrecision, 0, 1);
            Assert.InRange(metrics.NegativeRecall, 0, 1);
            Assert.InRange(metrics.PositivePrecision, 0, 1);
            Assert.InRange(metrics.PositiveRecall, 0, 1);
        }

        /// <summary>
        /// Check that a <see cref="CalibratedBinaryClassificationMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetrics(CalibratedBinaryClassificationMetrics metrics)
        {
            Assert.InRange(metrics.Entropy, double.NegativeInfinity, 1);
            Assert.InRange(metrics.LogLoss, double.NegativeInfinity, 1);
            Assert.InRange(metrics.LogLossReduction, double.NegativeInfinity, 100);
            AssertMetrics(metrics as BinaryClassificationMetrics);
        }

        /// <summary>
        /// Check that a <see cref="ClusteringMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetrics(ClusteringMetrics metrics)
        {
            Assert.True(metrics.AverageDistance >= 0);
            Assert.True(metrics.DaviesBouldinIndex >= 0);
            if (!double.IsNaN(metrics.NormalizedMutualInformation))
                Assert.True(metrics.NormalizedMutualInformation >= 0 && metrics.NormalizedMutualInformation <= 1);
        }

        /// <summary>
        /// Check that a <see cref="MulticlassClassificationMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetrics(MulticlassClassificationMetrics metrics)
        {
            Assert.InRange(metrics.MacroAccuracy, 0, 1);
            Assert.InRange(metrics.MicroAccuracy, 0, 1);
            Assert.True(metrics.LogLoss >= 0);
            Assert.InRange(metrics.TopKAccuracy, 0, 1);
        }

        /// <summary>
        /// Check that a <see cref="RankingMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetrics(RankingMetrics metrics)
        {
            foreach (var dcg in metrics.DiscountedCumulativeGains)
                Assert.True(dcg >= 0);
            foreach (var ndcg in metrics.NormalizedDiscountedCumulativeGains)
                Assert.InRange(ndcg, 0, 100);
        }

        /// <summary>
        /// Check that a <see cref="RegressionMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetrics(RegressionMetrics metrics)
        {
            Assert.True(metrics.RootMeanSquaredError >= 0);
            Assert.True(metrics.MeanAbsoluteError >= 0);
            Assert.True(metrics.MeanSquaredError >= 0);
            Assert.True(metrics.RSquared <= 1);
        }

        /// <summary>
        /// Check that a <see cref="MetricStatistics"/> object is valid.
        /// </summary>
        /// <param name="metric">The <see cref="MetricStatistics"/> object.</param>
        public static void AssertMetricStatistics(MetricStatistics metric)
        {
            Assert.True(metric.StandardDeviation >= 0);
            Assert.True(metric.StandardError >= 0);
        }

        /// <summary>
        /// Check that a <see cref="RegressionMetricsStatistics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void AssertMetricsStatistics(RegressionMetricsStatistics metrics)
        {
            AssertMetricStatistics(metrics.RootMeanSquaredError);
            AssertMetricStatistics(metrics.MeanAbsoluteError);
            AssertMetricStatistics(metrics.MeanSquaredError);
            AssertMetricStatistics(metrics.RSquared);
            AssertMetricStatistics(metrics.LossFunction);
        }

        /// <summary>
        /// Assert that two float arrays are not equal.
        /// </summary>
        /// <param name="array1">An array of floats.</param>
        /// <param name="array2">An array of floats.</param>
        public static void AssertNotEqual(float[] array1, float[] array2)
        {
            Assert.NotNull(array1);
            Assert.NotNull(array2);
            Assert.Equal(array1.Length, array2.Length);

            bool mismatch = false;
            for (int i = 0; i < array1.Length; i++)
                try
                {
                    // Use Assert to test for equality rather than
                    // to roll our own float equality checker.
                    Assert.Equal(array1[i], array2[i]);
                }
                catch(EqualException)
                {
                    mismatch = true;
                    break;
                }
            Assert.True(mismatch);
        }

        /// <summary>
        /// Verify that a float array has no NaNs or infinities.
        /// </summary>
        /// <param name="array">An array of doubles.</param>
        public static void AssertFiniteNumbers(IList<float> array, int ignoreElementAt = -1)
        {
            for (int i = 0; i < array.Count; i++)
            {
                if (i == ignoreElementAt)
                    continue;
                Assert.False(float.IsNaN(array[i]));
                Assert.False(float.IsInfinity(array[i]));
            }
        }

        /// <summary>
        /// Verify that a double array has no NaNs or infinities.
        /// </summary>
        /// <param name="array">An array of doubles.</param>
        public static void AssertFiniteNumbers(IList<double> array, int ignoreElementAt = -1)
        {
            for (int i = 0; i < array.Count; i++)
            {
                if (i == ignoreElementAt)
                    continue;
                Assert.False(double.IsNaN(array[i]));
                Assert.False(double.IsInfinity(array[i]));
            }
        }
    }
}
