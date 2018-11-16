//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Tests
{
    public class PermutationFeatureImportanceTests : BaseTestPredictors
    {
        public PermutationFeatureImportanceTests(ITestOutputHelper output) : base(output)
        {
        }

        private class FeatureMetricData : IComparable
        {
            public Double Metric;
            public string Feature;

            public int CompareTo(object obj)
            {
                var that = obj as FeatureMetricData;
                if (Metric < that.Metric)
                    return -1;
                if (Metric > that.Metric)
                    return 1;
                return 0;
            }
        }

        private const string Category = "FeatureImportance";

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random and Label y is to dependant on xRand.
        /// Test verifies that xRand has the least importance: L1, L2, RMS and Loss-Fn do not change a lot when xRand is permuted.
        /// Also test checks that x2 has the biggest importance.
        /// </summary>
        [Fact]
        public void FeatureImportanceDenseSGD()
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 1000;
            var rand = new Random(10);
            Float[] yArray = new Float[numberOfInstances],
                x1Array = new Float[numberOfInstances],
                x2Array = new Float[numberOfInstances],
                x3Array = new Float[numberOfInstances],
                x4RandArray = new Float[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x2Important = rand.Next(10000);
                x2Array[i] = x2Important;
                var x3 = rand.Next(5000);
                x3Array[i] = x3;
                var x4Rand = rand.Next(1000);
                x4RandArray[i] = x4Rand;

                var noise = rand.Next(50);
                yArray[i] = (Float)(10 * x1 + 20 * x2Important + 5.5 * x3 + noise);
            }

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2Important", NumberType.Float, x2Array);
            bldr.AddColumn("X3", NumberType.Float, x3Array);
            bldr.AddColumn("X4Rand", NumberType.Float, x4RandArray);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            var srcDV = bldr.GetDataView();

            // Create data loader and SGD predictor.

            //var dataLoader = Env.CreateTransform("Concat{col=Features:X1,X2Important,X3,X4Rand}", srcDV);
            //dataLoader = Env.CreateTransform("minmax{col=Features}", dataLoader);
            //var exRoleMapped = Env.CreateExamples(dataLoader, "Features", "Label");
            //var apiPredictor = Env.TrainPredictor("OnlineGradientDescent", exRoleMapped);

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .Append(ML.Transforms.Normalize("Features"))
                .Append(ML.Regression.Trainers.OnlineGradientDescent());

            var model = pipeline.Fit(srcDV);


            // Call pfi and get IDV containing info on how badly evaluator's metrics get affected if given feature is permuted.
            var pfi = new PermutationFeatureImportanceRegression(Env);
            var results = pfi.GetImportanceMetricsMatrix(model, srcDV);

            // permuted X4Rand should have min impact on SGD metrics, X2 -- max impact
            Assert.True(results.OrderBy(r => r.metricsDelta.L1).First().featureName == "X4Rand");
            Assert.True(results.OrderByDescending(r => r.metricsDelta.L1).First().featureName == "X2Important");
            Assert.True(results.OrderBy(r => r.metricsDelta.L2).First().featureName == "X4Rand");
            Assert.True(results.OrderByDescending(r => r.metricsDelta.L2).First().featureName == "X2Important");
            Assert.True(results.OrderBy(r => r.metricsDelta.Rms).First().featureName == "X4Rand");
            Assert.True(results.OrderByDescending(r => r.metricsDelta.Rms).First().featureName == "X2Important");

            //Assert.True(deltas["L2(avg)_Delta"].Min().Feature == "X4Rand");
            //Assert.True(deltas["L2(avg)_Delta"].Max().Feature == "X2Important");
            //Assert.True(deltas["RMS(avg)_Delta"].Min().Feature == "X4Rand");
            //Assert.True(deltas["RMS(avg)_Delta"].Max().Feature == "X2Important");
            //Assert.True(deltas["Loss-fn(avg)_Delta"].Min().Feature == "X4Rand");
            //Assert.True(deltas["Loss-fn(avg)_Delta"].Max().Feature == "X2Important");
            //Assert.True(deltas["R Squared_Delta"].Min().Feature == "X2Important");
            //Assert.True(deltas["R Squared_Delta"].Max().Feature == "X4Rand");

            Done();
        }

        /// <summary>
        /// Features: x1, x2vBuff(sparce vector), x3. 
        /// y = 10x1 + 10x2vBuff + 30x3 + e.
        /// Within xBuff feature  2nd slot will be sparse most of the time.
        /// Test verifies that 2nd slot of xBuff has the least importance: L1, L2, RMS and Loss-Fn do not change a lot when this slot is permuted.
        /// Also test checks that x2 has the biggest importance.
        /// </summary>
//        [Fact]
//        public void FeatureImportanceSparseSGD()
//        {
//            // Setup synthetic dataset.
//            const int numberOfInstances = 10000;
//            var rand = new Random(10);
//            Float[] yArray = new Float[numberOfInstances],
//                x1Array = new Float[numberOfInstances],
//                x3Array = new Float[numberOfInstances];

//            VBuffer<Float>[] vbArray = new VBuffer<Float>[numberOfInstances];

//            for (var i = 0; i < numberOfInstances; i++)
//            {
//                var x1 = rand.Next(1000);
//                x1Array[i] = x1;
//                var x3Important = rand.Next(10000);
//                x3Array[i] = x3Important;

//                VBuffer<Float> vb;

//                if (i % 10 != 0)
//                {
//                    vb = new VBuffer<Float>(4, 3, new Float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 2, 3 });
//                }
//                else
//                {
//                    vb = new VBuffer<Float>(4, 4, new Float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 1, 2, 3 });
//                }

//                vbArray[i] = vb;

//                Float vbSum = 0;
//                foreach (var vbValue in vb.DenseValues())
//                {
//                    vbSum += vbValue * 10;
//                }

//                var noise = rand.Next(50);
//                yArray[i] = 10 * x1 + vbSum + 20 * x3Important + noise;
//            }

//            // Create data view.
//            var bldr = new ArrayDataViewBuilder(Env);
//            bldr.AddColumn("X1", NumberType.Float, x1Array);
//            bldr.AddColumn("X2VBuffer", NumberType.Float, vbArray);
//            bldr.AddColumn("X3Important", NumberType.Float, x3Array);
//            bldr.AddColumn("Label", NumberType.Float, yArray);
//            var srcDV = bldr.GetDataView();

//            // Create data loader and SGD predictor.
//            var dataLoader = Env.CreateTransform("Concat{col=Features:X1,X2VBuffer,X3Important}", srcDV);
//            dataLoader = Env.CreateTransform("minmax{col=Features}", dataLoader);
//            var exRoleMapped = Env.CreateExamples(dataLoader, "Features", "Label");
//            var apiPredictor = Env.TrainPredictor("OnlineGradientDescent", exRoleMapped);
//#pragma warning disable 618 // Disable warnings about accessing IPredictor from Predictor.
//            var predictor = apiPredictor.GetPredictorObject() as IPredictor;
//#pragma warning restore 618

//            // Call pfi and get IDV containing info on how badly evaluator's metrics get affected if given feature is permuted.
//            var pfi = new PermutationFeatureImportance(Env);
//            var evaluatedFeaturesCount = 0;
//            var resultView = pfi.GetImportanceMetricsMatrix(predictor, dataLoader, "Features", "Label", true, out evaluatedFeaturesCount, 0);
//            Dictionary<string, List<FeatureMetricData>> deltas = BuildMetricsDictionary(resultView);

//            // Permuted 2nd slot (f2) should have min impact on SGD metrics, X4 -- max impact.
//            //Assert.True(evaluatedFeaturesCount == exRoleMapped.Schema.Feature.Type.VectorSize);
//            Assert.True(deltas["L1(avg)_Delta"].Min().Feature == "f2");
//            Assert.True(deltas["L1(avg)_Delta"].Max().Feature == "X3Important");
//            Assert.True(deltas["L2(avg)_Delta"].Min().Feature == "f2");
//            Assert.True(deltas["L2(avg)_Delta"].Max().Feature == "X3Important");
//            Assert.True(deltas["RMS(avg)_Delta"].Min().Feature == "f2");
//            Assert.True(deltas["RMS(avg)_Delta"].Max().Feature == "X3Important");
//            Assert.True(deltas["Loss-fn(avg)_Delta"].Min().Feature == "f2");
//            Assert.True(deltas["Loss-fn(avg)_Delta"].Max().Feature == "X3Important");
//            Assert.True(deltas["R Squared_Delta"].Min().Feature == "X3Important");
//            Assert.True(deltas["R Squared_Delta"].Max().Feature == "f2");
//        }

        private Dictionary<string, List<FeatureMetricData>> BuildMetricsDictionary(IDataView view)
        {
            // Build a dictionary for easy metrics comparison
            Dictionary<string, List<FeatureMetricData>> deltas = new Dictionary<string, List<FeatureMetricData>>();
            using (var cursor = view.GetRowCursor(col => true))
            {
                var schema = view.Schema;
                var row = 0;
                while (cursor.MoveNext())
                {
                    var currentFeatureName = string.Empty;
                    Double metricValue = 0;
                    var metricName = string.Empty;
                    for (int j = 0; j < schema.ColumnCount; j++)
                    {
                        var type = schema.GetColumnType(j);
                        if (type == TextType.Instance)
                        {
                            var value = default(ReadOnlyMemory<char>);
                            cursor.GetGetter<ReadOnlyMemory<char>>(j)(ref value);
                            currentFeatureName = value.ToString();
                        }
                        else if (type == NumberType.R8)
                        {
                            cursor.GetGetter<Double>(j)(ref metricValue);
                            metricName = schema.GetColumnName(j);
                            var data = new FeatureMetricData() { Metric = metricValue };
                            if (!deltas.ContainsKey(metricName))
                            {
                                deltas.Add(metricName, new List<FeatureMetricData>());
                            }
                            deltas[metricName].Add(data);
                        }

                        // Some funky code to make test independent of the fact whether Feature names columns is first one or not.
                        if (j == schema.ColumnCount - 1)
                            foreach (var metric in deltas.Keys)
                            {
                                var resultRow = 0;
                                foreach (var d in deltas[metric])
                                {
                                    if (resultRow == row)
                                    {
                                        d.Feature = currentFeatureName;
                                    }
                                    resultRow++;
                                }
                            }
                    }

                    row++;
                }
            }

            return deltas;
        }
    }
}
