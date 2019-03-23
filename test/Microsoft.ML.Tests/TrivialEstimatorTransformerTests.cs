// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class TrivialEstimatorTransformerTests : BaseTestClass
    {
        public TrivialEstimatorTransformerTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        void SimpleTest()
        {
            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "NFL" , Age = 14 },
                new DataPoint() { Category = "NFL" , Age = 15 },
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLS" , Age = 14 },
            };

            // Load the data from enumerable.
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Define a TrivialEstimator and Transform data.
            var transformedData = mlContext.Transforms.CopyColumns("CopyAge", "Age").Transform(data);

            // Inspect output and check that it actually transforms data.
            var outEnum = mlContext.Data.CreateEnumerable<OutDataPoint>(transformedData, true, true);
            foreach(var outDataPoint in outEnum)
                Assert.True(outDataPoint.CopyAge != 0);
        }

        [Fact]
        void TrivialEstimatorChainsTest()
        {
            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLB" , Age = 14 },
                new DataPoint() { Category = "MLB" , Age = 15 },
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLB" , Age = 14 },
            };

            // Load the data from enumerable.
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Define a TrivialEstimatorChain by appending two TrivialEstimators.
            var trivialEstimatorChain = mlContext.Transforms.CopyColumns("CopyAge", "Age")
                .Append(mlContext.Transforms.CopyColumns("CopyCategory", "Category"));

            // Transform data directly.
            var transformedData = trivialEstimatorChain.Transform(data);

            // Inspect output and check that it actually transforms data.
            var outEnum = mlContext.Data.CreateEnumerable<OutDataPoint>(transformedData, true, true);
            foreach (var outDataPoint in outEnum)
            {
                Assert.True(outDataPoint.CopyAge != 0);
                Assert.True(outDataPoint.CopyCategory == "MLB");
            }
        }


        [Fact]
        void EstimatorChainsTest()
        {
            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLB" , Age = 14 },
                new DataPoint() { Category = "MLB" , Age = 15 },
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLB" , Age = 14 },
            };

            // Load the data from enumerable.
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // Define same TrivialEstimatorChain by appending two TrivialEstimators.
            var trivialEstimatorChain = mlContext.Transforms.CopyColumns("CopyAge", "Age")
                .Append(mlContext.Transforms.CopyColumns("CopyCategory", "Category"));
            
            // Check that this is TrivialEstimatorChain and that I can transform data directly.
            Assert.True(trivialEstimatorChain is TrivialEstimatorChain<ColumnCopyingTransformer>);
            var transformedData = trivialEstimatorChain.Transform(data);

            // Append a non trivial estimator to the chain.
            var estimatorChain = trivialEstimatorChain.Append(mlContext.Transforms.Categorical.OneHotEncoding("OneHotAge", "Age"));

            // The below gives an ERROR since the type becomes EstimatorChain as OneHotEncoding is not a trivial estimator. Uncomment to check!
            //transformedData = estimatorChain.Transform(data);
            Assert.True(estimatorChain is EstimatorChain<OneHotEncodingTransformer>);

            // Use .Fit() and .Transform() to transform data after training the transform.
            transformedData = estimatorChain.Fit(data).Transform(data);

            // Check that adding a TrivialEstimator does not bring us back to a TrivialEstimatorChain since we have a trainable transform.
            var newEstimatorChain = estimatorChain.Append(mlContext.Transforms.CopyColumns("CopyOneHotAge", "OneHotAge"));

            // The below gives an ERROR since the type stays EstimatorChain as there is non trivial estimator in the chain. Uncomment to check!
            //transformedData = newEstimatorChain.Transform(data);
            Assert.True(newEstimatorChain is EstimatorChain<ColumnCopyingTransformer>);

            // Use .Fit() and .Transform() to transform data after training the transform.
            transformedData = newEstimatorChain.Fit(data).Transform(data);

            // Check that the data has actually been transformed
            var outEnum = mlContext.Data.CreateEnumerable<OutDataPoint>(transformedData, true, true);
            foreach (var outDataPoint in outEnum)
            {
                Assert.True(outDataPoint.CopyAge != 0);
                Assert.True(outDataPoint.CopyCategory == "MLB");
                Assert.NotNull(outDataPoint.OneHotAge);
                Equals(outDataPoint.CopyOneHotAge, outDataPoint.OneHotAge);
            }

        }

        [Fact]
        void TrivialEstimatorChainWorkoutTest()
        {
            // Get a small dataset as an IEnumerable.
            var rawData = new[] {
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLB" , Age = 14 },
                new DataPoint() { Category = "MLB" , Age = 15 },
                new DataPoint() { Category = "MLB" , Age = 18 },
                new DataPoint() { Category = "MLB" , Age = 14 },
            };

            // Load the data from enumerable.
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            var trivialEstiamtorChain = new TrivialEstimatorChain<ITransformer>();
            var estimatorChain = new EstimatorChain<ITransformer>();

            var transformedData1 = trivialEstiamtorChain.Transform(data);
            var transformedData2 = estimatorChain.Fit(data).Transform(data);

            Assert.Equal(transformedData1.Schema.Count, transformedData2.Schema.Count);
            Assert.True(transformedData1.Schema.Count == 2);
        }

        private class DataPoint
        {
            public string Category { get; set; }
            public uint Age { get; set; }
        }

        private class OutDataPoint
        {
            public string Category { get; set; }
            public string CopyCategory { get; set; }
            public uint Age { get; set; }
            public uint CopyAge { get; set; }
            public float[] OneHotAge { get; set; }
            public float[] CopyOneHotAge { get; set; }
        }
    }
}
