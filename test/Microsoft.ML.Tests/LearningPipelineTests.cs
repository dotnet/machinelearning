// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
#pragma warning disable 612, 618
    public class LearningPipelineTests : BaseTestClass
    {
        public LearningPipelineTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void ConstructorDoesntThrow()
        {
            Assert.NotNull(new Legacy.LearningPipeline());
        }

        [Fact]
        public void CanAddAndRemoveFromPipeline()
        {
            var pipeline = new Legacy.LearningPipeline()
            {
                new CategoricalOneHotVectorizer("String1", "String2"),
                new ColumnConcatenator(outputColumn: "Features", "String1", "String2", "Number1", "Number2"),
                new StochasticDualCoordinateAscentRegressor()
            };
            Assert.NotNull(pipeline);
            Assert.Equal(3, pipeline.Count);

            pipeline.Remove(pipeline.ElementAt(2));
            Assert.Equal(2, pipeline.Count);

            pipeline.Add(new Legacy.Trainers.StochasticDualCoordinateAscentRegressor());
            Assert.Equal(3, pipeline.Count);
        }

        private class InputData
        {
            [LoadColumn(columnIndex: 1)]
            public string F1;
        }

        private class TransformedData
        {
#pragma warning disable 649
            [ColumnName("F1")]
            public float[] TransformedF1;
#pragma warning restore 649
        }

        [Fact]
        public void TransformOnlyPipeline()
        {
            const string _dataPath = @"..\..\Data\breast-cancer.txt";
            var pipeline = new Legacy.LearningPipeline(seed: 1, conc: 1);
            pipeline.Add(new ML.Legacy.Data.TextLoader(_dataPath).CreateFrom<InputData>(useHeader: false));
            pipeline.Add(new CategoricalHashOneHotVectorizer("F1") { HashBits = 10, Seed = 314489979, OutputKind = OneHotEncodingTransformerOutputKind.Bag });
            var model = pipeline.Train<InputData, TransformedData>();
            var predictionModel = model.Predict(new InputData() { F1 = "5" });

            Assert.NotNull(predictionModel);
            Assert.NotNull(predictionModel.TransformedF1);
            Assert.Equal(1024, predictionModel.TransformedF1.Length);

            for (int index = 0; index < 1024; index++)
                if (index == 265)
                    Assert.Equal(1, predictionModel.TransformedF1[index]);
                else
                    Assert.Equal(0, predictionModel.TransformedF1[index]);
        }

        public class Data
        {
            [ColumnName("Features")]
            [VectorType(2)]
            public float[] Features;

            [ColumnName("Label")]
            public float Label;
        }

        public class Prediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedLabel;
        }

        [Fact]
        public void NoTransformPipeline()
        {
            var data = new Data[1];
            data[0] = new Data
            {
                Features = new float[] { 0.0f, 1.0f },
                Label = 0f
            };
            var pipeline = new Legacy.LearningPipeline();
            pipeline.Add(CollectionDataSource.Create(data));
            pipeline.Add(new FastForestBinaryClassifier());
            var model = pipeline.Train<Data, Prediction>();
        }

        public class BooleanLabelData
        {
            [ColumnName("Features")]
            [VectorType(2)]
            public float[] Features;

            [ColumnName("Label")]
            public bool Label;
        }

        [Fact]
        public void BooleanLabelPipeline()
        {
            var data = new BooleanLabelData[1];
            data[0] = new BooleanLabelData
            {
                Features = new float[] { 0.0f, 1.0f },
                Label = false
            };
            var pipeline = new Legacy.LearningPipeline();
            pipeline.Add(CollectionDataSource.Create(data));
            pipeline.Add(new FastForestBinaryClassifier());
            var model = pipeline.Train<Data, Prediction>();
        }

        [Fact]
        public void AppendPipeline()
        {
            var pipeline = new Legacy.LearningPipeline();
            pipeline.Append(new CategoricalOneHotVectorizer("String1", "String2"))
                .Append(new ColumnConcatenator(outputColumn: "Features", "String1", "String2", "Number1", "Number2"))
                .Append(new StochasticDualCoordinateAscentRegressor());
            Assert.NotNull(pipeline);
            Assert.Equal(3, pipeline.Count);

            pipeline.Remove(pipeline.ElementAt(2));
            Assert.Equal(2, pipeline.Count);

            pipeline.Append(new StochasticDualCoordinateAscentRegressor());
            Assert.Equal(3, pipeline.Count);
        }
    }
#pragma warning restore 612, 618
}
