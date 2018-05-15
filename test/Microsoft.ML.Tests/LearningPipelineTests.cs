// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
    public class LearningPipelineTests : BaseTestClass
    {
        public LearningPipelineTests(ITestOutputHelper output)
            : base(output)
        {

        }

        [Fact]
        public void ConstructorDoesntThrow()
        {
            Assert.NotNull(new LearningPipeline());
        }

        [Fact]
        public void CanAddAndRemoveFromPipeline()
        {
            var pipeline = new LearningPipeline()
            {
                new Transforms.CategoricalOneHotVectorizer("String1", "String2"),
                new Transforms.ColumnConcatenator(outputColumn: "Features", "String1", "String2", "Number1", "Number2"),
                new Trainers.StochasticDualCoordinateAscentRegressor()
            };
            Assert.NotNull(pipeline);
            Assert.Equal(3, pipeline.Count);

            pipeline.Remove(pipeline.ElementAt(2));
            Assert.Equal(2, pipeline.Count);

            pipeline.Add(new Trainers.StochasticDualCoordinateAscentRegressor());
            Assert.Equal(3, pipeline.Count);
        }

        class InputData
        {
            [Column(ordinal: "1")]
            public string F1;
        }

        class TransformedData
        {
            [ColumnName("F1")]
            public float[] TransformedF1;
        }

        [Fact]
        public void TransformOnlyPipeline()
        {
            const string _dataPath = @"..\..\Data\breast-cancer.txt";
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<InputData>(_dataPath, useHeader: false));
            pipeline.Add(new CategoricalHashOneHotVectorizer("F1") { HashBits = 10, Seed = 314489979, OutputKind = CategoricalTransformOutputKind.Bag });
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

            predictionModel.TransformedF1 = null;
        }
    }
}
