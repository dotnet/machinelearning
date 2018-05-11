// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.TestFramework;
using System;
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
            Assert.NotNull(new LearningPipeline(seed:42));
            Assert.NotNull(new LearningPipeline(concurrency: 1));
            Assert.NotNull(new LearningPipeline(seed:42, concurrency: 1));
        }

        [Fact]
        public void CanAddAndRemoveFromPipeline()
        {
            var pipeline = new LearningPipeline(seed:42, concurrency: 1)
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

       
    }
}
