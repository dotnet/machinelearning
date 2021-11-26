// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class InferredPipelineTests : BaseTestClass
    {
        public InferredPipelineTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void InferredPipelinesHashTest()
        {
            var context = new MLContext(1);
            var columnInfo = new ColumnInformation();

            // test same learners with no hyperparameters have the same hash code
            var trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo);
            var trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo);
            var transforms1 = new List<SuggestedTransform>();
            var transforms2 = new List<SuggestedTransform>();
            var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, false);
            var inferredPipeline2 = new SuggestedPipeline(transforms2, new List<SuggestedTransform>(), trainer2, context, false);
            Assert.Equal(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // test same learners with hyperparameters set vs empty hyperparameters have different hash codes
            var hyperparams1 = new ParameterSet(new List<IParameterValue>() { new LongParameterValue("NumberOfLeaves", 2) });
            trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo, hyperparams1);
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo);
            inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, false);
            inferredPipeline2 = new SuggestedPipeline(transforms2, new List<SuggestedTransform>(), trainer2, context, false);
            Assert.NotEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // same learners with different hyperparameters
            hyperparams1 = new ParameterSet(new List<IParameterValue>() { new LongParameterValue("NumberOfLeaves", 2) });
            var hyperparams2 = new ParameterSet(new List<IParameterValue>() { new LongParameterValue("NumberOfLeaves", 6) });
            trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo, hyperparams1);
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo, hyperparams2);
            inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, false);
            inferredPipeline2 = new SuggestedPipeline(transforms2, new List<SuggestedTransform>(), trainer2, context, false);
            Assert.NotEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // same learners with same transforms
            trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo);
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo);
            transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            transforms2 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, false);
            inferredPipeline2 = new SuggestedPipeline(transforms2, new List<SuggestedTransform>(), trainer2, context, false);
            Assert.Equal(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // same transforms with different learners
            trainer1 = new SuggestedTrainer(context, new SdcaLogisticRegressionBinaryExtension(), columnInfo);
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), columnInfo);
            transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            transforms2 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, false);
            inferredPipeline2 = new SuggestedPipeline(transforms2, new List<SuggestedTransform>(), trainer2, context, false);
            Assert.NotEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());
        }
    }
}
