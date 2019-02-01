// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class InferredPipelineTests
    {
        [TestMethod]
        public void InferredPipelinesHashTest()
        {
            var context = new MLContext();

            // test same learners with no hyperparams have the same hash code
            var trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension());
            var trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension());
            var transforms1 = new List<SuggestedTransform>();
            var transforms2 = new List<SuggestedTransform>();
            var inferredPipeline1 = new SuggestedPipeline(transforms1, trainer1);
            var inferredPipeline2 = new SuggestedPipeline(transforms2, trainer2);
            Assert.AreEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // test same learners with hyperparams set vs empty hyperparams have different hash codes
            var hyperparams1 = new ParameterSet(new List<IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
            trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), hyperparams1);
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension());
            inferredPipeline1 = new SuggestedPipeline(transforms1, trainer1);
            inferredPipeline2 = new SuggestedPipeline(transforms2, trainer2);
            Assert.AreNotEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // same learners with different hyperparams
            hyperparams1 = new ParameterSet(new List<IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
            var hyperparams2 = new ParameterSet(new List<IParameterValue>() { new LongParameterValue("NumLeaves", 6) });
            trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), hyperparams1);
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), hyperparams2);
            inferredPipeline1 = new SuggestedPipeline(transforms1, trainer1);
            inferredPipeline2 = new SuggestedPipeline(transforms2, trainer2);
            Assert.AreNotEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // same learners with same transforms
            trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension());
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension());
            transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            transforms2 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            inferredPipeline1 = new SuggestedPipeline(transforms1, trainer1);
            inferredPipeline2 = new SuggestedPipeline(transforms2, trainer2);
            Assert.AreEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());

            // same transforms with different learners
            trainer1 = new SuggestedTrainer(context, new SdcaBinaryExtension());
            trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension());
            transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            transforms2 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
            inferredPipeline1 = new SuggestedPipeline(transforms1, trainer1);
            inferredPipeline2 = new SuggestedPipeline(transforms2, trainer2);
            Assert.AreNotEqual(inferredPipeline1.GetHashCode(), inferredPipeline2.GetHashCode());
        }
    }
}
