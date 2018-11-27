﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Legacy.Models
{
    public sealed partial class BinaryClassificationEvaluator
    {
        /// <summary>
        /// Computes the quality metrics for the PredictionModel using the specified data set.
        /// </summary>
        /// <param name="model">
        /// The trained PredictionModel to be evaluated.
        /// </param>
        /// <param name="testData">
        /// The test data that will be predicted and used to evaluate the model.
        /// </param>
        /// <returns>
        /// A BinaryClassificationMetrics instance that describes how well the model performed against the test data.
        /// </returns>
        public BinaryClassificationMetrics Evaluate(PredictionModel model, ILearningPipelineLoader testData)
        {
            var environment = new MLContext();
            environment.CheckValue(model, nameof(model));
            environment.CheckValue(testData, nameof(testData));

            Experiment experiment = environment.CreateExperiment();

            ILearningPipelineStep testDataStep = testData.ApplyStep(previousStep: null, experiment);
            if (!(testDataStep is ILearningPipelineDataStep testDataOutput))
            {
                throw environment.Except($"The {nameof(ILearningPipelineLoader)} did not return a {nameof(ILearningPipelineDataStep)} from ApplyStep.");
            }

            var datasetScorer = new DatasetTransformScorer
            {
                Data = testDataOutput.Data
            };
            DatasetTransformScorer.Output scoreOutput = experiment.Add(datasetScorer);

            Data = scoreOutput.ScoredData;
            Output evaluteOutput = experiment.Add(this);

            experiment.Compile();

            experiment.SetInput(datasetScorer.TransformModel, model.PredictorModel);
            testData.SetInput(environment, experiment);

            experiment.Run();

            IDataView overallMetrics = experiment.GetOutput(evaluteOutput.OverallMetrics);
            if (overallMetrics == null)
            {
                throw environment.Except($"Could not find OverallMetrics in the results returned in {nameof(BinaryClassificationEvaluator)} Evaluate.");
            }

            IDataView confusionMatrix = experiment.GetOutput(evaluteOutput.ConfusionMatrix);
            if (confusionMatrix == null)
            {
                throw environment.Except($"Could not find ConfusionMatrix in the results returned in {nameof(BinaryClassificationEvaluator)} Evaluate.");
            }

            var metric = BinaryClassificationMetrics.FromMetrics(environment, overallMetrics, confusionMatrix);

            if (metric.Count != 1)
                throw environment.Except($"Exactly one metric set was expected but found {metric.Count} metrics");

            return metric[0];
        }
    }
}
