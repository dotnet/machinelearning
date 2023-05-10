// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Microsoft.ML.Fairlearn.AutoML
{
    /// <summary>
    /// An internal class that holds the gridLimit value to conduct gridsearch.
    /// Needed to pass the value into the AutoMLExperiment as a singleton
    /// </summary>
    internal class GridLimit
    {
        public float Value { get; set; }
    }
    /// <summary>
    /// An extension class used to add more options to the Fairlearn girdsearch experiment
    /// </summary>
    public static class AutoMLExperimentExtension
    {
        public static AutoMLExperiment SetBinaryClassificationMoment(this AutoMLExperiment experiment, ClassificationMoment moment)
        {
            experiment.ServiceCollection.AddSingleton(moment);

            return experiment;
        }

        public static AutoMLExperiment SetGridLimit(this AutoMLExperiment experiment, float gridLimit)
        {
            var gridLimitObject = new GridLimit();
            gridLimitObject.Value = gridLimit;
            experiment.ServiceCollection.AddSingleton(gridLimitObject);
            experiment.SetTuner<CostFrugalWithLambdaTunerFactory>();

            return experiment;
        }

        public static AutoMLExperiment SetBinaryClassificationMetricWithFairLearn(
            this AutoMLExperiment experiment,
            string labelColumn,
            string predictedColumn,
            string sensitiveColumnName,
            string exampleWeightColumnName,
            float gridLimit = 10f,
            bool negativeAllowed = true)
        {
            experiment.ServiceCollection.AddSingleton<ClassificationMoment>((serviceProvider) =>
            {
                var datasetManager = serviceProvider.GetRequiredService<TrainValidateDatasetManager>();
                var moment = new UtilityParity();
                var sensitiveFeature = DataFrameColumn.Create("group_id", datasetManager.TrainDataset.GetColumn<string>(sensitiveColumnName));
                var label = DataFrameColumn.Create("label", datasetManager.TrainDataset.GetColumn<bool>(labelColumn));
                moment.LoadData(datasetManager.TrainDataset, label, sensitiveFeature);
                var lambdaSearchSpace = Utilities.GenerateBinaryClassificationLambdaSearchSpace(moment, gridLimit, negativeAllowed);
                experiment.AddSearchSpace("_lambda_search_space", lambdaSearchSpace);

                return moment;
            });

            experiment.SetTrialRunner((serviceProvider) =>
            {
                var context = serviceProvider.GetRequiredService<MLContext>();
                var moment = serviceProvider.GetRequiredService<ClassificationMoment>();
                var datasetManager = serviceProvider.GetRequiredService<TrainValidateDatasetManager>();
                var pipeline = serviceProvider.GetRequiredService<SweepablePipeline>();
                return new GridSearchTrailRunner(context, datasetManager.TrainDataset, datasetManager.ValidateDataset, labelColumn, sensitiveColumnName, pipeline, moment);
            });
            experiment.SetRandomSearchTuner();

            return experiment;
        }
    }
}
