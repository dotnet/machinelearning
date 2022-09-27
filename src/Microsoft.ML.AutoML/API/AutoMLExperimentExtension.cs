// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Runtime;
using static Microsoft.ML.DataOperationsCatalog;

namespace Microsoft.ML.AutoML
{
    public static class AutoMLExperimentExtension
    {
        /// <summary>
        /// Set train and validation dataset for <see cref="AutoMLExperiment"/>. This will make <see cref="AutoMLExperiment"/> uses <paramref name="train"/>
        /// to train a model, and use <paramref name="validation"/> to evaluate the model.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="train">dataset for training a model.</param>
        /// <param name="validation">dataset for validating a model during training.</param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetDataset(this AutoMLExperiment experiment, IDataView train, IDataView validation)
        {
            var datasetManager = new TrainTestDatasetManager()
            {
                TrainDataset = train,
                TestDataset = validation
            };

            experiment.ServiceCollection.AddSingleton<IDatasetManager>(datasetManager);
            experiment.ServiceCollection.AddSingleton(datasetManager);

            return experiment;
        }

        /// <summary>
        /// Set train and validation dataset for <see cref="AutoMLExperiment"/>. This will make <see cref="AutoMLExperiment"/> uses <see cref="TrainTestData.TrainSet"/> from <paramref name="trainValidationSplit"/>
        /// to train a model, and use <see cref="TrainTestData.TestSet"/> from <paramref name="trainValidationSplit"/> to evaluate the model.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="trainValidationSplit">a <see cref="TrainTestData"/> for train and validation.</param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetDataset(this AutoMLExperiment experiment, TrainTestData trainValidationSplit)
        {
            return experiment.SetDataset(trainValidationSplit.TrainSet, trainValidationSplit.TestSet);
        }

        /// <summary>
        /// Set cross-validation dataset for <see cref="AutoMLExperiment"/>. This will make <see cref="AutoMLExperiment"/> use n=<paramref name="fold"/> cross-validation split on <paramref name="dataset"/>
        /// to train and evaluate a model.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="dataset">dataset for cross-validation split.</param>
        /// <param name="fold"></param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetDataset(this AutoMLExperiment experiment, IDataView dataset, int fold = 10)
        {
            var datasetManager = new CrossValidateDatasetManager()
            {
                Dataset = dataset,
                Fold = fold,
            };

            experiment.ServiceCollection.AddSingleton<IDatasetManager>(datasetManager);
            experiment.ServiceCollection.AddSingleton(datasetManager);

            return experiment;
        }

        /// <summary>
        /// Set <see cref="BinaryMetricManager"/> as evaluation manager for <see cref="AutoMLExperiment"/>. This will make
        /// <see cref="AutoMLExperiment"/> uses <paramref name="metric"/> as evaluation metric.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="metric">evaluation metric.</param>
        /// <param name="labelColumn">label column.</param>
        /// <param name="predictedColumn">predicted column.</param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetBinaryClassificationMetric(this AutoMLExperiment experiment, BinaryClassificationMetric metric, string labelColumn = "label", string predictedColumn = "PredictedLabel")
        {
            var metricManager = new BinaryMetricManager(metric, labelColumn, predictedColumn);
            return experiment.SetEvaluateMetric(metricManager);
        }

        /// <summary>
        /// Set <see cref="MultiClassMetricManager"/> as evaluation manager for <see cref="AutoMLExperiment"/>. This will make
        /// <see cref="AutoMLExperiment"/> uses <paramref name="metric"/> as evaluation metric.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="metric">evaluation metric.</param>
        /// <param name="labelColumn">label column.</param>
        /// <param name="predictedColumn">predicted column.</param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetMulticlassClassificationMetric(this AutoMLExperiment experiment, MulticlassClassificationMetric metric, string labelColumn = "label", string predictedColumn = "PredictedLabel")
        {
            var metricManager = new MultiClassMetricManager()
            {
                Metric = metric,
                PredictedColumn = predictedColumn,
                LabelColumn = labelColumn,
            };

            return experiment.SetEvaluateMetric(metricManager);
        }

        /// <summary>
        /// Set <see cref="RegressionMetricManager"/> as evaluation manager for <see cref="AutoMLExperiment"/>. This will make
        /// <see cref="AutoMLExperiment"/> uses <paramref name="metric"/> as evaluation metric.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="metric">evaluation metric.</param>
        /// <param name="labelColumn">label column.</param>
        /// <param name="scoreColumn">score column.</param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetRegressionMetric(this AutoMLExperiment experiment, RegressionMetric metric, string labelColumn = "Label", string scoreColumn = "Score")
        {
            var metricManager = new RegressionMetricManager()
            {
                Metric = metric,
                ScoreColumn = scoreColumn,
                LabelColumn = labelColumn,
            };

            return experiment.SetEvaluateMetric(metricManager);
        }

        /// <summary>
        /// Set <paramref name="pipeline"/> for training. This also make <see cref="AutoMLExperiment"/> uses <see cref="SweepablePipelineRunner"/>
        /// , <see cref="MLContextMonitor"/> and <see cref="EciCostFrugalTuner"/> for automl traininng as well.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="pipeline"><see cref="SweepablePipeline"/></param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetPipeline(this AutoMLExperiment experiment, SweepablePipeline pipeline)
        {
            experiment.AddSearchSpace(AutoMLExperiment.PipelineSearchspaceName, pipeline.SearchSpace);
            experiment.ServiceCollection.AddSingleton(pipeline);

            experiment.SetTrialRunner<SweepablePipelineRunner>();
            experiment.SetMonitor<MLContextMonitor>();
            experiment.SetTuner<EciCostFrugalTuner>();

            return experiment;
        }

        public static AutoMLExperiment SetPerformanceMonitor(this AutoMLExperiment experiment, int checkIntervalInMilliseconds = 1000)
        {
            experiment.SetPerformanceMonitor((service) =>
            {
                var channel = service.GetService<IChannel>();

                return new DefaultPerformanceMonitor(channel, checkIntervalInMilliseconds);
            });

            return experiment;
        }

        private static AutoMLExperiment SetEvaluateMetric<TEvaluateMetricManager>(this AutoMLExperiment experiment, TEvaluateMetricManager metricManager)
            where TEvaluateMetricManager : class, IEvaluateMetricManager
        {
            experiment.ServiceCollection.AddSingleton<IMetricManager>(metricManager);
            experiment.ServiceCollection.AddSingleton<IEvaluateMetricManager>(metricManager);

            return experiment;
        }
    }
}
