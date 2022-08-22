// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using static Microsoft.ML.DataOperationsCatalog;

namespace Microsoft.ML.AutoML.API
{
    public static class AutoMLExperimentExtension
    {
        public static AutoMLExperiment SetDataset(this AutoMLExperiment experiment, IDataView train, IDataView test)
        {
            var datasetManager = new TrainTestDatasetManager()
            {
                TrainDataset = train,
                TestDataset = test
            };

            experiment.ServiceCollection.AddSingleton<IDatasetManager>(datasetManager);
            experiment.ServiceCollection.AddSingleton(datasetManager);

            return experiment;
        }

        public static AutoMLExperiment SetDataset(this AutoMLExperiment experiment, TrainTestData trainTestSplit)
        {
            return experiment.SetDataset(trainTestSplit.TrainSet, trainTestSplit.TestSet);
        }

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


        public static AutoMLExperiment SetBinaryClassificationMetric(this AutoMLExperiment experiment, BinaryClassificationMetric metric, string labelColumn = "label", string predictedColumn = "PredictedLabel")
        {
            var metricManager = new BinaryMetricManager(metric, predictedColumn, labelColumn);
            return experiment.SetEvaluateMetric(metricManager);
        }

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

        public static AutoMLExperiment SetPipeline(this AutoMLExperiment experiment, SweepablePipeline pipeline)
        {
            experiment.AddSearchSpace(AutoMLExperiment.PipelineSearchspaceName, pipeline.SearchSpace);
            experiment.ServiceCollection.AddSingleton(pipeline);

            experiment.SetTrialRunner<SweepablePipelineRunner>();
            experiment.SetMonitor<MLContextMonitor>();
            experiment.SetTuner<EciCfoTuner>();

            return experiment;
        }

        private static AutoMLExperiment SetEvaluateMetric<TEvaluateMetricManager>(this AutoMLExperiment experiment, TEvaluateMetricManager metricManager)
            where TEvaluateMetricManager : class, IEvaluateMetricManager
        {
            experiment.ServiceCollection.AddSingleton<IMetricManager>(metricManager);
            experiment.ServiceCollection.AddSingleton<IEvaluateMetricManager>(metricManager);
            experiment.SetIsMaximize(metricManager.IsMaximize);

            return experiment;
        }
    }
}
