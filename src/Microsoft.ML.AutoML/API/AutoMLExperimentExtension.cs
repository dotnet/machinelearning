// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Runtime;
using Newtonsoft.Json;
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

        /// <summary>
        /// Set <see cref="SmacTuner"/> as tuner for hyper-parameter optimization. The performance of smac is in a large extend determined 
        /// by <paramref name="numberOfTrees"/>, <paramref name="nMinForSpit"/> and <paramref name="splitRatio"/>, which are used to fit smac's inner 
        /// regressor.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        /// <param name="numberOfTrees">number of regression trees when fitting random forest.</param>
        /// <param name="fitModelEveryNTrials">re-fit random forests in smac for every N trials.</param>
        /// <param name="numberInitialPopulation">Number of points to use for random initialization.</param>
        /// <param name="splitRatio">split ratio for fitting random forest in smac.</param>
        /// <param name="nMinForSpit">minimum number of data points required to be in a node if it is to be split further for fitting random forest in smac.</param>
        /// <param name="localSearchParentCount">Number of search parents to use for local search in maximizing EI acquisition function.</param>
        /// <param name="numRandomEISearchConfigurations">Number of random configurations when maximizing EI acquisition function.</param>
        /// <param name="numNeighboursForNumericalParams">Number of neighbours to sample from when applying one-step mutation for generating new parameters.</param>
        /// <param name="epsilon">the threshold to exit during maximizing EI acquisition function.</param>
        /// <returns></returns>
        public static AutoMLExperiment SetSmacTuner(
            this AutoMLExperiment experiment,
            int numberInitialPopulation = 20,
            int fitModelEveryNTrials = 10,
            int numberOfTrees = 10,
            int nMinForSpit = 2,
            float splitRatio = 0.8f,
            int localSearchParentCount = 5,
            int numRandomEISearchConfigurations = 5000,
            double epsilon = 1e-5,
            int numNeighboursForNumericalParams = 4)
        {
            experiment.SetTuner((service) =>
            {
                var channel = service.GetRequiredService<IChannel>();
                var settings = service.GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
                var context = service.GetRequiredService<MLContext>();
                var smac = new SmacTuner(context, settings.SearchSpace, numberInitialPopulation, fitModelEveryNTrials, numberOfTrees, nMinForSpit, splitRatio, localSearchParentCount, numRandomEISearchConfigurations, epsilon, numNeighboursForNumericalParams, settings.Seed, channel);

                return smac;
            });

            return experiment;
        }

        /// <summary>
        /// Set <see cref="CostFrugalTuner"/> as tuner for hyper-parameter optimization.
        /// </summary>
        /// <param name="experiment"></param>
        /// <returns></returns>
        public static AutoMLExperiment SetCostFrugalTuner(this AutoMLExperiment experiment)
        {
            experiment.SetTuner((service) =>
            {
                var settings = service.GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
                var cfo = new CostFrugalTuner(settings);

                return cfo;
            });

            return experiment;
        }

        /// <summary>
        /// set <see cref="RandomSearchTuner"/> as tuner for hyper parameter optimization. If <paramref name="seed"/> is provided, it will use that 
        /// seed to initialize <see cref="RandomSearchTuner"/>. Otherwise, <see cref="AutoMLExperiment.AutoMLExperimentSettings.Seed"/> will be used.
        /// </summary>
        /// <param name="seed"></param>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        public static AutoMLExperiment SetRandomSearchTuner(this AutoMLExperiment experiment, int? seed = null)
        {
            experiment.SetTuner((service) =>
            {
                var settings = service.GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
                seed = seed ?? settings.Seed;
                var tuner = new RandomSearchTuner(settings.SearchSpace, seed);

                return tuner;
            });

            return experiment;
        }

        /// <summary>
        /// set <see cref="GridSearchTuner"/> as tuner for hyper parameter optimization.
        /// </summary>
        /// <param name="step">step size for numeric option.</param>
        /// <param name="experiment"><see cref="AutoMLExperiment"/></param>
        public static AutoMLExperiment SetGridSearchTuner(this AutoMLExperiment experiment, int step = 10)
        {
            experiment.SetTuner((service) =>
            {
                var settings = service.GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
                var tuner = new GridSearchTuner(settings.SearchSpace, step);

                return tuner;
            });

            return experiment;
        }

        /// <summary>
        /// Set checkpoint folder for <see cref="AutoMLExperiment"/>. The checkpoint folder will be used to save
        /// temporary output, run history and many other stuff which will be used for restoring training process 
        /// from last checkpoint and continue training.
        /// </summary>
        /// <param name="experiment"><see cref="AutoMLExperiment"/>.</param>
        /// <param name="folder">checkpoint folder. This folder will be created if not exist.</param>
        /// <returns><see cref="AutoMLExperiment"/></returns>
        public static AutoMLExperiment SetCheckpoint(this AutoMLExperiment experiment, string folder)
        {
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }

            experiment.ServiceCollection.AddSingleton<ITrialResultManager>(serviceProvider =>
            {
                var channel = serviceProvider.GetRequiredService<IChannel>();
                var settings = serviceProvider.GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();

                // todo
                // pull out the logic of calculating experiment id into a stand-alone service.
                var metricManager = serviceProvider.GetService<IMetricManager>();
                var csvFileName = "trialResults";
                csvFileName += $"-{settings.SearchSpace.GetHashCode()}";
                if (metricManager is IMetricManager)
                {
                    csvFileName += $"-{metricManager.MetricName}";
                }
                csvFileName += ".csv";

                var csvFilePath = Path.Combine(folder, csvFileName);
                var trialResultManager = new CsvTrialResultManager(csvFilePath, settings.SearchSpace, channel);

                return trialResultManager;
            });

            return experiment;
        }

        /// <summary>
        /// set <see cref="EciCostFrugalTuner"/> as tuner for hyper-parameter optimization. This tuner only works with search space from <see cref="SweepablePipeline"/>.
        /// </summary>
        /// <param name="experiment"></param>
        /// <returns></returns>
        public static AutoMLExperiment SetEciCostFrugalTuner(this AutoMLExperiment experiment)
        {
            experiment.SetTuner<EciCostFrugalTuner>();

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
