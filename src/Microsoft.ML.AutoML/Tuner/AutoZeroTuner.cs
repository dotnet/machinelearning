// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML.Tuner
{
    internal class AutoZeroTuner : ITuner
    {
        private readonly List<Config> _configs = new List<Config>();
        private readonly IEnumerator<Config> _configsEnumerator;
        private readonly Dictionary<string, string> _pipelineStrings;
        private readonly SweepablePipeline _sweepablePipeline;
        private readonly Dictionary<int, Config> _configLookBook = new Dictionary<int, Config>();
        private readonly string _metricName;

        public AutoZeroTuner(SweepablePipeline pipeline, AggregateTrainingStopManager aggregateTrainingStopManager, IEvaluateMetricManager evaluateMetricManager, AutoMLExperiment.AutoMLExperimentSettings settings)
        {
            _configs = LoadConfigsFromCsv();
            _sweepablePipeline = pipeline;
            _pipelineStrings = _sweepablePipeline.Schema.ToTerms().Select(t => new
            {
                schema = t.ToString(),
                pipelineString = string.Join("=>", t.ValueEntities().Select(e => _sweepablePipeline.Estimators[e.ToString()].EstimatorType)),
            }).ToDictionary(kv => kv.schema, kv => kv.pipelineString);
            _configs = evaluateMetricManager switch
            {
                BinaryMetricManager => _configs.Where(c => c.Task == "binary-classification").ToList(),
                MultiClassMetricManager => _configs.Where(c => c.Task == "multi-classification").ToList(),
                RegressionMetricManager => _configs.Where(c => c.Task == "regression").ToList(),
                _ => throw new Exception(),
            };
            _metricName = evaluateMetricManager switch
            {
                BinaryMetricManager bm => bm.Metric.ToString(),
                MultiClassMetricManager mm => mm.Metric.ToString(),
                RegressionMetricManager rm => rm.Metric.ToString(),
                _ => throw new Exception(),
            };
            _configsEnumerator = _configs.GetEnumerator();
            aggregateTrainingStopManager.AddTrainingStopManager(new MaxModelStopManager(_configs.Count, null));
        }

        private List<Config> LoadConfigsFromCsv()
        {
            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = "Microsoft.ML.AutoML.Tuner.Portfolios.json";

            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            using (StreamReader reader = new StreamReader(stream))
            {
                var json = reader.ReadToEnd();
                var res = new List<Config>();
                var rows = JsonSerializer.Deserialize<List<Rows>>(json);
                foreach (var row in rows)
                {
                    var config = new Config();
                    config.Name = row.CustomDimensionsDataset;
                    if (row.CustomDimensionsBestPipeline.Contains("OneHotEncoding"))
                    {
                        config.CatalogTransformer = "OneHotEncoding";
                    }
                    else
                    {
                        config.CatalogTransformer = "OneHotHashEncoding";
                    }

                    config.Task = row.CustomDimensionsOptionsTask;
                    var i = 0;
                    foreach (var estimator in row.CustomDimensionsBestPipeline.Split(new[] { "=>" }, StringSplitOptions.RemoveEmptyEntries))
                    {
                        if (Enum.TryParse<EstimatorType>(estimator, out var estimatorType) && estimatorType.IsTrainer())
                        {
                            config.Trainer = estimator;
                            break;
                        }
                        i++;
                    }
                    var parameter = row.CustomDimensionsParameter;
                    var schema = parameter["_pipeline_"]["_SCHEMA_"].AsType<string>();
                    var trainerName = schema.Split('*').ToArray()[i].Trim();
                    parameter = parameter["_pipeline_"][trainerName];
                    config.TrainerParameter = parameter;
                    if (config.Task == "classification")
                    {
                        if (config.Trainer.Contains("Multi") || config.Trainer.Contains("Ova"))
                        {
                            config.Task = "multi-classification";
                        }
                        else
                        {
                            config.Task = "binary-classification";
                        }
                    }


                    res.Add(config);

                }

                return res;
            }
        }

        public Parameter Propose(TrialSettings settings)
        {
            if (_configsEnumerator.MoveNext())
            {
                var config = _configsEnumerator.Current;
                IEnumerable<KeyValuePair<string, string>> pipelineSchemas = default;
                if (_pipelineStrings.Any(kv => kv.Value.Contains("OneHotHashEncoding") || kv.Value.Contains("OneHotEncoding")))
                {
                    pipelineSchemas = _pipelineStrings.Where(kv => kv.Value.Contains(config.CatalogTransformer));
                }
                else
                {
                    pipelineSchemas = _pipelineStrings;
                }

                pipelineSchemas = pipelineSchemas.Where(kv => kv.Value.Contains(config.Trainer));
                var pipelineSchema = pipelineSchemas.First().Key;
                var pipeline = _sweepablePipeline.BuildSweepableEstimatorPipeline(pipelineSchema);
                var parameter = pipeline.SearchSpace.SampleFromFeatureSpace(pipeline.SearchSpace.Default);
                var trainerEstimatorName = pipeline.Estimators.Where(kv => kv.Value.EstimatorType.IsTrainer()).First().Key;
                var label = parameter[trainerEstimatorName]["LabelColumnName"].AsType<string>();
                parameter[trainerEstimatorName] = config.TrainerParameter;
                parameter[trainerEstimatorName]["LabelColumnName"] = Parameter.FromString(label);
                settings.Parameter[AutoMLExperiment.PipelineSearchspaceName] = parameter;
                _configLookBook[settings.TrialId] = config;
                return settings.Parameter;
            }

            throw new OperationCanceledException();
        }

        public void Update(TrialResult result)
        {
        }

        class Config
        {
            /// <summary>
            /// one of OneHot, HashEncoding
            /// </summary>
            public string CatalogTransformer { get; set; }

            /// <summary>
            /// One of Lgbm, Sdca, FastTree,,,
            /// </summary>
            public string Trainer { get; set; }

            public Parameter TrainerParameter { get; set; }

            public string Task { get; set; }

            public string Name { get; set; }
        }

        class Rows
        {
            public string CustomDimensionsDataset { get; set; }

            public string CustomDimensionsOptionsPrimaryMetric { get; set; }

            public string CustomDimensionsBestPipeline { get; set; }

            public string CustomDimensionsOptionsTask { get; set; }

            public Parameter CustomDimensionsParameter { get; set; }
        }
    }
}
