// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.AutoML.CodeGen;
using Newtonsoft.Json;
using static Microsoft.ML.AutoML.AutoMLExperiment;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// propose sweepable estimator pipeline from a group of candidates using eci in flaml (https://arxiv.org/abs/1911.04706)
    /// </summary>
    internal class PipelineProposer : ISavableProposer
    {
        private readonly Dictionary<EstimatorType, double> _estimatorCost;
        private Dictionary<string, double> _learnerInitialCost;

        // total time spent on last best error for each learner.
        private Dictionary<string, double> _k1;

        // total time spent on second last best error for each learner.
        private Dictionary<string, double> _k2;

        // last best error for each learner.
        private Dictionary<string, double> _e1;

        // second last best error for each learner.
        private Dictionary<string, double> _e2;

        // flaml ECI
        private Dictionary<string, double> _eci;
        private double _globalBestError;

        private readonly Random _rand;
        private MultiModelPipeline _multiModelPipeline;

        public PipelineProposer(AutoMLExperimentSettings settings)
        {
            // this cost is used to initialize eci when started, the smaller the number, the less cost this trainer will use at start, and more likely it will be
            // picked.
            this._estimatorCost = new Dictionary<EstimatorType, double>()
            {
                { EstimatorType.LightGbmRegression, 0.788 },
                { EstimatorType.FastTreeRegression, 0.382 },
                { EstimatorType.FastForestRegression, 0.374 },
                { EstimatorType.SdcaRegression, 0.566 },
                { EstimatorType.FastTreeTweedieRegression, 0.401 },
                { EstimatorType.LbfgsPoissonRegressionRegression, 4.73 },
                { EstimatorType.FastForestOva, 4.283 },
                { EstimatorType.FastTreeOva, 3.701 },
                { EstimatorType.LightGbmMulti, 14.765 },
                { EstimatorType.SdcaMaximumEntropyMulti, 1.129 },
                { EstimatorType.SdcaLogisticRegressionOva, 3.16 },
                { EstimatorType.LbfgsMaximumEntropyMulti, 7.980 },
                { EstimatorType.LbfgsLogisticRegressionOva, 11.513 },
                { EstimatorType.LightGbmBinary, 14.765 },
                { EstimatorType.FastTreeBinary, 3.701 },
                { EstimatorType.FastForestBinary, 4.283 },
                { EstimatorType.SdcaLogisticRegressionBinary, 3.16 },
                { EstimatorType.LbfgsLogisticRegressionBinary, 11.513 },
                { EstimatorType.ForecastBySsa, 1 },
                { EstimatorType.ImageClassificationMulti, 1 },
                { EstimatorType.MatrixFactorization, 1 },
            };
            this._rand = new Random(settings.Seed ?? 0);

            this._multiModelPipeline = null;
        }

        public TrialSettings Propose(TrialSettings settings)
        {
            this._multiModelPipeline = settings.ExperimentSettings.Pipeline;
            this._learnerInitialCost = _multiModelPipeline.PipelineIds.ToDictionary(kv => kv, kv => this.GetEstimatedCostForPipeline(kv, this._multiModelPipeline));
            var pipelineIds = this._multiModelPipeline.PipelineIds;

            if (this._eci == null)
            {
                // initialize eci with the estimated cost and always start from pipeline which has lowest cost.
                this._eci = pipelineIds.ToDictionary(kv => kv, kv => this.GetEstimatedCostForPipeline(kv, this._multiModelPipeline));
                settings.Schema = this._eci.OrderBy(kv => kv.Value).First().Key;
            }
            else
            {
                var probabilities = pipelineIds.Select(id => this._eci[id]).ToArray();
                probabilities = ArrayMath.Inverse(probabilities);
                probabilities = ArrayMath.Normalize(probabilities);

                // sample
                var randdouble = this._rand.NextDouble();
                var sum = 0.0;
                // selected pipeline id index
                int i;

                for (i = 0; i != pipelineIds.Length; ++i)
                {
                    sum += ((double[])probabilities)[i];
                    if (sum > randdouble)
                    {
                        break;
                    }
                }

                settings.Schema = pipelineIds[i];
            }

            settings.Pipeline = this._multiModelPipeline.BuildSweepableEstimatorPipeline(settings.Schema);
            return settings;
        }

        public void SaveStatusToFile(string fileName)
        {
            using (var writer = new FileStream(fileName, FileMode.Create))
            {
                this.SaveStatusToStream(writer);
            }
        }

        public void SaveStatusToStream(Stream stream)
        {
            var status = new Status()
            {
                K1 = this._k1,
                K2 = this._k2,
                E1 = this._e1,
                E2 = this._e2,
                Eci = this._eci,
                GlobalBestError = this._globalBestError,
            };

            using (var fileWriter = new StreamWriter(stream))
            {
                var json = JsonConvert.SerializeObject(status);
                fileWriter.Write(json);
            }
        }

        public void LoadStatusFromFile(string fileName)
        {
            if (File.Exists(fileName))
            {
                var json = File.ReadAllText(fileName);
                var status = JsonConvert.DeserializeObject<Status>(json);
                this._k1 = status.K1;
                this._k2 = status.K2;
                this._e1 = status.E1;
                this._e2 = status.E2;
                this._eci = status.Eci;
                this._globalBestError = status.GlobalBestError;
            }
        }

        public void Update(TrialSettings parameter, TrialResult result)
        {
            var schema = parameter.Schema;
            var error = this.CaculateError(result.Metric, result.TrialSettings.ExperimentSettings.EvaluateMetric.IsMaximize);
            var duration = result.DurationInMilliseconds / 1000;
            var pipelineIds = this._multiModelPipeline.PipelineIds;
            var isSuccess = duration != 0;

            // if k1 is null, it means this is the first completed trial.
            // in that case, initialize k1, k2, e1, e2 in the following way:
            // k1: for every learner, k1[l] = c * duration where c is a ratio defined in learnerInitialCost
            // k2: k2 = k1, which indicates the hypothesis that it costs the same time for learners to reach the next break through.
            // e1: current error
            // e2: 1.001*e1

            if (isSuccess)
            {
                if (this._k1 == null)
                {
                    this._k1 = pipelineIds.ToDictionary(id => id, id => duration * this._learnerInitialCost[id] / this._learnerInitialCost[schema]);
                    this._k2 = this._k1.ToDictionary(kv => kv.Key, kv => kv.Value);
                    this._e1 = pipelineIds.ToDictionary(id => id, id => error);
                    this._e2 = pipelineIds.ToDictionary(id => id, id => 1.05 * error);
                    this._globalBestError = error;
                }
                else if (error >= this._e1[schema])
                {
                    // if error is larger than current best error, which means there's no improvements for
                    // the last trial with the current learner.
                    // In that case, simply increase the total spent time since the last best error for that learner.
                    this._k1[schema] += duration;
                }
                else
                {
                    // there's an improvement.
                    // k2 <= k1 && e2 <= e1, and update k1, e2.
                    this._k2[schema] = this._k1[schema];
                    this._k1[schema] = duration;
                    this._e2[schema] = this._e1[schema];
                    this._e1[schema] = error;

                    // update global best error as well
                    if (error < this._globalBestError)
                    {
                        this._globalBestError = error;
                    }
                }

                // update eci
                var eci1 = Math.Max(this._k1[schema], this._k2[schema]);
                var estimatorCostForBreakThrough = 2 * (error - this._globalBestError) / ((this._e2[schema] - this._e1[schema]) / (this._k2[schema] + this._k1[schema]));
                this._eci[schema] = Math.Max(eci1, estimatorCostForBreakThrough);
            }
            else
            {
                // double eci of current trial twice of maxium ecis.
                this._eci[schema] = this._eci.Select(kv => kv.Value).Max() * 2;
            }

            // normalize eci
            var sum = this._eci.Select(x => x.Value).Sum();
            this._eci = this._eci.Select(x => (x.Key, x.Value / sum)).ToDictionary(x => x.Key, x => x.Item2);

            // TODO
            // save k1,k2,e1,e2,eci,bestError to training configuration
            return;
        }

        private double CaculateError(double loss, bool isMaximize)
        {
            return isMaximize ? 1 - loss : loss;
        }

        private double GetEstimatedCostForPipeline(string kv, MultiModelPipeline multiModelPipeline)
        {
            var entity = Entity.FromExpression(kv);

            var estimatorTypes = entity.ValueEntities().Where(v => v is StringEntity s && s.Value != "Nil")
                                         .Select(v =>
                                         {
                                             var s = v as StringEntity;
                                             var estimator = multiModelPipeline.Estimators[s.Value];
                                             return estimator.EstimatorType;
                                         });

            var res = 1;

            foreach (var estimatorType in estimatorTypes)
            {
                if (this._estimatorCost.ContainsKey(estimatorType))
                {
                    return this._estimatorCost[estimatorType];
                }
            }

            return res;
        }

        public class Status
        {
            public Dictionary<string, double> K1 { get; set; }

            public Dictionary<string, double> K2 { get; set; }

            public Dictionary<string, double> E1 { get; set; }

            public Dictionary<string, double> E2 { get; set; }

            public Dictionary<string, double> Eci { get; set; }

            public double GlobalBestError { get; set; }
        }
    }
}
