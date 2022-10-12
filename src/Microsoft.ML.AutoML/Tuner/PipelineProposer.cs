// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace.Option;
using Newtonsoft.Json;
using Tensorflow;
using static Microsoft.ML.AutoML.AutoMLExperiment;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// propose sweepable estimator pipeline from a group of candidates using eci in flaml (https://arxiv.org/abs/1911.04706)
    /// </summary>
    internal class PipelineProposer
    {
        private readonly Dictionary<EstimatorType, double> _estimatorCost;
        private readonly Dictionary<string, double> _learnerInitialCost;

        // total time spent on last best error for each learner.
        private Dictionary<string, double> _k1;

        // total time spent on second last best error for each learner.
        private Dictionary<string, double> _k2;

        // last best error for each learner.
        private Dictionary<string, double> _e1;

        // second last best error for each learner.
        private Dictionary<string, double> _e2;

        // flaml ECI
        private readonly Dictionary<string, double> _eci;
        private double _globalBestError;

        private readonly Random _rand;
        private readonly SweepablePipeline _sweepablePipeline;
        private readonly string[] _pipelineSchemas;
        private bool _initialized = false;

        public PipelineProposer(SweepablePipeline sweepablePipeline, AutoMLExperimentSettings settings)
        {
            // this cost is used to initialize eci when started, the smaller the number, the less cost this trainer will use at start, and more likely it will be
            // picked.
            _estimatorCost = new Dictionary<EstimatorType, double>()
            {
                { EstimatorType.LightGbmRegression, 0.788 },
                { EstimatorType.FastTreeRegression, 0.382 },
                { EstimatorType.FastForestRegression, 0.374 },
                { EstimatorType.SdcaRegression, 0.566 },
                { EstimatorType.FastTreeTweedieRegression, 0.401 },
                { EstimatorType.LbfgsPoissonRegressionRegression, 4.73 },
                { EstimatorType.FastForestOva, 4.283 },
                { EstimatorType.FastTreeOva, 3.701 },
                { EstimatorType.LightGbmMulti, 4.765 },
                { EstimatorType.SdcaMaximumEntropyMulti, 10.129 },
                { EstimatorType.SdcaLogisticRegressionOva, 13.16 },
                { EstimatorType.LbfgsMaximumEntropyMulti, 7.980 },
                { EstimatorType.LbfgsLogisticRegressionOva, 11.513 },
                { EstimatorType.LightGbmBinary, 4.765 },
                { EstimatorType.FastTreeBinary, 3.701 },
                { EstimatorType.FastForestBinary, 4.283 },
                { EstimatorType.SdcaLogisticRegressionBinary, 13.16 },
                { EstimatorType.LbfgsLogisticRegressionBinary, 11.513 },
                { EstimatorType.ForecastBySsa, 1 },
                { EstimatorType.ImageClassificationMulti, 1 },
                { EstimatorType.MatrixFactorization, 1 },
            };

            _sweepablePipeline = sweepablePipeline;

            if (settings.Seed.HasValue)
            {
                _rand = new Random(settings.Seed.Value);
            }
            else
            {
                _rand = new Random();
            }

            _pipelineSchemas = _sweepablePipeline.Schema.ToTerms().Select(t => t.ToString()).ToArray();
            _learnerInitialCost = _pipelineSchemas.ToDictionary(kv => kv, kv => GetEstimatedCostForPipeline(kv, _sweepablePipeline));
            // initialize eci with the estimated cost and always start from pipeline which has lowest cost.
            _eci = _pipelineSchemas.ToDictionary(kv => kv, kv => GetEstimatedCostForPipeline(kv, _sweepablePipeline));
        }

        public string ProposeSearchSpace()
        {
            string schema;
            if (!_initialized)
            {
                schema = _eci.OrderBy(kv => kv.Value).First().Key;
                _initialized = true;
            }
            else
            {
                var probabilities = _pipelineSchemas.Select(id => _eci[id]).ToArray();
                probabilities = ArrayMath.Inverse(probabilities);
                probabilities = ArrayMath.Normalize(probabilities);

                // sample
                var randdouble = _rand.NextDouble();
                var sum = 0.0;
                // selected pipeline id index
                int i;

                for (i = 0; i != _pipelineSchemas.Length; ++i)
                {
                    sum += probabilities[i];
                    if (sum > randdouble)
                    {
                        break;
                    }
                }

                schema = _pipelineSchemas[i];
            }

            return schema;
        }

        public void Update(TrialResult result, string schema)
        {
            var loss = result.Loss;
            var duration = result.DurationInMilliseconds / 1000;
            var isSuccess = duration != 0;

            // if k1 is null, it means this is the first completed trial.
            // in that case, initialize k1, k2, e1, e2 in the following way:
            // k1: for every learner, k1[l] = c * duration where c is a ratio defined in learnerInitialCost
            // k2: k2 = k1, which indicates the hypothesis that it costs the same time for learners to reach the next break through.
            // e1: current error
            // e2: e1 + 1.05 * |e1|

            if (isSuccess)
            {
                if (_k1 == null)
                {
                    _k1 = _pipelineSchemas.ToDictionary(id => id, id => duration * _learnerInitialCost[id] / _learnerInitialCost[schema]);
                    _k2 = _k1.ToDictionary(kv => kv.Key, kv => kv.Value);
                    _e1 = _pipelineSchemas.ToDictionary(id => id, id => loss);
                    _e2 = _pipelineSchemas.ToDictionary(id => id, id => loss + 0.05 * Math.Abs(loss));
                    _globalBestError = loss;
                }
                else if (loss >= _e1[schema])
                {
                    // if error is larger than current best error, which means there's no improvements for
                    // the last trial with the current learner.
                    // In that case, simply increase the total spent time since the last best error for that learner.
                    _k1[schema] += duration;
                }
                else
                {
                    // there's an improvement.
                    // k2 <= k1 && e2 <= e1, and update k1, e2.
                    _k2[schema] = _k1[schema];
                    _k1[schema] = duration;
                    _e2[schema] = _e1[schema];
                    _e1[schema] = loss;

                    // update global best error as well
                    if (loss < _globalBestError)
                    {
                        _globalBestError = loss;
                    }
                }

                // update eci
                var eci1 = Math.Max(_k1[schema], _k2[schema]);
                var estimatorCostForBreakThrough = (2 * (loss - _globalBestError) + double.Epsilon) / ((_e2[schema] - _e1[schema]) / (_k2[schema] + _k1[schema]) + double.Epsilon);
                _eci[schema] = Math.Max(eci1, estimatorCostForBreakThrough);
            }
            else
            {
                // double eci of current trial twice of maxium ecis.
                _eci[schema] = _eci.Select(kv => kv.Value).Max() * 2;
            }

            _initialized = true;

            return;
        }

        private double GetEstimatedCostForPipeline(string schema, SweepablePipeline pipeline)
        {
            var entity = Entity.FromExpression(schema);

            var estimatorTypes = entity.ValueEntities().Where(v => v is StringEntity s && s.Value != "Nil")
                                         .Select(v =>
                                         {
                                             var s = v as StringEntity;
                                             var estimator = pipeline.Estimators[s.Value];
                                             return estimator.EstimatorType;
                                         });

            var res = 1;

            foreach (var estimatorType in estimatorTypes)
            {
                if (_estimatorCost.ContainsKey(estimatorType))
                {
                    return _estimatorCost[estimatorType];
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
