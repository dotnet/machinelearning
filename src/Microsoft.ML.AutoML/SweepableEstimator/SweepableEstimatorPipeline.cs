// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    [JsonConverter(typeof(SweepableEstimatorPipelineConverter))]
    public class SweepableEstimatorPipeline
    {
        private readonly List<SweepableEstimator> _estimators;

        public SweepableEstimatorPipeline()
        {
            _estimators = new List<SweepableEstimator>();
            Parameter = Parameter.CreateNestedParameter();
        }

        internal SweepableEstimatorPipeline(IEnumerable<SweepableEstimator> estimators)
        {
            _estimators = estimators.ToList();
            Parameter = Parameter.CreateNestedParameter();
            int i = 0;
            foreach (var e in estimators)
            {
                Parameter[i.ToString()] = e.Parameter;
                i++;
            }
        }

        internal SweepableEstimatorPipeline(IEnumerable<SweepableEstimator> estimators, Parameter parameter)
        {
            _estimators = estimators.ToList();
            Parameter = parameter;
            int i = 0;
            foreach (var e in estimators)
            {
                e.Parameter = parameter[i.ToString()];
                i++;
            }
        }

        public SearchSpace.SearchSpace SearchSpace
        {
            get
            {
                var searchSpace = new SearchSpace.SearchSpace();
                var kvPairs = _estimators.Select((e, i) => new KeyValuePair<string, SearchSpace.SearchSpace>(i.ToString(), e.SearchSpace));
                foreach (var kv in kvPairs)
                {
                    if (kv.Value != null)
                    {
                        searchSpace.Add(kv.Key, kv.Value);
                    }
                }

                return searchSpace;
            }
        }

        public IEnumerable<SweepableEstimator> Estimators { get => _estimators; }

        public Parameter Parameter { get; set; }

        public SweepableEstimatorPipeline Append(SweepableEstimator estimator)
        {
            return new SweepableEstimatorPipeline(_estimators.Concat(new[] { estimator }));
        }

        public EstimatorChain<ITransformer> BuildTrainingPipeline(MLContext context, Parameter parameter)
        {
            Parameter = parameter;
            var pipeline = new EstimatorChain<ITransformer>();

            for (int i = 0; i != _estimators.Count(); ++i)
            {
                var ssName = i.ToString();
                pipeline = pipeline.Append(_estimators[i].BuildFromOption(context, parameter[ssName]));
            }

            return pipeline;
        }

        public override string ToString()
        {
            var estimatorName = _estimators.Select(e => e.EstimatorType.ToString());
            return string.Join("=>", estimatorName);
        }
    }
}
