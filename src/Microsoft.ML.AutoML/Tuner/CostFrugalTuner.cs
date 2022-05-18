// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.AutoMLService;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    // an implemetation of "Frugal Optimization for Cost-related Hyperparameters" : https://www.aaai.org/AAAI21Papers/AAAI-10128.WuQ.pdf
    internal class CostFrugalTuner : ITuner
    {
        private readonly RandomNumberGenerator _rng = new RandomNumberGenerator();
        private readonly SearchSpace.SearchSpace _searchSpace;
        private readonly bool _minimize;
        private readonly Flow2 _localSearch;
        private readonly Dictionary<int, SearchThread> _searchThreadPool = new Dictionary<int, SearchThread>();
        private int _currentThreadId;
        private readonly Dictionary<int, int> _trialProposedBy = new Dictionary<int, int>();

        private readonly Dictionary<int, double[]> _configs = new Dictionary<int, double[]>();
        private readonly double[] _lsBoundMax;
        private readonly double[] _lsBoundMin;
        private bool _initUsed = false;
        private double _bestMetric;

        public CostFrugalTuner(SearchSpace.SearchSpace searchSpace, Parameter initValue = null, bool minimizeMode = true)
        {
            _searchSpace = searchSpace;
            _minimize = minimizeMode;

            _localSearch = new Flow2(searchSpace, initValue, true);
            _currentThreadId = 0;
            _lsBoundMin = _searchSpace.MappingToFeatureSpace(initValue);
            _lsBoundMax = _searchSpace.MappingToFeatureSpace(initValue);
            _initUsed = false;
            _bestMetric = double.MaxValue;
        }

        public Parameter Propose(TrialSettings settings)
        {
            var trialId = settings.TrialId;
            if (_initUsed)
            {
                var searchThread = _searchThreadPool[_currentThreadId];
                _configs[trialId] = _searchSpace.MappingToFeatureSpace(searchThread.Suggest(trialId));
                _trialProposedBy[trialId] = _currentThreadId;
            }
            else
            {
                _configs[trialId] = CreateInitConfigFromAdmissibleRegion();
                _trialProposedBy[trialId] = _currentThreadId;
            }

            var param = _configs[trialId];
            return _searchSpace.SampleFromFeatureSpace(param);
        }

        public Parameter BestConfig { get; set; }

        public void Update(TrialResult result)
        {
            var trialId = result.TrialSettings.TrialId;
            var metric = result.Metric;
            metric = _minimize ? metric : -metric;
            if (metric < _bestMetric)
            {
                BestConfig = _searchSpace.SampleFromFeatureSpace(_configs[trialId]);
                _bestMetric = metric;
            }

            var cost = result.DurationInMilliseconds;
            int threadId = _trialProposedBy[trialId];
            if (_searchThreadPool.Count == 0)
            {
                var initParameter = _searchSpace.SampleFromFeatureSpace(_configs[trialId]);
                _searchThreadPool[_currentThreadId] = _localSearch.CreateSearchThread(initParameter, metric, cost);
                _initUsed = true;
                UpdateAdmissibleRegion(_configs[trialId]);
            }
            else
            {
                _searchThreadPool[threadId].OnTrialComplete(trialId, metric, cost);
                if (_searchThreadPool[threadId].IsConverged)
                {
                    _searchThreadPool.Remove(threadId);
                    _currentThreadId += 1;
                    _initUsed = false;
                }
            }
        }

        private void UpdateAdmissibleRegion(double[] config)
        {
            for (int i = 0; i != config.Length; ++i)
            {
                if (config[i] < _lsBoundMin[i])
                {
                    _lsBoundMin[i] = config[i];
                    continue;
                }

                if (config[i] > _lsBoundMax[i])
                {
                    _lsBoundMax[i] = config[i];
                    continue;
                }
            }
        }

        private double[] CreateInitConfigFromAdmissibleRegion()
        {
            var res = new double[_lsBoundMax.Length];
            for (int i = 0; i != _lsBoundMax.Length; ++i)
            {
                res[i] = _rng.Uniform(_lsBoundMin[i], _lsBoundMax[i]);
            }

            return res;
        }
    }
}
