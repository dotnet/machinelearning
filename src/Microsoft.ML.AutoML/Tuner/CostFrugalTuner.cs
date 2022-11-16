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
        private readonly Flow2 _localSearch;
        private readonly Dictionary<int, SearchThread> _searchThreadPool = new Dictionary<int, SearchThread>();
        private int _currentThreadId;
        private readonly Dictionary<int, int> _trialProposedBy = new Dictionary<int, int>();

        private readonly double[] _lsBoundMax;
        private readonly double[] _lsBoundMin;
        private bool _initUsed = false;
        private double _bestLoss;

        public CostFrugalTuner(AutoMLExperiment.AutoMLExperimentSettings settings, ITrialResultManager trialResultManager = null)
            : this(settings.SearchSpace, settings.SearchSpace.SampleFromFeatureSpace(settings.SearchSpace.Default), trialResultManager.GetAllTrialResults(), settings.Seed)
        {
        }

        public CostFrugalTuner(SearchSpace.SearchSpace searchSpace, Parameter initValue = null, IEnumerable<TrialResult> trialResults = null, int? seed = null)
        {
            _searchSpace = searchSpace;
            _currentThreadId = 0;
            _lsBoundMin = _searchSpace.MappingToFeatureSpace(initValue);
            _lsBoundMax = _searchSpace.MappingToFeatureSpace(initValue);
            _initUsed = false;
            _bestLoss = double.MaxValue;
            if (seed is int s)
            {
                _rng = new RandomNumberGenerator(s);
            }
            _localSearch = new Flow2(searchSpace, initValue, true, rng: _rng);
            if (trialResults != null)
            {
                foreach (var trial in trialResults)
                {
                    Update(trial);
                }
            }
        }

        public Parameter Propose(TrialSettings settings)
        {
            var trialId = settings.TrialId;
            Parameter param;
            if (_initUsed)
            {
                var searchThread = _searchThreadPool[_currentThreadId];
                param = searchThread.Suggest(trialId);
                _trialProposedBy[trialId] = _currentThreadId;
            }
            else
            {
                param = _searchSpace.SampleFromFeatureSpace(CreateInitConfigFromAdmissibleRegion());
                _trialProposedBy[trialId] = _currentThreadId;
            }

            return param;
        }

        public Parameter BestConfig { get; set; }

        public void Update(TrialResult result)
        {
            var trialId = result.TrialSettings.TrialId;
            var parameter = result.TrialSettings.Parameter;
            var loss = result.Loss;

            if (loss < _bestLoss)
            {
                BestConfig = parameter;
                _bestLoss = loss;
            }

            var cost = result.DurationInMilliseconds;
            int threadId = _trialProposedBy.ContainsKey(trialId) ? _trialProposedBy[trialId] : _currentThreadId;
            if (_searchThreadPool.Count == 0)
            {
                _searchThreadPool[_currentThreadId] = _localSearch.CreateSearchThread(parameter, loss, cost);
                _initUsed = true;
                UpdateAdmissibleRegion(_searchSpace.MappingToFeatureSpace(parameter));
            }
            else
            {
                _searchThreadPool[threadId].OnTrialComplete(parameter, loss, cost);
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
