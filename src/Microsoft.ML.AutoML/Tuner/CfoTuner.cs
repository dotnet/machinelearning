// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.AutoMLService;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class CfoTuner : ITuner
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

        public CfoTuner(SearchSpace.SearchSpace searchSpace, Parameter initValue = null, bool minimizeMode = true)
        {
            this._searchSpace = searchSpace;
            this._minimize = minimizeMode;

            this._localSearch = new Flow2(searchSpace, initValue, true);
            this._currentThreadId = 0;
            this._lsBoundMin = this._searchSpace.MappingToFeatureSpace(initValue);
            this._lsBoundMax = this._searchSpace.MappingToFeatureSpace(initValue);
            this._initUsed = false;
            this._bestMetric = double.MaxValue;
        }

        public Parameter Propose(TrialSettings settings)
        {
            var trialId = settings.TrialId;
            if (this._initUsed)
            {
                var searchThread = this._searchThreadPool[this._currentThreadId];
                this._configs[trialId] = this._searchSpace.MappingToFeatureSpace(searchThread.Suggest(trialId));
                this._trialProposedBy[trialId] = this._currentThreadId;
            }
            else
            {
                this._configs[trialId] = this.CreateInitConfigFromAdmissibleRegion();
                this._trialProposedBy[trialId] = this._currentThreadId;
            }

            var param = this._configs[trialId];
            return this._searchSpace.SampleFromFeatureSpace(param);
        }

        public Parameter BestConfig { get; set; }

        public void Update(TrialResult result)
        {
            var trialId = result.TrialSettings.TrialId;
            var metric = result.Metric;
            metric = this._minimize ? metric : -metric;
            if (metric < this._bestMetric)
            {
                this.BestConfig = this._searchSpace.SampleFromFeatureSpace(this._configs[trialId]);
                this._bestMetric = metric;
            }

            var cost = result.DurationInMilliseconds;
            int threadId = this._trialProposedBy[trialId];
            if (this._searchThreadPool.Count == 0)
            {
                var initParameter = this._searchSpace.SampleFromFeatureSpace(this._configs[trialId]);
                this._searchThreadPool[this._currentThreadId] = this._localSearch.CreateSearchThread(initParameter, metric, cost);
                this._initUsed = true;
                this.UpdateAdmissibleRegion(this._configs[trialId]);
            }
            else
            {
                this._searchThreadPool[threadId].OnTrialComplete(trialId, metric, cost);
                if (this._searchThreadPool[threadId].IsConverged)
                {
                    this._searchThreadPool.Remove(threadId);
                    this._currentThreadId += 1;
                    this._initUsed = false;
                }
            }
        }

        private void UpdateAdmissibleRegion(double[] config)
        {
            for (int i = 0; i != config.Length; ++i)
            {
                if (config[i] < this._lsBoundMin[i])
                {
                    this._lsBoundMin[i] = config[i];
                    continue;
                }

                if (config[i] > this._lsBoundMax[i])
                {
                    this._lsBoundMax[i] = config[i];
                    continue;
                }
            }
        }

        private double[] CreateInitConfigFromAdmissibleRegion()
        {
            var res = new double[this._lsBoundMax.Length];
            for (int i = 0; i != this._lsBoundMax.Length; ++i)
            {
                res[i] = this._rng.Uniform(this._lsBoundMin[i], this._lsBoundMax[i]);
            }

            return res;
        }
    }
}
