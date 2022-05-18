// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class SearchThread
    {
        private const double _eps = 1e-10;

        private readonly Flow2 _searchAlg;

        private double _costBest;
        private double _costBest1;
        private double _costBest2;
        private double _costLast;
        private double _costTotal;
        private double _objBest1;
        private double _objBest2;
        private double _speed = 0;

        public SearchThread(Flow2 searchAlgorithm)
        {
            _searchAlg = searchAlgorithm;
            _costLast = searchAlgorithm.CostIncumbent == null ? 0 : (double)searchAlgorithm.CostIncumbent;
            _costTotal = _costLast;
            _costBest = _costLast;
            _costBest1 = _costLast;
            _costBest2 = 0;
            _objBest1 = searchAlgorithm.BestObj == null ? double.PositiveInfinity : (double)searchAlgorithm.BestObj;
            _objBest2 = _objBest1;
        }

        public bool IsConverged { get => _searchAlg.IsConverged; }

        public Flow2 SearchArg { get => _searchAlg; }

        public void OnTrialComplete(int trialId, double metric, double cost)
        {
            _searchAlg.ReceiveTrialResult(trialId, metric, cost);
            _costLast = cost;
            _costTotal += cost;
            if (metric < _objBest1)
            {
                _costBest2 = _costBest1;
                _costBest1 = _costTotal;
                _objBest2 = double.IsInfinity(_objBest1) ? metric : _objBest1;
                _objBest1 = metric;
                _costBest = _costLast;
            }

            if (_objBest2 > _objBest1)
            {
                _speed = (_objBest2 - _objBest1) / (_costTotal - _costBest2 + _eps);
            }
            else
            {
                _speed = 0;
            }
        }

        public Parameter Suggest(int trialId)
        {
            return _searchAlg.Suggest(trialId);
        }
    }
}
