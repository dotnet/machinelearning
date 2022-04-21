// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.SearchSpace;
using FlamlParameters = System.Collections.Generic.Dictionary<string, double>;

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
            this._searchAlg = searchAlgorithm;
            this._costLast = searchAlgorithm.CostIncumbent == null ? 0 : (double)searchAlgorithm.CostIncumbent;
            this._costTotal = this._costLast;
            this._costBest = this._costLast;
            this._costBest1 = this._costLast;
            this._costBest2 = 0;
            this._objBest1 = searchAlgorithm.BestObj == null ? double.PositiveInfinity : (double)searchAlgorithm.BestObj;
            this._objBest2 = this._objBest1;
        }

        public bool IsConverged { get => this._searchAlg.IsConverged; }

        public Flow2 SearchArg { get => this._searchAlg; }

        public void OnTrialComplete(int trialId, double metric, double cost)
        {
            this._searchAlg.ReceiveTrialResult(trialId, metric, cost);
            this._costLast = cost;
            this._costTotal += cost;
            if (metric < this._objBest1)
            {
                this._costBest2 = this._costBest1;
                this._costBest1 = this._costTotal;
                this._objBest2 = double.IsInfinity(this._objBest1) ? metric : this._objBest1;
                this._objBest1 = metric;
                this._costBest = this._costLast;
            }

            if (this._objBest2 > this._objBest1)
            {
                this._speed = (this._objBest2 - this._objBest1) / (this._costTotal - this._costBest2 + _eps);
            }
            else
            {
                this._speed = 0;
            }
        }

        public Parameter Suggest(int trialId)
        {
            return this._searchAlg.Suggest(trialId);
        }
    }
}
