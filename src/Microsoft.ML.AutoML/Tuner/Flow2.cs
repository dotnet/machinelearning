// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Microsoft.ML.AutoMLService;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class Flow2
    {
        private const double _stepSize = 0.1;
        private const double _stepLowerBound = 0.0001;

        private readonly RandomNumberGenerator _rng = new RandomNumberGenerator();

        public double? BestObj = null;
        public double? CostIncumbent = null;
        private readonly Parameter _initConfig;
        private double _step;

        private readonly SearchSpace.SearchSpace _searchSpace;
        private readonly bool _minimize;
        private readonly double _convergeSpeed = 2;
        private Parameter _bestConfig;
        private readonly Dictionary<int, Parameter> _configs = new Dictionary<int, Parameter>();
        private double _costComplete4Incumbent = 0;
        private readonly int _dim;
        private double[] _directionTried = null;
        private double[] _incumbent;
        private int _numAllowed4Incumbent = 0;
        private readonly Dictionary<int, double[]> _proposedBy = new Dictionary<int, double[]>();
        private readonly double _stepUpperBound;
        private int _trialCount = 1;

        public Flow2(SearchSpace.SearchSpace searchSpace, Parameter initValue = null, bool minimizeMode = true, double convergeSpeed = 1.5)
        {
            this._searchSpace = searchSpace;
            this._minimize = minimizeMode;

            this._initConfig = initValue;
            this._bestConfig = this._initConfig;
            this._incumbent = this._searchSpace.MappingToFeatureSpace(this._bestConfig);
            this._dim = this._searchSpace.Count;
            this._numAllowed4Incumbent = 2 * this._dim;
            this._step = _stepSize * Math.Sqrt(this._dim);
            this._stepUpperBound = Math.Sqrt(this._dim);
            this._convergeSpeed = convergeSpeed;
            if (this._step > this._stepUpperBound)
            {
                this._step = this._stepUpperBound;
            }
        }

        public bool IsConverged
        {
            get => this._step < _stepLowerBound;
        }

        public Parameter BestConfig
        {
            get => this._bestConfig;
        }

        public SearchThread CreateSearchThread(Parameter config, double metric, double cost)
        {
            var flow2 = new Flow2(this._searchSpace, config, this._minimize, convergeSpeed: this._convergeSpeed);
            flow2.BestObj = metric;
            flow2.CostIncumbent = cost;
            return new SearchThread(flow2);
        }

        public Parameter Suggest(int trialId)
        {
            this._numAllowed4Incumbent -= 1;
            double[] move;
            if (this._directionTried != null)
            {
                move = ArrayMath.Sub(this._incumbent, this._directionTried);

                this._directionTried = null;
            }
            else
            {
                this._directionTried = this.RandVectorSphere();
                move = ArrayMath.Add(this._incumbent, this._directionTried);
            }

            move = this.Project(move);
            var config = this._searchSpace.SampleFromFeatureSpace(move);
            this._proposedBy[trialId] = this._incumbent;
            this._configs[trialId] = config;
            return config;
        }

        public void ReceiveTrialResult(int trialId, double metric, double cost)
        {
            this._trialCount += 1;
            double obj = metric;  // flipped in BlendSearch
            //double obj = minimize ? metric : -metric;
            if (this.BestObj == null || obj < this.BestObj)
            {
                this.BestObj = obj;
                this._bestConfig = this._configs[trialId];
                this._incumbent = this._searchSpace.MappingToFeatureSpace(this._bestConfig);
                this.CostIncumbent = cost;
                this._costComplete4Incumbent = 0;
                this._numAllowed4Incumbent = 2 * this._dim;
                this._proposedBy.Clear();
                this._step *= this._convergeSpeed;
                this._step = Math.Min(this._step, this._stepUpperBound);
                this._directionTried = null;
                return;
            }
            else
            {
                this._costComplete4Incumbent += cost;
                if (this._numAllowed4Incumbent == 0)
                {
                    this._numAllowed4Incumbent = 2;
                    if (!this.IsConverged)
                    {
                        this._step /= this._convergeSpeed;
                    }
                }
            }
        }

        private double[] RandVectorSphere()
        {
            double[] vec = this._rng.Normal(0, 1, this._searchSpace.FeatureSpaceDim);
            double mag = ArrayMath.Norm(vec);
            vec = ArrayMath.Mul(vec, this._step / mag);

            return vec;
        }

        private double[] Project(double[] move)
        {
            return move.Select(x =>
            {
                if (x < 0)
                {
                    x = 0;
                }
                else if (x > 1)
                {
                    x = 0.99999999;
                }

                return x;
            }).ToArray();
        }
    }
}
