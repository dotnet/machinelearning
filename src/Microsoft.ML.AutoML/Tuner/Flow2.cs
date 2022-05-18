// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.AutoMLService;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// An implementation of Flow2 from https://www.aaai.org/AAAI21Papers/AAAI-10128.WuQ.pdf
    /// </summary>
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
            _searchSpace = searchSpace;
            _minimize = minimizeMode;

            _initConfig = initValue;
            _bestConfig = _initConfig;
            _incumbent = _searchSpace.MappingToFeatureSpace(_bestConfig);
            _dim = _searchSpace.Count;
            _numAllowed4Incumbent = 2 * _dim;
            _step = _stepSize * Math.Sqrt(_dim);
            _stepUpperBound = Math.Sqrt(_dim);
            _convergeSpeed = convergeSpeed;
            if (_step > _stepUpperBound)
            {
                _step = _stepUpperBound;
            }
        }

        public bool IsConverged
        {
            get => _step < _stepLowerBound;
        }

        public Parameter BestConfig
        {
            get => _bestConfig;
        }

        public SearchThread CreateSearchThread(Parameter config, double metric, double cost)
        {
            var flow2 = new Flow2(_searchSpace, config, _minimize, convergeSpeed: _convergeSpeed);
            flow2.BestObj = metric;
            flow2.CostIncumbent = cost;
            return new SearchThread(flow2);
        }

        public Parameter Suggest(int trialId)
        {
            _numAllowed4Incumbent -= 1;
            double[] move;
            if (_directionTried != null)
            {
                move = ArrayMath.Sub(_incumbent, _directionTried);

                _directionTried = null;
            }
            else
            {
                _directionTried = RandVectorSphere();
                move = ArrayMath.Add(_incumbent, _directionTried);
            }

            move = Project(move);
            var config = _searchSpace.SampleFromFeatureSpace(move);
            _proposedBy[trialId] = _incumbent;
            _configs[trialId] = config;
            return config;
        }

        public void ReceiveTrialResult(int trialId, double metric, double cost)
        {
            _trialCount += 1;
            if (BestObj == null || metric < BestObj)
            {
                BestObj = metric;
                _bestConfig = _configs[trialId];
                _incumbent = _searchSpace.MappingToFeatureSpace(_bestConfig);
                CostIncumbent = cost;
                _costComplete4Incumbent = 0;
                _numAllowed4Incumbent = 2 * _dim;
                _proposedBy.Clear();
                _step *= _convergeSpeed;
                _step = Math.Min(_step, _stepUpperBound);
                _directionTried = null;
                return;
            }
            else
            {
                _costComplete4Incumbent += cost;
                if (_numAllowed4Incumbent == 0)
                {
                    _numAllowed4Incumbent = 2;
                    if (!IsConverged)
                    {
                        _step /= _convergeSpeed;
                    }
                }
            }
        }

        private double[] RandVectorSphere()
        {
            double[] vec = _rng.Normal(0, 1, _searchSpace.FeatureSpaceDim);
            double mag = ArrayMath.Norm(vec);
            vec = ArrayMath.Mul(vec, _step / mag);

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
