// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Sweeper;

[assembly: LoadableClass(typeof(NelderMeadSweeper), typeof(NelderMeadSweeper.Arguments), typeof(SignatureSweeper),
    "Nelder Mead Sweeper", "NelderMeadSweeper", "NelderMead", "NM")]

namespace Microsoft.ML.Runtime.Sweeper
{
    public sealed class NelderMeadSweeper : ISweeper
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Swept parameters", ShortName = "p", SignatureType = typeof(SignatureSweeperParameter))]
            public IComponentFactory<IValueGenerator>[] SweptParameters;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The sweeper used to get the initial results.", ShortName = "init", SignatureType = typeof(SignatureSweeperFromParameterList))]
            public IComponentFactory<IValueGenerator[], ISweeper> FirstBatchSweeper;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random number generator for the first batch sweeper", ShortName = "seed")]
            public int RandomSeed;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Simplex diameter for stopping", ShortName = "dstop")]
            public Float StoppingSimplexDiameter = (Float)0.001;

            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "If iteration point is outside parameter definitions, should it be projected?", ShortName = "project")]
            public bool ProjectInbounds = true;

            #region Core algorithm constants
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Reflection parameter", ShortName = "dr")]
            public Float DeltaReflection = (Float)1.0;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Expansion parameter", ShortName = "de")]
            public Float DeltaExpansion = (Float)1.5;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Inside contraction parameter", ShortName = "dic")]
            public Float DeltaInsideContraction = -(Float)0.5;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Outside contraction parameter", ShortName = "doc")]
            public Float DeltaOutsideContraction = (Float)0.5;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Shrinkage parameter", ShortName = "ds")]
            public Float GammaShrink = (Float)0.5;
            #endregion
        }

        private enum OptimizationStage
        {
            NeedReflectionPoint,
            WaitingForReflectionResult,
            WaitingForExpansionResult,
            WaitingForOuterContractionResult,
            WaitingForInnerContractionResult,
            WaitingForReductionResult,
            Done
        }

        private readonly ISweeper _initSweeper;
        private readonly Arguments _args;

        private SortedList<IRunResult, Float[]> _simplexVertices;
        private readonly int _dim;

        private OptimizationStage _stage;
        private readonly List<KeyValuePair<ParameterSet, Float[]>> _pendingSweeps;
        private Queue<KeyValuePair<ParameterSet, Float[]>> _pendingSweepsNotSubmitted;
        private KeyValuePair<IRunResult, Float[]> _lastReflectionResult;

        private KeyValuePair<IRunResult, Float[]> _worst;
        private KeyValuePair<IRunResult, Float[]> _secondWorst;
        private KeyValuePair<IRunResult, Float[]> _best;

        private Float[] _centroid;

        private readonly List<IValueGenerator> _sweepParameters;

        public NelderMeadSweeper(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckUserArg(-1 < args.DeltaInsideContraction, nameof(args.DeltaInsideContraction), "Must be greater than -1");
            env.CheckUserArg(args.DeltaInsideContraction < 0, nameof(args.DeltaInsideContraction), "Must be less than 0");
            env.CheckUserArg(0 < args.DeltaOutsideContraction, nameof(args.DeltaOutsideContraction), "Must be greater than 0");
            env.CheckUserArg(args.DeltaReflection > args.DeltaOutsideContraction, nameof(args.DeltaReflection), "Must be greater than " + nameof(args.DeltaOutsideContraction));
            env.CheckUserArg(args.DeltaExpansion > args.DeltaReflection, nameof(args.DeltaExpansion), "Must be greater than " + nameof(args.DeltaReflection));
            env.CheckUserArg(0 < args.GammaShrink && args.GammaShrink < 1, nameof(args.GammaShrink), "Must be between 0 and 1");
            env.CheckValue(args.FirstBatchSweeper, nameof(args.FirstBatchSweeper) , "First Batch Sweeper Contains Null Value");

            _args = args;

            _sweepParameters = new List<IValueGenerator>();
            foreach (var sweptParameter in args.SweptParameters)
            {
                var parameter = sweptParameter.CreateComponent(env);
                // REVIEW: ideas about how to support discrete values:
                // 1. assign each discrete value a random number (1-n) to make mirroring possible
                // 2. each time we need to mirror a discrete value, sample from the remaining value
                // 2.1. make the sampling non-uniform by learning "weights" for the different discrete values based on
                // the metric values that we get when using them. (E.g. if, for a given discrete value, we get a bad result,
                // we lower its weight, but if we get a good result we increase its weight).
                var parameterNumeric = parameter as INumericValueGenerator;
                env.CheckUserArg(parameterNumeric != null, nameof(args.SweptParameters), "Nelder-Mead sweeper can only sweep over numeric parameters");
                _sweepParameters.Add(parameterNumeric);
            }

            _initSweeper = args.FirstBatchSweeper.CreateComponent(env, _sweepParameters.ToArray());
            _dim = _sweepParameters.Count;
            env.CheckUserArg(_dim > 1, nameof(args.SweptParameters), "Nelder-Mead sweeper needs at least two parameters to sweep over.");

            _simplexVertices = new SortedList<IRunResult, Float[]>(new SimplexVertexComparer());
            _stage = OptimizationStage.NeedReflectionPoint;
            _pendingSweeps = new List<KeyValuePair<ParameterSet, Float[]>>();
            _pendingSweepsNotSubmitted = new Queue<KeyValuePair<ParameterSet, Float[]>>();
        }

        public ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            int numSweeps = Math.Min(maxSweeps, _dim + 1 - _simplexVertices.Count);

            if (previousRuns == null)
                return _initSweeper.ProposeSweeps(numSweeps, previousRuns);

            foreach (var run in previousRuns)
                Contracts.Check(run != null);

            foreach (var run in previousRuns)
            {
                if (_simplexVertices.Count == _dim + 1)
                    break;

                if (!_simplexVertices.ContainsKey(run))
                    _simplexVertices.Add(run, ParameterSetAsFloatArray(run.ParameterSet));

                if (_simplexVertices.Count == _dim + 1)
                    ComputeExtremes();
            }

            if (_simplexVertices.Count < _dim + 1)
            {
                numSweeps = Math.Min(maxSweeps, _dim + 1 - _simplexVertices.Count);
                return _initSweeper.ProposeSweeps(numSweeps, previousRuns);
            }

            switch (_stage)
            {
            case OptimizationStage.NeedReflectionPoint:
                _pendingSweeps.Clear();
                var nextPoint = GetNewPoint(_centroid, _worst.Value, _args.DeltaReflection);
                if (OutOfBounds(nextPoint) && _args.ProjectInbounds)
                {
                    // if the reflection point is out of bounds, get the inner contraction point.
                    nextPoint = GetNewPoint(_centroid, _worst.Value, _args.DeltaInsideContraction);
                    _stage = OptimizationStage.WaitingForInnerContractionResult;
                }
                else
                    _stage = OptimizationStage.WaitingForReflectionResult;
                _pendingSweeps.Add(new KeyValuePair<ParameterSet, Float[]>(FloatArrayAsParameterSet(nextPoint), nextPoint));
                if (previousRuns.Any(runResult => runResult.ParameterSet.Equals(_pendingSweeps[0].Key)))
                {
                    _stage = OptimizationStage.WaitingForReductionResult;
                    _pendingSweeps.Clear();
                    if (!TryGetReductionPoints(maxSweeps, previousRuns))
                    {
                        _stage = OptimizationStage.Done;
                        return null;
                    }
                    return _pendingSweeps.Select(kvp => kvp.Key).ToArray();
                }
                return new ParameterSet[] { _pendingSweeps[0].Key };

            case OptimizationStage.WaitingForReflectionResult:
                Contracts.Assert(_pendingSweeps.Count == 1);
                _lastReflectionResult = FindRunResult(previousRuns)[0];
                if (_secondWorst.Key.CompareTo(_lastReflectionResult.Key) < 0 && _lastReflectionResult.Key.CompareTo(_best.Key) <= 0)
                {
                    // the reflection result is better than the second worse, but not better than the best
                    UpdateSimplex(_lastReflectionResult.Key, _lastReflectionResult.Value);
                    goto case OptimizationStage.NeedReflectionPoint;
                }

                if (_lastReflectionResult.Key.CompareTo(_best.Key) > 0)
                {
                    // the reflection result is the best so far
                    nextPoint = GetNewPoint(_centroid, _worst.Value, _args.DeltaExpansion);
                    if (OutOfBounds(nextPoint) && _args.ProjectInbounds)
                    {
                        // if the expansion point is out of bounds, get the inner contraction point.
                        nextPoint = GetNewPoint(_centroid, _worst.Value, _args.DeltaInsideContraction);
                        _stage = OptimizationStage.WaitingForInnerContractionResult;
                    }
                    else
                        _stage = OptimizationStage.WaitingForExpansionResult;
                }
                else if (_lastReflectionResult.Key.CompareTo(_worst.Key) > 0)
                {
                    // other wise, get results for the outer contraction point.
                    nextPoint = GetNewPoint(_centroid, _worst.Value, _args.DeltaOutsideContraction);
                    _stage = OptimizationStage.WaitingForOuterContractionResult;
                }
                else
                {
                    // other wise, reflection result is not better than worst, get results for the inner contraction point
                    nextPoint = GetNewPoint(_centroid, _worst.Value, _args.DeltaInsideContraction);
                    _stage = OptimizationStage.WaitingForInnerContractionResult;
                }
                _pendingSweeps.Clear();
                _pendingSweeps.Add(new KeyValuePair<ParameterSet, Float[]>(FloatArrayAsParameterSet(nextPoint), nextPoint));
                if (previousRuns.Any(runResult => runResult.ParameterSet.Equals(_pendingSweeps[0].Key)))
                {
                    _stage = OptimizationStage.WaitingForReductionResult;
                    _pendingSweeps.Clear();
                    if (!TryGetReductionPoints(maxSweeps, previousRuns))
                    {
                        _stage = OptimizationStage.Done;
                        return null;
                    }
                    return _pendingSweeps.Select(kvp => kvp.Key).ToArray();
                }
                return new ParameterSet[] { _pendingSweeps[0].Key };

            case OptimizationStage.WaitingForExpansionResult:
                Contracts.Assert(_pendingSweeps.Count == 1);
                var expansionResult = FindRunResult(previousRuns)[0].Key;
                if (expansionResult.CompareTo(_lastReflectionResult.Key) > 0)
                {
                    // expansion point is better than reflection point
                    UpdateSimplex(expansionResult, _pendingSweeps[0].Value);
                    goto case OptimizationStage.NeedReflectionPoint;
                }
                // reflection point is better than expansion point
                UpdateSimplex(_lastReflectionResult.Key, _lastReflectionResult.Value);
                goto case OptimizationStage.NeedReflectionPoint;

            case OptimizationStage.WaitingForOuterContractionResult:
                Contracts.Assert(_pendingSweeps.Count == 1);
                var outerContractionResult = FindRunResult(previousRuns)[0].Key;
                if (outerContractionResult.CompareTo(_lastReflectionResult.Key) > 0)
                {
                    // outer contraction point is better than reflection point
                    UpdateSimplex(outerContractionResult, _pendingSweeps[0].Value);
                    goto case OptimizationStage.NeedReflectionPoint;
                }
                // get the reduction points
                _stage = OptimizationStage.WaitingForReductionResult;
                _pendingSweeps.Clear();
                if (!TryGetReductionPoints(maxSweeps, previousRuns))
                {
                    _stage = OptimizationStage.Done;
                    return null;
                }
                return _pendingSweeps.Select(kvp => kvp.Key).ToArray();

            case OptimizationStage.WaitingForInnerContractionResult:
                Contracts.Assert(_pendingSweeps.Count == 1);
                var innerContractionResult = FindRunResult(previousRuns)[0].Key;
                if (innerContractionResult.CompareTo(_worst.Key) > 0)
                {
                    // inner contraction point is better than worst point
                    UpdateSimplex(innerContractionResult, _pendingSweeps[0].Value);
                    goto case OptimizationStage.NeedReflectionPoint;
                }
                // get the reduction points
                _stage = OptimizationStage.WaitingForReductionResult;
                _pendingSweeps.Clear();
                if (!TryGetReductionPoints(maxSweeps, previousRuns))
                {
                    _stage = OptimizationStage.Done;
                    return null;
                }
                return _pendingSweeps.Select(kvp => kvp.Key).ToArray();

            case OptimizationStage.WaitingForReductionResult:
                Contracts.Assert(_pendingSweeps.Count + _pendingSweepsNotSubmitted.Count == _dim);
                if (_pendingSweeps.Count < _dim)
                    return SubmitMoreReductionPoints(maxSweeps);
                ReplaceSimplexVertices(previousRuns);

                // if the diameter of the new simplex has become too small, stop sweeping.
                if (SimplexDiameter() < _args.StoppingSimplexDiameter)
                    return null;

                goto case OptimizationStage.NeedReflectionPoint;
            case OptimizationStage.Done:
            default:
                return null;
            }
        }

        private void UpdateSimplex(IRunResult newVertexResult, Float[] newVertex)
        {
            Contracts.Assert(_centroid != null);
            Contracts.Assert(_simplexVertices.Count == _dim + 1);

            _simplexVertices.Remove(_worst.Key);
            _simplexVertices.Add(newVertexResult, newVertex);

            ComputeExtremes();
        }

        private void ComputeExtremes()
        {
            _worst = _simplexVertices.ElementAt(0);
            _secondWorst = _simplexVertices.ElementAt(1);
            _best = _simplexVertices.ElementAt(_simplexVertices.Count - 1);
            _centroid = GetCentroid();
        }

        private Float SimplexDiameter()
        {
            Float maxDistance = Float.MinValue;

            var simplexVertices = _simplexVertices.ToArray();
            for (int i = 0; i < simplexVertices.Length; i++)
            {
                var x = simplexVertices[i].Value;
                for (int j = i + 1; j < simplexVertices.Length; j++)
                {
                    var y = simplexVertices[j].Value;
                    var dist = VectorUtils.Distance(x, y);
                    if (dist > maxDistance)
                        maxDistance = dist;
                }
            }
            return maxDistance;
        }

        private bool OutOfBounds(Float[] point)
        {
            Contracts.Assert(point.Length == _sweepParameters.Count);
            for (int i = 0; i < _sweepParameters.Count; i++)
            {
                var param = _sweepParameters[i].CreateFromNormalized(point[i]);
                if (!((INumericValueGenerator)_sweepParameters[i]).InRange(param))
                    return true;
            }
            return false;
        }

        private void ReplaceSimplexVertices(IEnumerable<IRunResult> previousRuns)
        {
            var results = FindRunResult(previousRuns);
            var newSimplexVertices = new SortedList<IRunResult, Float[]>(new SimplexVertexComparer());
            foreach (var result in results)
                newSimplexVertices.Add(result.Key, result.Value);
            newSimplexVertices.Add(_best.Key, _best.Value);
            _simplexVertices = newSimplexVertices;

            ComputeExtremes();
        }

        // given some ParameterSets, find their results.
        private List<KeyValuePair<IRunResult, Float[]>> FindRunResult(IEnumerable<IRunResult> previousRuns)
        {
            var result = new List<KeyValuePair<IRunResult, Float[]>>();
            foreach (var sweep in _pendingSweeps)
            {
                foreach (var run in previousRuns)
                {
                    if (run.ParameterSet.Equals(sweep.Key))
                    {
                        result.Add(new KeyValuePair<IRunResult, Float[]>(run, sweep.Value));
                        break;
                    }
                }
            }
            if (result.Count != _pendingSweeps.Count)
                throw Contracts.Except("previous runs do not contain results for expected runs");

            return result;
        }

        private Float[] GetCentroid()
        {
            var centroid = new Float[_dim];
            Float scale = (Float)1 / _dim;
            for (int i = 1; i < _simplexVertices.Count; i++)
                VectorUtils.AddMult(_simplexVertices.ElementAt(i).Value, centroid, scale);
            return centroid;
        }

        private Float[] GetNewPoint(Float[] centroid, Float[] worst, Float delta)
        {
            var newPoint = new Float[centroid.Length];
            VectorUtils.AddMult(centroid, newPoint, 1 + delta);
            VectorUtils.AddMult(worst, newPoint, -delta);

            return newPoint;
        }

        private bool TryGetReductionPoints(int maxSweeps, IEnumerable<IRunResult> previousRuns)
        {
            int numPoints = Math.Min(maxSweeps, _dim);
            var sortedVertices = _simplexVertices.ToArray();
            for (int i = 0; i < _simplexVertices.Count - 1; i++)
            {
                var newPoint = GetNewPoint(_secondWorst.Value, sortedVertices[i].Value, -_args.GammaShrink);
                var newParameterSet = FloatArrayAsParameterSet(newPoint);
                if (previousRuns.Any(runResult => runResult.ParameterSet.Equals(newParameterSet)))
                    return false;
                if (i < numPoints)
                    _pendingSweeps.Add(new KeyValuePair<ParameterSet, Float[]>(newParameterSet, newPoint));
                else
                    _pendingSweepsNotSubmitted.Enqueue(new KeyValuePair<ParameterSet, Float[]>(FloatArrayAsParameterSet(newPoint), newPoint));
            }
            return true;
        }

        private ParameterSet[] SubmitMoreReductionPoints(int maxSweeps)
        {
            int numPoints = Math.Min(maxSweeps, _pendingSweepsNotSubmitted.Count);

            var result = new ParameterSet[numPoints];
            for (int i = 0; i < numPoints; i++)
            {
                var point = _pendingSweepsNotSubmitted.Dequeue();
                _pendingSweeps.Add(point);
                result[i] = point.Key;
            }
            return result;
        }

        private Float[] ParameterSetAsFloatArray(ParameterSet parameterSet)
        {
            Contracts.Assert(parameterSet.Count == _sweepParameters.Count);

            var result = new List<Float>();
            for (int i = 0; i < _sweepParameters.Count; i++)
            {
                Contracts.AssertValue(parameterSet[_sweepParameters[i].Name]);
                result.Add(((INumericValueGenerator)_sweepParameters[i]).NormalizeValue(parameterSet[_sweepParameters[i].Name]));
            }

            return result.ToArray();
        }

        private ParameterSet FloatArrayAsParameterSet(Float[] array)
        {
            Contracts.Assert(array.Length == _sweepParameters.Count);

            var parameters = new List<IParameterValue>();
            for (int i = 0; i < _sweepParameters.Count; i++)
            {
                parameters.Add(_sweepParameters[i].CreateFromNormalized(array[i]));
            }

            return new ParameterSet(parameters);
        }

        private sealed class SimplexVertexComparer : IComparer<IRunResult>
        {
            public int Compare(IRunResult x, IRunResult y)
            {
                if (x.ParameterSet.Equals(y.ParameterSet))
                    return 0;
                if (x.CompareTo(y) == 0)
                    return x.ParameterSet.ToString().CompareTo(y.ParameterSet.ToString());
                return x.CompareTo(y);
            }
        }
    }
}
