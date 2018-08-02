// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public interface IStepSearch
    {
        void AdjustTreeOutputs(IChannel ch, RegressionTree tree, DocumentPartitioning partitioning, ScoreTracker trainingScores);
    }

    public sealed class LineSearch : IStepSearch, IFastTrainingScoresUpdate
    {
        private double _historicStepSize;
        private int _numPostbracketSteps;
        private double _minStepSize;

        public LineSearch(Test lossCalculator, int lossIndex, int numPostbracketSteps, double minStepSize)
            : this(lossCalculator, lossIndex) { _numPostbracketSteps = numPostbracketSteps; _minStepSize = minStepSize; }

        public LineSearch(Test lossCalculator, int lossIndex)
        {
            _lo = new StepScoresAndLoss(lossCalculator, lossIndex);
            _hi = new StepScoresAndLoss(lossCalculator, lossIndex);
            _left = new StepScoresAndLoss(lossCalculator, lossIndex);
            _right = new StepScoresAndLoss(lossCalculator, lossIndex);
            _historicStepSize = Math.Max(1.0, _minStepSize);
        }

        private static readonly double _phi = (1.0 + Math.Sqrt(5)) / 2;

        private static void Swap<T>(ref T a, ref T b)
        {
            T t = a;
            a = b;
            b = t;
        }

        private static void Rotate<T>(ref T a, ref T b, ref T c)
        {
            T t = a;
            a = b;
            b = c;
            c = t;
        }

        private sealed class StepScoresAndLoss
        {
            private readonly Test _lossCalculator;
            private readonly int _lossIndex;

            public StepScoresAndLoss(Test lossCalculator, int lossIndex)
            {
                _lossCalculator = lossCalculator;
                _lossIndex = lossIndex;
            }

            private RegressionTree _tree;
            private DocumentPartitioning _partitioning;
            private ScoreTracker _previousScores;

            public void Initialize(RegressionTree tree, DocumentPartitioning partitioning, ScoreTracker previousScores)
            {
                _tree = tree;
                _partitioning = partitioning;
                _previousScores = previousScores;
            }

            private double _step;

            public ScoreTracker Scores;
            public TestResult Loss;

            public double Step
            {
                get { return _step; }
                set
                {
                    if (Scores == null || Scores.Dataset != _previousScores.Dataset)
                        Scores = new ScoreTracker(_previousScores);
                    _step = value;
                    Scores.Initialize(_previousScores, _tree, _partitioning, _step);
                    Loss = _lossCalculator.ComputeTests(Scores.Scores).ToList()[_lossIndex];
                }
            }
        }

        private StepScoresAndLoss _lo;
        private StepScoresAndLoss _left;
        private StepScoresAndLoss _right;
        private StepScoresAndLoss _hi;

        public void AdjustTreeOutputs(IChannel ch, RegressionTree tree, DocumentPartitioning partitioning,
            ScoreTracker previousScores)
        {
            _lo.Initialize(tree, partitioning, previousScores);
            _hi.Initialize(tree, partitioning, previousScores);
            _left.Initialize(tree, partitioning, previousScores);
            _right.Initialize(tree, partitioning, previousScores);

            _lo.Step = _historicStepSize / _phi;
            _left.Step = _historicStepSize;

            if (_lo.Loss.CompareTo(_left.Loss) == 1) // backtrack
            {
                do
                {
                    Rotate(ref _hi, ref _left, ref _lo);
                    if (_hi.Step <= _minStepSize)
                        goto FINISHED;
                    _lo.Step = _left.Step / _phi;
                } while (_lo.Loss.CompareTo(_left.Loss) == 1);
            }
            else // extend (or stay)
            {
                _hi.Step = _historicStepSize * _phi;
                while (_hi.Loss.CompareTo(_left.Loss) == 1)
                {
                    Rotate(ref _lo, ref _left, ref _hi);
                    _hi.Step = _left.Step * _phi;
                }
            }

            if (_numPostbracketSteps > 0)
            {
                _right.Step = _lo.Step + (_hi.Step - _lo.Step) / _phi;
                for (int step = 0; step < _numPostbracketSteps; ++step)
                {
                    int cmp = _right.Loss.CompareTo(_left.Loss);
                    if (cmp == 0)
                        break;

                    if (cmp == 1) // move right
                    {
                        Rotate(ref _lo, ref _left, ref _right);
                        _right.Step = _lo.Step + (_hi.Step - _lo.Step) / _phi;
                    }
                    else // move left
                    {
                        Rotate(ref _hi, ref _right, ref _left);
                        if (_hi.Step <= _minStepSize)
                            goto FINISHED;
                        _left.Step = _hi.Step - (_hi.Step - _lo.Step) / _phi;
                    }
                }

                // prepare to return _left
                if (_right.Loss.CompareTo(_left.Loss) == 1)
                    Swap(ref _left, ref _right);
            }

        FINISHED:
            if (_hi.Step < _minStepSize)
                _left.Step = _minStepSize;
            else if (_hi.Step == _minStepSize)
                Swap(ref _hi, ref _left);

            double bestStep = _left.Step;

            ch.Info("multiplier: {0}", bestStep);
            _historicStepSize = bestStep;
            tree.ScaleOutputsBy(bestStep);
        }

        ScoreTracker IFastTrainingScoresUpdate.GetUpdatedTrainingScores()
        {
            ScoreTracker result = _left.Scores;
            _left.Scores = null; //We need to set it to null so that next call to AdjustTreeOutputs will not destroy returned object
            return result;
        }
    }
}
