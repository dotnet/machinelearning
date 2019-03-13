// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Numeric
{
    /// <summary>
    /// An object which is used to decide whether to stop optimization.
    /// </summary>
    internal interface ITerminationCriterion
    {
        /// <summary>
        /// Name appropriate for display to the user.
        /// </summary>
        string FriendlyName { get; }

        /// <summary>
        /// Determines whether to stop optimization
        /// </summary>
        /// <param name="state">the state of the optimizer</param>
        /// <param name="message">a message to be printed (or null for no message)</param>
        /// <returns>true iff criterion is met, i.e. optimization should halt</returns>
        bool Terminate(Optimizer.OptimizerState state, out string message);

        /// <summary>
        /// Prepares the ITerminationCriterion for a new round of optimization
        /// </summary>
        void Reset();
    }

    /// <summary>
    /// A wrapper for a termination criterion that checks the gradient at a specified interval
    /// </summary>
    internal sealed class GradientCheckingMonitor : ITerminationCriterion
    {
        private const string _checkingMessage = "  Checking gradient...";
        private readonly ITerminationCriterion _termCrit;
        private readonly int _gradCheckInterval;
        // Reusable vectors utilized by the gradient tester.
        private VBuffer<float> _newGrad;
        private VBuffer<float> _newX;

        /// <summary>
        /// Initializes a new instance of the <see cref="GradientCheckingMonitor"/> class.
        /// </summary>
        /// <param name="termCrit">The termination criterion</param>
        /// <param name="gradientCheckInterval">The gradient check interval.</param>
        public GradientCheckingMonitor(ITerminationCriterion termCrit, int gradientCheckInterval)
        {
            Contracts.CheckParam(gradientCheckInterval > 0, nameof(gradientCheckInterval),
                "gradientCheckInterval must be positive.");

            _termCrit = termCrit;
            _gradCheckInterval = gradientCheckInterval;
        }

        public string FriendlyName { get { return "Gradient Checking Monitor wrapping " + _termCrit.FriendlyName; } }

        /// <summary>
        /// Determines whether to stop optimization
        /// </summary>
        /// <param name="state">the state of the optimizer</param>
        /// <param name="message">a message to be printed (or null for no message)</param>
        /// <returns>
        /// true iff criterion is met, i.e. optimization should halt
        /// </returns>
        public bool Terminate(Optimizer.OptimizerState state, out string message)
        {
            bool terminate = _termCrit.Terminate(state, out message);

            if (terminate || state.Iter % _gradCheckInterval == 1)
                message += string.Format("  GradCheck: {0,0:0.0000e0}", Check(state));

            return terminate;
        }

        private float Check(Optimizer.OptimizerState state)
        {
            Console.Error.Write(_checkingMessage);
            Console.Error.Flush();
            VBuffer<float> x = state.X;
            var lastDir = state.LastDir;
            float checkResult = GradientTester.Test(state.Function, in x, ref lastDir, true, ref _newGrad, ref _newX);
            for (int i = 0; i < _checkingMessage.Length; i++)
                Console.Error.Write('\b');
            return checkResult;
        }

        /// <summary>
        /// Prepares the ITerminationCriterion for a new round of optimization
        /// </summary>
        public void Reset()
        {
            _termCrit.Reset();
        }
    }

    /// <summary>
    /// An abstract partial implementation of ITerminationCriterion for those which do not require resetting
    /// </summary>
    internal abstract class StaticTerminationCriterion : ITerminationCriterion
    {
        public abstract string FriendlyName { get; }

        /// <summary>
        /// Determines whether to stop optimization
        /// </summary>
        /// <param name="state">the state of the optimizer</param>
        /// <param name="message">a message to be printed (or null for no message)</param>
        /// <returns>
        /// true iff criterion is met, i.e. optimization should halt
        /// </returns>
        public abstract bool Terminate(Optimizer.OptimizerState state, out string message);

        /// <summary>
        /// Prepares the ITerminationCriterion for a new round of optimization
        /// </summary>
        public void Reset() { }
    }

    /// <summary>
    /// Terminates when the geometrically-weighted average improvement falls below the tolerance
    /// </summary>
    internal sealed class MeanImprovementCriterion : ITerminationCriterion
    {
        private readonly float _tol;
        private readonly float _lambda;
        private readonly int _maxIterations;
        private float _unnormMeanImprovement;

        /// <summary>
        /// Initializes a new instance of the <see cref="MeanImprovementCriterion"/> class.
        /// </summary>
        /// <param name="tol">The tolerance parameter</param>
        /// <param name="lambda">The geometric weighting factor. Higher means more heavily weighted toward older values.</param>
        /// <param name="maxIterations">Maximum amount of iteration</param>
        public MeanImprovementCriterion(float tol = (float)1e-4, float lambda = (float)0.5, int maxIterations = int.MaxValue)
        {
            _tol = tol;
            _lambda = lambda;
            _maxIterations = maxIterations;
        }

        /// <summary>
        /// When criterion drops below this value, optimization is terminated
        /// </summary>
        public float Tolerance
        {
            get { return _tol; }
        }

        public string FriendlyName { get { return "Mean Improvement"; } }

        /// <summary>
        /// Determines whether to stop optimization
        /// </summary>
        /// <param name="state">the state of the optimizer</param>
        /// <param name="message">a message to be printed (or null for no message)</param>
        /// <returns>
        /// true iff criterion is met, i.e. optimization should halt
        /// </returns>
        public bool Terminate(Optimizer.OptimizerState state, out string message)
        {
            _unnormMeanImprovement = (state.LastValue - state.Value) + _lambda * _unnormMeanImprovement;

            float crit = _unnormMeanImprovement * (1 - _lambda) / (1 - MathUtils.Pow(_lambda, state.Iter));
            message = string.Format("{0:0.000e0}", crit);
            return (crit < _tol || state.Iter >= _maxIterations);
        }

        /// <summary>
        /// Prepares the ITerminationCriterion for a new round of optimization
        /// </summary>
        public void Reset()
        {
            _unnormMeanImprovement = 0;
        }
    }

    /// <summary>
    /// Stops optimization when the average objective improvement over the last
    /// n iterations, normalized by the function value, is small enough.
    /// </summary>
    /// <remarks>
    /// Inappropriate for functions whose optimal value is non-positive, because of normalization
    /// </remarks>
    internal sealed class MeanRelativeImprovementCriterion : ITerminationCriterion
    {
        private readonly int _n;
        private readonly float _tol;
        private readonly int _maxIterations;
        private Queue<float> _pastValues;

        /// <summary>
        /// When criterion drops below this value, optimization is terminated
        /// </summary>
        public float Tolerance
        {
            get { return _tol; }
        }

        /// <summary>
        /// Number of previous iterations to store
        /// </summary>
        public int Iters
        {
            get { return _n; }
        }

        /// <summary>
        /// Create a MeanRelativeImprovementCriterion
        /// </summary>
        /// <param name="tol">tolerance level</param>
        /// <param name="n">number of past iterations to average over</param>
        /// <param name="maxIterations">Maximum amount of iteration</param>
        public MeanRelativeImprovementCriterion(float tol = (float)1e-4, int n = 5, int maxIterations = int.MaxValue)
        {
            _tol = tol;
            _n = n;
            _maxIterations = maxIterations;
            _pastValues = new Queue<float>(n);
        }

        public string FriendlyName { get { return ToString(); } }

        /// <summary>
        /// Returns true if the average objective improvement over the last
        /// n iterations, normalized by the function value, is less than the tolerance
        /// </summary>
        /// <param name="state">current state of the optimizer</param>
        /// <param name="message">the current value of the criterion</param>
        /// <returns>true if criterion is less than tolerance</returns>
        public bool Terminate(Optimizer.OptimizerState state, out string message)
        {
            float value = state.Value;

            if (_pastValues.Count < _n)
            {
                _pastValues.Enqueue(value);
                message = "...";
                return false;
            }

            float avgImprovement = (_pastValues.Dequeue() - value) / _n;
            _pastValues.Enqueue(value);
            float val = avgImprovement / Math.Abs(value);
            message = string.Format("{0,0:0.0000e0}", val);
            return (val < _tol || state.Iter >= _maxIterations);
        }

        /// <summary>
        /// String summary of criterion
        /// </summary>
        /// <returns>summary of criterion</returns>
        public override string ToString()
        {
            return string.Format("Mean rel impr over {0} iter'ns < tol: {1,0:0.000e0}", _n, _tol);
        }

        /// <summary>
        /// Prepares the ITerminationCriterion for a new round of optimization
        /// </summary>
        public void Reset()
        {
            _pastValues.Clear();
        }
    }

    /// <summary>
    /// Uses the gradient to determine an upper bound on (relative) distance from the optimum.
    /// </summary>
    /// <remarks>
    /// Works if the objective uses L2 prior (or in general if the hessian H is such
    /// that H > (1 / sigmaSq) * I at all points)
    /// Inappropriate for functions whose optimal value is non-positive, because of normalization
    /// </remarks>
    internal sealed class UpperBoundOnDistanceWithL2 : StaticTerminationCriterion
    {
        private readonly float _sigmaSq;
        private readonly float _tol;
        private float _bestBoundOnMin;

        /// <summary>
        /// When criterion drops below this value, optimization is terminated
        /// </summary>
        public float Tolerance
        {
            get { return _tol; }
        }

        /// <summary>
        /// Create termination criterion with supplied value of sigmaSq and tolerance
        /// </summary>
        /// <param name="sigmaSq">value of sigmaSq in L2 regularizer</param>
        /// <param name="tol">tolerance level</param>
        public UpperBoundOnDistanceWithL2(float sigmaSq = 1, float tol = (float)1e-2)
        {
            _sigmaSq = sigmaSq;
            _tol = tol;

            // REVIEW: Why shouldn't this be "Reset"?
            _bestBoundOnMin = float.NegativeInfinity;
        }

        public override string FriendlyName { get { return ToString(); } }

        /// <summary>
        /// Returns true if the proved bound on the distance from the optimum,
        /// normalized by the function value, is less than the tolerance
        /// </summary>
        /// <param name="state">current state of the optimizer</param>
        /// <param name="message">value of criterion</param>
        /// <returns>true if criterion is less than tolerance</returns>
        public override bool Terminate(Optimizer.OptimizerState state, out string message)
        {
            var gradient = state.Grad;
            float gradientNormSquared = VectorUtils.NormSquared(gradient);
            float value = state.Value;
            float newBoundOnMin = value - (float)0.5 * _sigmaSq * gradientNormSquared;
            _bestBoundOnMin = Math.Max(_bestBoundOnMin, newBoundOnMin);
            float val = (value - _bestBoundOnMin) / Math.Abs(value);
            message = string.Format("{0,0:0.0000e0}", val);
            return (val < _tol);
        }

        /// <summary>
        /// String summary of criterion
        /// </summary>
        /// <returns>summary of criterion</returns>
        public override string ToString()
        {
            return string.Format("UB rel dist from opt, σ² = {0,0:0.00e0}, tol = {1,0:0.00e0}", _sigmaSq, _tol);
        }
    }

    /// <summary>
    /// Criterion based on the norm of the gradient being small enough
    /// </summary>
    /// <remarks>
    /// Inappropriate for functions whose optimal value is non-positive, because of normalization
    /// </remarks>
    internal sealed class RelativeNormGradient : StaticTerminationCriterion
    {
        private readonly float _tol;

        /// <summary>
        /// When criterion drops below this value, optimization is terminated
        /// </summary>
        public float Tolerance
        {
            get { return _tol; }
        }

        /// <summary>
        /// Create a RelativeNormGradient with the supplied tolerance
        /// </summary>
        /// <param name="tol">tolerance level</param>
        public RelativeNormGradient(float tol = (float)1e-4)
        {
            _tol = tol;
        }

        public override string FriendlyName { get { return ToString(); } }

        /// <summary>
        /// Returns true if the norm of the gradient, divided by the value, is less than the tolerance.
        /// </summary>
        /// <param name="state">current state of the optimzer</param>
        /// <param name="message">the current value of the criterion</param>
        /// <returns>true iff criterion is less than the tolerance</returns>
        public override bool Terminate(Optimizer.OptimizerState state, out string message)
        {
            var grad = state.Grad;
            float norm = VectorUtils.Norm(grad);
            float val = norm / Math.Abs(state.Value);
            message = string.Format("{0,0:0.0000e0}", val);
            return val < _tol;
        }

        /// <summary>
        /// String summary of criterion
        /// </summary>
        /// <returns>summary of criterion</returns>
        public override string ToString()
        {
            return string.Format("Norm of grad / value < {0,0:0.00e0}", _tol);
        }
    }
}
