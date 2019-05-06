// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Numeric
{
    /// <summary>
    /// Delegate for functions that determine whether to terminate search. Called after each update.
    /// </summary>
    /// <param name="x">Current iterate</param>
    /// <returns>True if search should terminate</returns>
    internal delegate bool DTerminate(in VBuffer<float> x);

    /// <summary>
    /// Stochastic gradient descent with variations (minibatch, momentum, averaging).
    /// </summary>
    internal sealed class SgdOptimizer
    {
        private int _batchSize;

        /// <summary>
        /// Size of minibatches
        /// </summary>
        public int BatchSize
        {
            get { return _batchSize; }
            set
            {
                Contracts.Check(value > 0);
                _batchSize = value;
            }
        }

        private float _momentum;

        /// <summary>
        /// Momentum parameter
        /// </summary>
        public float Momentum
        {
            get { return _momentum; }
            set
            {
                Contracts.Check(0 <= value && value < 1);
                _momentum = value;
            }
        }

        private float _t0;

        /// <summary>
        /// Base of step size schedule s_t = 1 / (t0 + f(t))
        /// </summary>
        public float T0
        {
            get { return _t0; }
            set
            {
                Contracts.Check(value >= 0);
                _t0 = value;
            }
        }

        /// <summary>
        /// Termination criterion
        /// </summary>
        private readonly DTerminate _terminate;

        private bool _averaging;

        /// <summary>
        /// If true, iterates are averaged
        /// </summary>
        public bool Averaging
        {
            get { return _averaging; }
            set { _averaging = value; }
        }

        private RateScheduleType _rateSchedule;

        /// <summary>
        /// Gets/Sets rate schedule type
        /// </summary>
        public RateScheduleType RateSchedule
        {
            get { return _rateSchedule; }
            set { _rateSchedule = value; }
        }

        private int _maxSteps;

        /// <summary>
        /// Gets/Sets maximum number of steps. Set to 0 for no max
        /// </summary>
        public int MaxSteps
        {
            get { return _maxSteps; }
            set
            {
                Contracts.Check(value >= 0);
                _maxSteps = value;
            }
        }

        /// <summary>
        /// Annealing schedule for learning rate
        /// </summary>
        public enum RateScheduleType
        {
            /// <summary>
            /// r_t = 1 / t0
            /// </summary>
            Constant,

            /// <summary>
            /// r_t = 1 / (t0 + sqrt(t))
            /// </summary>
            Sqrt,

            /// <summary>
            /// r_t = 1 / (t0 + t)
            /// </summary>
            Linear
        }

        /// <summary>
        /// Creates SGDOptimizer and sets optimization parameters
        /// </summary>
        /// <param name="terminate">Termination criterion</param>
        /// <param name="rateSchedule">Annealing schedule type for learning rate</param>
        /// <param name="averaging">If true, all iterates are averaged</param>
        /// <param name="t0">Base for learning rate schedule</param>
        /// <param name="batchSize">Average this number of stochastic gradients for each update</param>
        /// <param name="momentum">Momentum parameter</param>
        /// <param name="maxSteps">Maximum number of updates (0 for no max)</param>
        public SgdOptimizer(DTerminate terminate, RateScheduleType rateSchedule = RateScheduleType.Sqrt, bool averaging = false, float t0 = 1, int batchSize = 1, float momentum = 0, int maxSteps = 0)
        {
            _terminate = terminate;
            _rateSchedule = rateSchedule;
            _averaging = averaging;
            _t0 = t0;
            _batchSize = batchSize;
            _momentum = momentum;
            _maxSteps = maxSteps;
        }

        /// <summary>
        /// Delegate for functions to query stochastic gradient at a point
        /// </summary>
        /// <param name="x">Point at which to evaluate</param>
        /// <param name="grad">Vector to be filled in with gradient</param>
        public delegate void DStochasticGradient(in VBuffer<float> x, ref VBuffer<float> grad);

        /// <summary>
        /// Minimize the function represented by <paramref name="f"/>.
        /// </summary>
        /// <param name="f">Stochastic gradients of function to minimize</param>
        /// <param name="initial">Initial point</param>
        /// <param name="result">Approximate minimum of <paramref name="f"/></param>
        public void Minimize(DStochasticGradient f, ref VBuffer<float> initial, ref VBuffer<float> result)
        {
            Contracts.Check(FloatUtils.IsFinite(initial.GetValues()), "The initial vector contains NaNs or infinite values.");
            int dim = initial.Length;

            VBuffer<float> grad = VBufferUtils.CreateEmpty<float>(dim);
            VBuffer<float> step = VBufferUtils.CreateEmpty<float>(dim);
            VBuffer<float> x = default(VBuffer<float>);
            initial.CopyTo(ref x);
            VBuffer<float> prev = default(VBuffer<float>);
            VBuffer<float> avg = VBufferUtils.CreateEmpty<float>(dim);

            for (int n = 0; _maxSteps == 0 || n < _maxSteps; ++n)
            {
                if (_momentum == 0)
                    VBufferUtils.Resize(ref step, step.Length, 0);
                else
                    VectorUtils.ScaleBy(ref step, _momentum);

                float stepSize;
                switch (_rateSchedule)
                {
                    case RateScheduleType.Constant:
                        stepSize = 1 / _t0;
                        break;
                    case RateScheduleType.Sqrt:
                        stepSize = 1 / (_t0 + MathUtils.Sqrt(n));
                        break;
                    case RateScheduleType.Linear:
                        stepSize = 1 / (_t0 + n);
                        break;
                    default:
                        throw Contracts.Except();
                }

                float scale = (1 - _momentum) / _batchSize;
                for (int i = 0; i < _batchSize; ++i)
                {
                    f(in x, ref grad);
                    VectorUtils.AddMult(in grad, scale, ref step);
                }

                if (_averaging)
                {
                    Utils.Swap(ref avg, ref prev);
                    VectorUtils.ScaleBy(prev, ref avg, (float)n / (n + 1));
                    VectorUtils.AddMult(in step, -stepSize, ref x);
                    VectorUtils.AddMult(in x, (float)1 / (n + 1), ref avg);

                    if ((n > 0 && TerminateTester.ShouldTerminate(in avg, in prev)) || _terminate(in avg))
                    {
                        result = avg;
                        return;
                    }
                }
                else
                {
                    Utils.Swap(ref x, ref prev);
                    VectorUtils.AddMult(in step, -stepSize, ref prev, ref x);
                    if ((n > 0 && TerminateTester.ShouldTerminate(in x, in prev)) || _terminate(in x))
                    {
                        result = x;
                        return;
                    }
                }
            }

            result = _averaging ? avg : x;
        }
    }

    /// <summary>
    /// Deterministic gradient descent with line search
    /// </summary>
    internal class GDOptimizer
    {
        /// <summary>
        /// Line search to use.
        /// </summary>
        public IDiffLineSearch LineSearch { get; set; }

        private int _maxSteps;

        /// <summary>
        /// Gets/Sets maximum number of steps. Set to 0 for no max.
        /// </summary>
        public int MaxSteps
        {
            get { return _maxSteps; }
            set
            {
                Contracts.Check(value >= 0);
                _maxSteps = value;
            }
        }

        /// <summary>
        /// Gets/sets termination criterion.
        /// </summary>
        public DTerminate Terminate { get; set; }

        /// <summary>
        /// Gets/sets whether to use nonlinear conjugate gradient.
        /// </summary>
        public bool UseCG { get; set; }

        /// <summary>
        /// Makes a new GDOptimizer with the given optimization parameters
        /// </summary>
        /// <param name="terminate">Termination criterion</param>
        /// <param name="lineSearch">Line search to use</param>
        /// <param name="maxSteps">Maximum number of updates</param>
        /// <param name="useCG">Use Cubic interpolation line search or Backtracking line search with Armijo condition</param>
        public GDOptimizer(DTerminate terminate, IDiffLineSearch lineSearch = null, bool useCG = false, int maxSteps = 0)
        {
            Terminate = terminate;
            if (LineSearch == null)
            {
                if (useCG)
                    LineSearch = new CubicInterpLineSearch((float)0.01);
                else
                    LineSearch = new BacktrackingLineSearch();
            }
            else
                LineSearch = lineSearch;
            _maxSteps = maxSteps;
            UseCG = useCG;
        }

        private class LineFunc
        {
            private bool _useCG;

            private VBuffer<float> _point;
            private VBuffer<float> _newPoint;
            private VBuffer<float> _grad;
            private VBuffer<float> _newGrad;
            private VBuffer<float> _dir;

            public VBuffer<float> NewPoint => _newPoint;

            private float _value;
            private float _newValue;

            public float Value => _value;

            private DifferentiableFunction _func;

            public float Deriv => VectorUtils.DotProduct(in _dir, in _grad);

            public LineFunc(DifferentiableFunction function, in VBuffer<float> initial, bool useCG = false)
            {
                int dim = initial.Length;

                initial.CopyTo(ref _point);
                _func = function;
                // REVIEW: plumb the IProgressChannelProvider through.
                _value = _func(in _point, ref _grad, null);
                VectorUtils.ScaleInto(in _grad, -1, ref _dir);

                _useCG = useCG;
            }

            public float Eval(float step, out float deriv)
            {
                VectorUtils.AddMultInto(in _point, step, in _dir, ref _newPoint);
                _newValue = _func(in _newPoint, ref _newGrad, null);
                deriv = VectorUtils.DotProduct(in _dir, in _newGrad);
                return _newValue;
            }

            public void ChangeDir()
            {
                if (_useCG)
                {
                    float newByNew = VectorUtils.NormSquared(_newGrad);
                    float newByOld = VectorUtils.DotProduct(in _newGrad, in _grad);
                    float oldByOld = VectorUtils.NormSquared(_grad);
                    float betaPR = (newByNew - newByOld) / oldByOld;
                    float beta = Math.Max(0, betaPR);
                    VectorUtils.ScaleBy(ref _dir, beta);
                    VectorUtils.AddMult(in _newGrad, -1, ref _dir);
                }
                else
                    VectorUtils.ScaleInto(in _newGrad, -1, ref _dir);
                _newPoint.CopyTo(ref _point);
                _newGrad.CopyTo(ref _grad);
                _value = _newValue;
            }
        }

        /// <summary>
        /// Finds approximate minimum of the function
        /// </summary>
        /// <param name="function">Function to minimize</param>
        /// <param name="initial">Initial point</param>
        /// <param name="result">Approximate minimum</param>
        public void Minimize(DifferentiableFunction function, in VBuffer<float> initial, ref VBuffer<float> result)
        {
            Contracts.Check(FloatUtils.IsFinite(initial.GetValues()), "The initial vector contains NaNs or infinite values.");
            LineFunc lineFunc = new LineFunc(function, in initial, UseCG);
            VBuffer<float> prev = default(VBuffer<float>);
            initial.CopyTo(ref prev);

            for (int n = 0; _maxSteps == 0 || n < _maxSteps; ++n)
            {
                float step = LineSearch.Minimize(lineFunc.Eval, lineFunc.Value, lineFunc.Deriv);
                var newPoint = lineFunc.NewPoint;
                bool terminateNow = n > 0 && TerminateTester.ShouldTerminate(in newPoint, in prev);
                if (terminateNow || Terminate(in newPoint))
                    break;
                newPoint.CopyTo(ref prev);
                lineFunc.ChangeDir();
            }

            lineFunc.NewPoint.CopyTo(ref result);
        }
    }

    /// <summary>
    /// Terminates the optimization if NA value appears in result or no progress is made.
    /// </summary>
    internal static class TerminateTester
    {
        /// <summary>
        /// Test whether the optimization should terminate. Returns true if x contains NA or +/-Inf or x equals xprev.
        /// </summary>
        /// <param name="x">The current value.</param>
        /// <param name="xprev">The value from the previous iteration.</param>
        /// <returns>True if the optimization routine should terminate at this iteration.</returns>
        internal static bool ShouldTerminate(in VBuffer<float> x, in VBuffer<float> xprev)
        {
            Contracts.Assert(x.Length == xprev.Length, "Vectors must have the same dimensionality.");
            Contracts.Assert(FloatUtils.IsFinite(xprev.GetValues()));

            var xValues = x.GetValues();
            if (!FloatUtils.IsFinite(xValues))
                return true;

            var xprevValues = xprev.GetValues();
            if (x.IsDense && xprev.IsDense)
            {
                for (int i = 0; i < xValues.Length; i++)
                {
                    if (xValues[i] != xprevValues[i])
                        return false;
                }
            }
            else if (xprev.IsDense)
            {
                var xIndices = x.GetIndices();
                int j = 0;
                for (int ii = 0; ii < xValues.Length; ii++)
                {
                    int i = xIndices[ii];
                    while (j < i)
                    {
                        if (xprevValues[j++] != 0)
                            return false;
                    }
                    Contracts.Assert(i == j);
                    if (xValues[ii] != xprevValues[j++])
                        return false;
                }

                while (j < xprevValues.Length)
                {
                    if (xprevValues[j++] != 0)
                        return false;
                }
            }
            else if (x.IsDense)
            {
                var xprevIndices = xprev.GetIndices();
                int i = 0;
                for (int jj = 0; jj < xprevValues.Length; jj++)
                {
                    int j = xprevIndices[jj];
                    while (i < j)
                    {
                        if (xValues[i++] != 0)
                            return false;
                    }
                    Contracts.Assert(j == i);
                    if (xValues[i++] != xprevValues[jj])
                        return false;
                }

                while (i < xValues.Length)
                {
                    if (xValues[i++] != 0)
                        return false;
                }
            }
            else
            {
                // Both sparse.
                var xIndices = x.GetIndices();
                var xprevIndices = xprev.GetIndices();
                int ii = 0;
                int jj = 0;
                while (ii < xValues.Length && jj < xprevValues.Length)
                {
                    int i = xIndices[ii];
                    int j = xprevIndices[jj];
                    if (i == j)
                    {
                        if (xValues[ii++] != xprevValues[jj++])
                            return false;
                    }
                    else if (i < j)
                    {
                        if (xValues[ii++] != 0)
                            return false;
                    }
                    else
                    {
                        if (xprevValues[jj++] != 0)
                            return false;
                    }
                }

                while (ii < xValues.Length)
                {
                    if (xValues[ii++] != 0)
                        return false;
                }

                while (jj < xprevValues.Length)
                {
                    if (xprevValues[jj++] != 0)
                        return false;
                }
            }

            return true;
        }
    }
}
