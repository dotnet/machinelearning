// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Numeric
{
    /// <summary>
    /// Delegate for functions that determine whether to terminate search. Called after each update.
    /// </summary>
    /// <param name="x">Current iterate</param>
    /// <returns>True if search should terminate</returns>
    public delegate bool DTerminate(ref VBuffer<Float> x);

    /// <summary>
    /// Stochastic gradient descent with variations (minibatch, momentum, averaging).
    /// </summary>
    public sealed class SgdOptimizer
    {
        private int _batchSize;

        /// <summary>
        /// Size of minibatches
        /// </summary>
        public int BatchSize {
            get { return _batchSize; }
            set {
                Contracts.Check(value > 0);
                _batchSize = value;
            }
        }

        private Float _momentum;

        /// <summary>
        /// Momentum parameter
        /// </summary>
        public Float Momentum {
            get { return _momentum; }
            set {
                Contracts.Check(0 <= value && value < 1);
                _momentum = value;
            }
        }

        private Float _t0;

        /// <summary>
        /// Base of step size schedule s_t = 1 / (t0 + f(t))
        /// </summary>
        public Float T0 {
            get { return _t0; }
            set {
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
        public bool Averaging {
            get { return _averaging; }
            set { _averaging = value; }
        }

        private RateScheduleType _rateSchedule;

        /// <summary>
        /// Gets/Sets rate schedule type
        /// </summary>
        public RateScheduleType RateSchedule {
            get { return _rateSchedule; }
            set { _rateSchedule = value; }
        }

        private int _maxSteps;

        /// <summary>
        /// Gets/Sets maximum number of steps. Set to 0 for no max
        /// </summary>
        public int MaxSteps {
            get { return _maxSteps; }
            set {
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
        public SgdOptimizer(DTerminate terminate, RateScheduleType rateSchedule = RateScheduleType.Sqrt, bool averaging = false, Float t0 = 1, int batchSize = 1, Float momentum = 0, int maxSteps = 0)
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
        public delegate void DStochasticGradient(ref VBuffer<Float> x, ref VBuffer<Float> grad);

        /// <summary>
        /// Minimize the function represented by <paramref name="f"/>.
        /// </summary>
        /// <param name="f">Stochastic gradients of function to minimize</param>
        /// <param name="initial">Initial point</param>
        /// <param name="result">Approximate minimum of <paramref name="f"/></param>
        public void Minimize(DStochasticGradient f, ref VBuffer<Float> initial, ref VBuffer<Float> result)
        {
            Contracts.Check(FloatUtils.IsFinite(initial.Values, initial.Count), "The initial vector contains NaNs or infinite values.");
            int dim = initial.Length;

            VBuffer<Float> grad = VBufferUtils.CreateEmpty<Float>(dim);
            VBuffer<Float> step = VBufferUtils.CreateEmpty<Float>(dim);
            VBuffer<Float> x = default(VBuffer<Float>);
            initial.CopyTo(ref x);
            VBuffer<Float> prev = default(VBuffer<Float>);
            VBuffer<Float> avg = VBufferUtils.CreateEmpty<Float>(dim);

            for (int n = 0; _maxSteps == 0 || n < _maxSteps; ++n)
            {
                if (_momentum == 0)
                    step = new VBuffer<Float>(step.Length, 0, step.Values, step.Indices);
                else
                    VectorUtils.ScaleBy(ref step, _momentum);

                Float stepSize;
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

                Float scale = (1 - _momentum) / _batchSize;
                for (int i = 0; i < _batchSize; ++i)
                {
                    f(ref x, ref grad);
                    VectorUtils.AddMult(in grad, scale, ref step);
                }

                if (_averaging)
                {
                    Utils.Swap(ref avg, ref prev);
                    VectorUtils.ScaleBy(prev, ref avg, (Float)n / (n + 1));
                    VectorUtils.AddMult(in step, -stepSize, ref x);
                    VectorUtils.AddMult(in x, (Float)1 / (n + 1), ref avg);

                    if ((n > 0 && TerminateTester.ShouldTerminate(ref avg, ref prev)) || _terminate(ref avg))
                    {
                        result = avg;
                        return;
                    }
                }
                else
                {
                    Utils.Swap(ref x, ref prev);
                    VectorUtils.AddMult(in step, -stepSize, ref prev, ref x);
                    if ((n > 0 && TerminateTester.ShouldTerminate(ref x, ref prev)) || _terminate(ref x))
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
    public class GDOptimizer
    {
        /// <summary>
        /// Line search to use.
        /// </summary>
        public IDiffLineSearch LineSearch { get; set; }

        private int _maxSteps;

        /// <summary>
        /// Gets/Sets maximum number of steps. Set to 0 for no max.
        /// </summary>
        public int MaxSteps {
            get { return _maxSteps; }
            set {
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
                    LineSearch = new CubicInterpLineSearch((Float)0.01);
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

            private VBuffer<Float> _point;
            private VBuffer<Float> _newPoint;
            private VBuffer<Float> _grad;
            private VBuffer<Float> _newGrad;
            private VBuffer<Float> _dir;

            public VBuffer<Float> NewPoint => _newPoint;

            private Float _value;
            private Float _newValue;

            public Float Value => _value;

            private DifferentiableFunction _func;

            public Float Deriv => VectorUtils.DotProduct(in _dir, in _grad);

            public LineFunc(DifferentiableFunction function, ref VBuffer<Float> initial, bool useCG = false)
            {
                int dim = initial.Length;

                initial.CopyTo(ref _point);
                _func = function;
                // REVIEW: plumb the IProgressChannelProvider through.
                _value = _func(ref _point, ref _grad, null);
                VectorUtils.ScaleInto(in _grad, -1, ref _dir);

                _useCG = useCG;
            }

            public Float Eval(Float step, out Float deriv)
            {
                VectorUtils.AddMultInto(in _point, step, in _dir, ref _newPoint);
                _newValue = _func(ref _newPoint, ref _newGrad, null);
                deriv = VectorUtils.DotProduct(in _dir, in _newGrad);
                return _newValue;
            }

            public void ChangeDir()
            {
                if (_useCG)
                {
                    Float newByNew = VectorUtils.NormSquared(_newGrad);
                    Float newByOld = VectorUtils.DotProduct(in _newGrad, in _grad);
                    Float oldByOld = VectorUtils.NormSquared(_grad);
                    Float betaPR = (newByNew - newByOld) / oldByOld;
                    Float beta = Math.Max(0, betaPR);
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
        public void Minimize(DifferentiableFunction function, ref VBuffer<Float> initial, ref VBuffer<Float> result)
        {
            Contracts.Check(FloatUtils.IsFinite(initial.Values, initial.Count), "The initial vector contains NaNs or infinite values.");
            LineFunc lineFunc = new LineFunc(function, ref initial, UseCG);
            VBuffer<Float> prev = default(VBuffer<Float>);
            initial.CopyTo(ref prev);

            for (int n = 0; _maxSteps == 0 || n < _maxSteps; ++n)
            {
                Float step = LineSearch.Minimize(lineFunc.Eval, lineFunc.Value, lineFunc.Deriv);
                var newPoint = lineFunc.NewPoint;
                bool terminateNow = n > 0 && TerminateTester.ShouldTerminate(ref newPoint, ref prev);
                if (terminateNow || Terminate(ref newPoint))
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
        internal static bool ShouldTerminate(ref VBuffer<Float> x, ref VBuffer<Float> xprev)
        {
            Contracts.Assert(x.Length == xprev.Length, "Vectors must have the same dimensionality.");
            Contracts.Assert(FloatUtils.IsFinite(xprev.Values, xprev.Count));

            if (!FloatUtils.IsFinite(x.Values, x.Count))
                return true;

            if (x.IsDense && xprev.IsDense)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    if (x.Values[i] != xprev.Values[i])
                        return false;
                }
            }
            else if (xprev.IsDense)
            {
                int j = 0;
                for (int ii = 0; ii < x.Count; ii++)
                {
                    int i = x.Indices[ii];
                    while (j < i)
                    {
                        if (xprev.Values[j++] != 0)
                            return false;
                    }
                    Contracts.Assert(i == j);
                    if (x.Values[ii] != xprev.Values[j++])
                        return false;
                }

                while (j < xprev.Length)
                {
                    if (xprev.Values[j++] != 0)
                        return false;
                }
            }
            else if (x.IsDense)
            {
                int i = 0;
                for (int jj = 0; jj < xprev.Count; jj++)
                {
                    int j = xprev.Indices[jj];
                    while (i < j)
                    {
                        if (x.Values[i++] != 0)
                            return false;
                    }
                    Contracts.Assert(j == i);
                    if (x.Values[i++] != xprev.Values[jj])
                        return false;
                }

                while (i < x.Length)
                {
                    if (x.Values[i++] != 0)
                        return false;
                }
            }
            else
            {
                // Both sparse.
                int ii = 0;
                int jj = 0;
                while (ii < x.Count && jj < xprev.Count)
                {
                    int i = x.Indices[ii];
                    int j = xprev.Indices[jj];
                    if (i == j)
                    {
                        if (x.Values[ii++] != xprev.Values[jj++])
                            return false;
                    }
                    else if (i < j)
                    {
                        if (x.Values[ii++] != 0)
                            return false;
                    }
                    else
                    {
                        if (xprev.Values[jj++] != 0)
                            return false;
                    }
                }

                while (ii < x.Count)
                {
                    if (x.Values[ii++] != 0)
                        return false;
                }

                while (jj < xprev.Count)
                {
                    if (xprev.Values[jj++] != 0)
                        return false;
                }
            }

            return true;
        }
    }
}
