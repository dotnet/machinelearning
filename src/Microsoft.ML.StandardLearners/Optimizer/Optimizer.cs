// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Numeric
{
    /// <summary>
    /// Limited-memory BFGS quasi-Newton optimization routine
    /// </summary>
    public class Optimizer
    {
        /// Based on Nocedal and Wright, "Numerical Optimization, Second Edition"

        protected readonly bool EnforceNonNegativity;
        private ITerminationCriterion _staticTerm;

        // Whether the optimizer state should keep its internal vectors dense or not.
        // Turning on dense internal vectors can relieve load on the garbage collector,
        // but can possibly lead to higher overall memory utilization.
        protected readonly bool KeepDense;

        /// <summary>
        /// The host environment to use for reporting progress and exceptions.
        /// </summary>
        protected readonly IHostEnvironment Env;

        /// <summary>
        /// Number of previous iterations to remember for estimate of Hessian.
        /// </summary>
        /// <remarks>
        /// Higher M means better approximation to Newton's method, but uses more memory,
        /// and requires more time to compute direction. The optimal setting of M is problem
        /// specific, depending on such factors as how expensive is function evaluation
        /// compared to choosing the direction, how easily approximable is the function's
        /// Hessian, etc.
        /// M = 15..20 is usually reasonable but if necessary even M=2 is better than
        /// gradient descent
        /// </remarks>
        public int M { get; }

        // REVIEW: The total memory limit appears to never be set to anything other than -1,
        // or exercised anywhere to actually constrain memory? Should it be, or should we remove it
        // and clean it up?
        /// <summary>
        /// Gets or sets a bound on the total number of bytes allowed.
        /// If the whole application is using more than this, no more vectors will be allocated.
        /// </summary>
        public long TotalMemoryLimit { get; }

        /// <summary>
        /// Create an optimizer with the supplied value of M and termination criterion
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="m">The number of previous iterations to store</param>
        /// <param name="keepDense">Whether the optimizer will keep its internal state dense</param>
        /// <param name="term">Termination criterion, defaults to MeanRelativeImprovement if null</param>
        /// <param name="enforceNonNegativity">The flag enforcing the non-negativity constraint</param>
        public Optimizer(IHostEnvironment env, int m = 20, bool keepDense = false, ITerminationCriterion term = null,
            bool enforceNonNegativity = false)
        {
            Contracts.CheckValue(env, nameof(env));
            Env = env;
            M = m;
            KeepDense = keepDense;
            _staticTerm = term ?? new MeanRelativeImprovementCriterion();
            TotalMemoryLimit = -1;
            EnforceNonNegativity = enforceNonNegativity;
        }

        /// <summary>
        /// A class for exceptions thrown by the optimizer.
        /// </summary>
        public abstract class OptimizerException : Exception
        {
            /// <summary>
            /// The state of the optimizer when premature convergence happened.
            /// </summary>
            public OptimizerState State { get; }

            internal OptimizerException(OptimizerState state, string message)
                : base(message)
            {
                State = state;
            }
        }

        internal virtual OptimizerState MakeState(IChannel ch, IProgressChannelProvider progress, DifferentiableFunction function, ref VBuffer<Float> initial)
        {
            return new FunctionOptimizerState(ch, progress, function, ref initial, M, TotalMemoryLimit, KeepDense, EnforceNonNegativity);
        }

        internal sealed class FunctionOptimizerState : OptimizerState
        {
            public override DifferentiableFunction Function { get; }

            internal FunctionOptimizerState(IChannel ch, IProgressChannelProvider progress, DifferentiableFunction function, ref VBuffer<Float> initial, int m,
                long totalMemLimit, bool keepDense, bool enforceNonNegativity)
                : base(ch, progress, ref initial, m, totalMemLimit, keepDense, enforceNonNegativity)
            {
                Function = function;
                Init();
            }

            public override Float Eval(ref VBuffer<Float> input, ref VBuffer<Float> gradient)
            {
                return Function(ref input, ref gradient, ProgressProvider);
            }
        }

        /// <summary>
        /// Contains information about the state of the optimizer
        /// </summary>
        public abstract class OptimizerState
        {
#pragma warning disable MSML_GeneralName // Too annoying in this case. Consider fixing later.
            protected internal VBuffer<Float> _x;
            protected internal VBuffer<Float> _grad;
            protected internal VBuffer<Float> _newX;
            protected internal VBuffer<Float> _newGrad;
            protected internal VBuffer<Float> _dir;
            protected internal VBuffer<Float> _steepestDescDir;
#pragma warning restore MSML_GeneralName

            /// <summary>
            /// The dimensionality of the function
            /// </summary>
            public readonly int Dim;

            protected readonly IChannel Ch;
            protected readonly IProgressChannelProvider ProgressProvider;
            protected readonly bool EnforceNonNegativity;

            /// <summary>
            /// The function being optimized
            /// </summary>
            public abstract DifferentiableFunction Function { get; }
            public abstract Float Eval(ref VBuffer<Float> input, ref VBuffer<Float> gradient);

            /// <summary>
            /// The current point being explored
            /// </summary>
            public VBuffer<Float> X { get { return _newX; } }

            /// <summary>
            /// The gradient at the current point
            /// </summary>
            public VBuffer<Float> Grad { get { return _newGrad; } }

            /// <summary>
            /// The direction of search that led to the current point
            /// </summary>
            public VBuffer<Float> LastDir { get { return _dir; } }

            /// <summary>
            /// The current function value
            /// </summary>
            public Float Value { get; protected internal set; }

            /// <summary>
            /// The function value at the last point
            /// </summary>
            public Float LastValue { get; protected internal set; }

            /// <summary>
            /// The number of iterations so far
            /// </summary>
            public int Iter { get; protected internal set; }

            /// <summary>
            /// The number of completed gradient calculations in the current iteration.
            /// </summary>
            /// <remarks>This is updated in derived classes, since they may call Eval at different times.</remarks>
            // REVIEW: instead, we could split into Eval and EvalCore and inject it there.
            public int GradientCalculations { get; protected internal set; }

            /// <summary>
            /// Whether the optimizer state will keep its internal vectors dense or not.
            /// This being true may lead to reduced load on the garbage collector, at the
            /// cost of possibly higher overall memory utilization.
            /// </summary>
            private readonly bool _keepDense;

            private readonly VBuffer<Float>[] _sList;
            private readonly VBuffer<Float>[] _yList;
            private readonly List<Float> _roList;

            private int _m;
            private readonly long _totalMemLimit;

            protected internal OptimizerState(IChannel ch, IProgressChannelProvider progress, ref VBuffer<Float> initial,
                int m, long totalMemLimit, bool keepDense, bool enforceNonNegativity)
            {
                Contracts.AssertValue(ch);
                Ch = ch;
                ch.AssertValueOrNull(progress);
                ProgressProvider = progress;
                Iter = 1;

                _keepDense = keepDense;
                Dim = initial.Length;

                _x = CreateWorkingVector();
                initial.CopyTo(ref _x);
                _m = m;
                _totalMemLimit = totalMemLimit;

                Dim = initial.Length;
                _grad = CreateWorkingVector();
                _dir = CreateWorkingVector();
                _newX = CreateWorkingVector();
                _newGrad = CreateWorkingVector();
                _steepestDescDir = CreateWorkingVector();

                _sList = new VBuffer<Float>[_m];
                _yList = new VBuffer<Float>[_m];
                _roList = new List<Float>();

                EnforceNonNegativity = enforceNonNegativity;
            }

            /// <summary>
            /// Convenience function to construct a working vector of length <c>Dim</c>.
            /// </summary>
            /// <returns></returns>
            protected VBuffer<Float> CreateWorkingVector()
            {
                // Owing to the way the operations are structured, if the "x", "newX", and "dir" vectors
                // start out (or somehow naturally become) dense, they will remain dense.
                return _keepDense ? VBufferUtils.CreateDense<Float>(Dim) : VBufferUtils.CreateEmpty<Float>(Dim);
            }

            // Leaf constructors must call this once they are fully initialized.
            protected virtual void Init()
            {
                Value = LastValue = Eval(ref _x, ref _grad);
                GradientCalculations++;
                if (!FloatUtils.IsFinite(LastValue))
                    throw Ch.Except("Optimizer unable to proceed with loss function yielding {0}", LastValue);
            }

            internal void MapDirByInverseHessian()
            {
                int count = _roList.Count;

                if (count != 0)
                {
                    Float[] alphas = new Float[count];

                    int lastGoodRo = -1;

                    for (int i = count - 1; i >= 0; i--)
                    {
                        if (_roList[i] > 0)
                        {
                            alphas[i] = -VectorUtils.DotProduct(ref _sList[i], ref _dir) / _roList[i];
                            VectorUtils.AddMult(ref _yList[i], alphas[i], ref _dir);
                            if (lastGoodRo == -1)
                                lastGoodRo = i;
                        }
                    }

                    // if we have no positive ros, dir doesn't change
                    if (lastGoodRo == -1)
                        return;

                    Float yDotY = VectorUtils.DotProduct(ref _yList[lastGoodRo], ref _yList[lastGoodRo]);
                    VectorUtils.ScaleBy(ref _dir, _roList[lastGoodRo] / yDotY);

                    for (int i = 0; i <= lastGoodRo; i++)
                    {
                        if (_roList[i] > 0)
                        {
                            Float beta = VectorUtils.DotProduct(ref _yList[i], ref _dir) / _roList[i];
                            VectorUtils.AddMult(ref _sList[i], -alphas[i] - beta, ref _dir);
                        }
                    }
                }
            }

            internal void DiscardOldVectors()
            {
                _roList.Clear();
                Array.Clear(_sList, 0, _sList.Length);
                Array.Clear(_yList, 0, _yList.Length);
            }

            protected void FixDirZeros()
            {
                VBufferUtils.ApplyWithEitherDefined(ref _steepestDescDir, ref _dir,
                    (int i, Float sdVal, ref Float dirVal) =>
                    {
                        if (sdVal == 0)
                            dirVal = 0;
                    });
            }

            internal virtual void UpdateDir()
            {
                if (EnforceNonNegativity)
                {
                    VBufferUtils.ApplyInto(ref _x, ref _grad, ref _steepestDescDir,
                        (ind, xVal, gradVal) =>
                        {
                            if (xVal > 0)
                                return -gradVal;
                            return -Math.Min(gradVal, 0);
                        });

                    _steepestDescDir.CopyTo(ref _dir);
                }
                else
                    VectorUtils.ScaleInto(ref _grad, -1, ref _dir);

                MapDirByInverseHessian();

                if (EnforceNonNegativity)
                    FixDirZeros();
            }

            internal void Shift()
            {
                if (_roList.Count < _m)
                {
                    if (_totalMemLimit > 0)
                    {
                        long totalMem = GC.GetTotalMemory(true);
                        if (totalMem > _totalMemLimit)
                            _m = _roList.Count;
                    }
                }

                VBuffer<Float> nextS;
                VBuffer<Float> nextY;

                if (_roList.Count == _m)
                {
                    // REVIEW: Goofy. Instead somehow consider the array
                    // "circular" in some sense.
                    nextS = _sList[0];
                    Array.Copy(_sList, 1, _sList, 0, _m - 1);
                    nextY = _yList[0];
                    Array.Copy(_yList, 1, _yList, 0, _m - 1);
                    _roList.RemoveAt(0);
                }
                else
                {
                    nextS = CreateWorkingVector();
                    nextY = CreateWorkingVector();
                }

                VectorUtils.AddMultInto(ref _newX, -1, ref _x, ref nextS);
                VectorUtils.AddMultInto(ref _newGrad, -1, ref _grad, ref nextY);
                Float ro = VectorUtils.DotProduct(ref nextS, ref nextY);
                if (ro == 0)
                    throw Ch.Process(new PrematureConvergenceException(this, "ro equals zero. Is your function linear?"));

                _sList[_roList.Count] = nextS;
                _yList[_roList.Count] = nextY;
                _roList.Add(ro);

                var temp = LastValue;
                LastValue = Value;
                Value = temp;
                Utils.Swap(ref _x, ref _newX);
                Utils.Swap(ref _grad, ref _newGrad);

                Iter++;
                GradientCalculations = 0;
            }

            /// <summary>
            /// An implementation of the line search for the Wolfe conditions, from Nocedal &amp; Wright
            /// </summary>
            internal virtual bool LineSearch(IChannel ch, bool force)
            {
                Contracts.AssertValue(ch);
                Float dirDeriv = VectorUtils.DotProduct(ref _dir, ref _grad);

                if (dirDeriv == 0)
                    throw ch.Process(new PrematureConvergenceException(this, "Directional derivative is zero. You may be sitting on the optimum."));

                // if a non-descent direction is chosen, the line search will break anyway, so throw here
                // The most likely reasons for this is a bug in your function's gradient computation,
                ch.Check(dirDeriv < 0, "L-BFGS chose a non-descent direction.");

                Float c1 = (Float)1e-4 * dirDeriv;
                Float c2 = (Float)0.9 * dirDeriv;

                Float alpha = (Iter == 1 ? (1 / VectorUtils.Norm(_dir)) : 1);

                PointValueDeriv last = new PointValueDeriv(0, LastValue, dirDeriv);
                PointValueDeriv aLo = new PointValueDeriv();
                PointValueDeriv aHi = new PointValueDeriv();

                // initial bracketing phase
                while (true)
                {
                    VectorUtils.AddMultInto(ref _x, alpha, ref _dir, ref _newX);
                    if (EnforceNonNegativity)
                    {
                        VBufferUtils.Apply(ref _newX, delegate(int ind, ref Float newXval)
                        {
                            if (newXval < 0.0)
                                newXval = 0;
                        });
                    }

                    Value = Eval(ref _newX, ref _newGrad);
                    GradientCalculations++;
                    if (Float.IsPositiveInfinity(Value))
                    {
                        alpha /= 2;
                        continue;
                    }

                    if (!FloatUtils.IsFinite(Value))
                        throw ch.Except("Optimizer unable to proceed with loss function yielding {0}", Value);

                    dirDeriv = VectorUtils.DotProduct(ref _dir, ref _newGrad);
                    PointValueDeriv curr = new PointValueDeriv(alpha, Value, dirDeriv);

                    if ((curr.V > LastValue + c1 * alpha) || (last.A > 0 && curr.V >= last.V))
                    {
                        aLo = last;
                        aHi = curr;
                        break;
                    }
                    else if (Math.Abs(curr.D) <= -c2)
                    {
                        return true;
                    }
                    else if (curr.D >= 0)
                    {
                        aLo = curr;
                        aHi = last;
                        break;
                    }

                    last = curr;
                    if (alpha == 0)
                        alpha = Float.Epsilon; // Robust to divisional underflow.
                    else
                        alpha *= 2;
                }

                Float minChange = (Float)0.01;
                int maxSteps = 10;

                // this loop is the "zoom" procedure described in Nocedal & Wright
                for (int step = 0; ; ++step)
                {
                    if (step == maxSteps && !force)
                        return false;

                    PointValueDeriv left = aLo.A < aHi.A ? aLo : aHi;
                    PointValueDeriv right = aLo.A < aHi.A ? aHi : aLo;
                    if (left.D > 0 && right.D < 0)
                    {
                        // interpolating cubic would have max in range, not min (can this happen?)
                        // set a to the one with smaller value
                        alpha = aLo.V < aHi.V ? aLo.A : aHi.A;
                    }
                    else
                    {
                        alpha = CubicInterp(aLo, aHi);
                        if (Float.IsNaN(alpha) || Float.IsInfinity(alpha))
                            alpha = (aLo.A + aHi.A) / 2;
                    }

                    // this is to ensure that the new point is within bounds
                    // and that the change is reasonably sized
                    Float ub = (minChange * left.A + (1 - minChange) * right.A);
                    if (alpha > ub)
                        alpha = ub;
                    Float lb = (minChange * right.A + (1 - minChange) * left.A);
                    if (alpha < lb)
                        alpha = lb;

                    VectorUtils.AddMultInto(ref _x, alpha, ref _dir, ref _newX);
                    if (EnforceNonNegativity)
                    {
                        VBufferUtils.Apply(ref _newX, delegate(int ind, ref Float newXval)
                        {
                            if (newXval < 0.0)
                                newXval = 0;
                        });
                    }

                    Value = Eval(ref _newX, ref _newGrad);
                    GradientCalculations++;
                    if (!FloatUtils.IsFinite(Value))
                        throw ch.Except("Optimizer unable to proceed with loss function yielding {0}", Value);
                    dirDeriv = VectorUtils.DotProduct(ref _dir, ref _newGrad);

                    PointValueDeriv curr = new PointValueDeriv(alpha, Value, dirDeriv);

                    if ((curr.V > LastValue + c1 * alpha) || (curr.V >= aLo.V))
                    {
                        if (aHi.A == curr.A)
                        {
                            if (force)
                                throw ch.Process(new PrematureConvergenceException(this, "Step size interval numerically zero."));
                            else
                                return false;
                        }
                        aHi = curr;
                    }
                    else if (Math.Abs(curr.D) <= -c2)
                    {
                        return true;
                    }
                    else
                    {
                        if (curr.D * (aHi.A - aLo.A) >= 0)
                            aHi = aLo;
                        if (aLo.A == curr.A)
                        {
                            if (force)
                                throw ch.Process(new PrematureConvergenceException(this, "Step size interval numerically zero."));
                            else
                                return false;
                        }
                        aLo = curr;
                    }
                }
            }

            /// <summary>
            /// Cubic interpolation routine from Nocedal and Wright
            /// </summary>
            /// <param name="p0">first point, with value and derivative</param>
            /// <param name="p1">second point, with value and derivative</param>
            /// <returns>local minimum of interpolating cubic polynomial</returns>
            private static Float CubicInterp(PointValueDeriv p0, PointValueDeriv p1)
            {
                double t1 = p0.D + p1.D - 3 * (p0.V - p1.V) / (p0.A - p1.A);
                double t2 = Math.Sign(p1.A - p0.A) * Math.Sqrt(t1 * t1 - p0.D * p1.D);
                double num = p1.D + t2 - t1;
                double denom = p1.D - p0.D + 2 * t2;
                return (Float)(p1.A - (p1.A - p0.A) * num / denom);
            }

            private struct PointValueDeriv
            {
                public readonly Float A;
                public readonly Float V;
                public readonly Float D;

                public PointValueDeriv(Float a, Float value, Float deriv)
                {
                    A = a;
                    V = value;
                    D = deriv;
                }
            }
        }

        /// <summary>
        /// Minimize a function using the MeanRelativeImprovement termination criterion with the supplied tolerance level
        /// </summary>
        /// <param name="function">The function to minimize</param>
        /// <param name="initial">The initial point</param>
        /// <param name="tolerance">Convergence tolerance (smaller means more iterations, closer to exact optimum)</param>
        /// <param name="result">The point at the optimum</param>
        /// <param name="optimum">The optimum function value</param>
        /// <exception cref="PrematureConvergenceException">Thrown if successive points are within numeric precision of each other, but termination condition is still unsatisfied.</exception>
        public void Minimize(DifferentiableFunction function, ref VBuffer<Float> initial, Float tolerance, ref VBuffer<Float> result, out Float optimum)
        {
            ITerminationCriterion term = new MeanRelativeImprovementCriterion(tolerance);
            Minimize(function, ref initial, term, ref result, out optimum);
        }

        /// <summary>
        /// Minimize a function.
        /// </summary>
        /// <param name="function">The function to minimize</param>
        /// <param name="initial">The initial point</param>
        /// <param name="result">The point at the optimum</param>
        /// <param name="optimum">The optimum function value</param>
        /// <exception cref="PrematureConvergenceException">Thrown if successive points are within numeric precision of each other, but termination condition is still unsatisfied.</exception>
        public void Minimize(DifferentiableFunction function, ref VBuffer<Float> initial, ref VBuffer<Float> result, out Float optimum)
        {
            Minimize(function, ref initial, _staticTerm, ref result, out optimum);
        }

        /// <summary>
        /// Minimize a function using the supplied termination criterion
        /// </summary>
        /// <param name="function">The function to minimize</param>
        /// <param name="initial">The initial point</param>
        /// <param name="term">termination criterion to use</param>
        /// <param name="result">The point at the optimum</param>
        /// <param name="optimum">The optimum function value</param>
        /// <exception cref="PrematureConvergenceException">Thrown if successive points are within numeric precision of each other, but termination condition is still unsatisfied.</exception>
        public void Minimize(DifferentiableFunction function, ref VBuffer<Float> initial, ITerminationCriterion term, ref VBuffer<Float> result, out Float optimum)
        {
            const string computationName = "LBFGS Optimizer";
            using (var pch = Env.StartProgressChannel(computationName))
            using (var ch = Env.Start(computationName))
            {
                ch.Info("Beginning optimization");
                ch.Info("num vars: {0}", initial.Length);
                ch.Info("improvement criterion: {0}", term.FriendlyName);

                OptimizerState state = MakeState(ch, pch, function, ref initial);
                term.Reset();

                var header = new ProgressHeader(new[] { "Loss", "Improvement" }, new[] { "iterations", "gradients" });
                pch.SetHeader(header,
(Action<IProgressEntry>)(                    e =>
                    {
                        e.SetProgress(0, (double)(state.Iter - 1));
                        e.SetProgress(1, state.GradientCalculations);
                    }));

                bool finished = false;
                pch.Checkpoint(state.Value, null, 0);
                state.UpdateDir();
                while (!finished)
                {
                    bool success = state.LineSearch(ch, false);
                    if (!success)
                    {
                        // problem may be numerical errors in previous gradients
                        // try to save state of optimization by discarding them
                        // and starting over with gradient descent.

                        state.DiscardOldVectors();

                        state.UpdateDir();

                        state.LineSearch(ch, true);
                    }

                    string message;
                    finished = term.Terminate(state, out message);

                    double? improvement = null;
                    double x;
                    int end;
                    if (message != null && DoubleParser.TryParse(out x, message.AsMemory().Span, out end))
                        improvement = x;

                    pch.Checkpoint(state.Value, improvement, state.Iter);

                    if (!finished)
                    {
                        state.Shift();
                        state.UpdateDir();
                    }
                }

                state.X.CopyTo(ref result);
                optimum = state.Value;
                ch.Done();
            }
        }

        /// <summary>
        /// This exception is thrown if successive differences between points
        /// reach the limits of numerical stability, but the termination condition
        /// still hasn't been satisfied
        /// </summary>
        public sealed class PrematureConvergenceException : OptimizerException
        {
            /// <summary>
            /// Makes a PrematureConvergenceException with the supplied message
            /// </summary>
            /// <param name="state">The OptimizerState when the exception was thrown</param>
            /// <param name="message">message for exception</param>
            public PrematureConvergenceException(OptimizerState state, string message) : base(state, message) { }
        }

        /// <summary>
        /// If true, suppresses all output.
        /// </summary>
        public bool Quiet { get; set; }
    }
}
