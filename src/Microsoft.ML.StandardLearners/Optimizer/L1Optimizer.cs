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
    /// Orthant-Wise Limited-memory Quasi-Newton algorithm
    /// for optimization of smooth convex objectives plus L1-regularization
    /// If you use this code for published research, please cite
    ///   Galen Andrew and Jianfeng Gao, "Scalable Training of L1-Regularized Log-Linear Models",	ICML 2007
    /// </summary>
    public sealed class L1Optimizer : Optimizer
    {
        // Biases do not contribute to the L1 norm and are assumed to be at the beginning of the weights.
        private readonly int _biasCount;
        private readonly Float _l1weight;

        /// <summary>
        /// Create an L1Optimizer with the supplied value of M and termination criterion
        /// </summary>
        /// <param name="env">The environment</param>
        /// <param name="biasCount">Number of biases</param>
        /// <param name="l1weight">Weight of L1 regularizer</param>
        /// <param name="m">The number of previous iterations to store</param>
        /// <param name="keepDense">Whether the optimizer will keep its internal state dense</param>
        /// <param name="term">Termination criterion</param>
        /// <param name="enforceNonNegativity">The flag enforcing the non-negativity constraint</param>
        public L1Optimizer(IHostEnvironment env, int biasCount, Float l1weight, int m = 20, bool keepDense = false,
            ITerminationCriterion term = null, bool enforceNonNegativity = false)
            : base(env, m, keepDense, term, enforceNonNegativity)
        {
            Env.Check(biasCount >= 0);
            Env.Check(l1weight >= 0);
            _biasCount = biasCount;
            _l1weight = l1weight;
        }

        internal override OptimizerState MakeState(IChannel ch, IProgressChannelProvider progress, DifferentiableFunction function, ref VBuffer<Float> initial)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(progress);

            if (EnforceNonNegativity)
            {
                VBufferUtils.Apply(ref initial, delegate(int ind, ref Float initialVal)
                {
                    if (initialVal < 0.0 && ind >= _biasCount)
                        initialVal = 0;
                });
            }

            if (_l1weight > 0 && _biasCount < initial.Length)
                return new L1OptimizerState(ch, progress, function, ref initial, M, TotalMemoryLimit, _biasCount, _l1weight, KeepDense, EnforceNonNegativity);
            return new FunctionOptimizerState(ch, progress, function, ref initial, M, TotalMemoryLimit, KeepDense, EnforceNonNegativity);
        }

        /// <summary>
        /// Contains information about the state of the optimizer
        /// </summary>
        public sealed class L1OptimizerState : OptimizerState
        {
            private const Float Gamma = (Float)1e-4;
            private const int MaxLineSearch = 8;

            private readonly DifferentiableFunction _function;
            private readonly int _biasCount;
            private readonly Float _l1weight;

            internal L1OptimizerState(IChannel ch, IProgressChannelProvider progress, DifferentiableFunction function, ref VBuffer<Float> initial, int m, long totalMemLimit,
                int biasCount, Float l1Weight, bool keepDense, bool enforceNonNegativity)
                : base(ch, progress, ref initial, m, totalMemLimit, keepDense, enforceNonNegativity)
            {
                Contracts.AssertValue(ch);
                ch.Assert(0 <= biasCount && biasCount < initial.Length);
                ch.Assert(l1Weight > 0);

                _biasCount = biasCount;
                _l1weight = l1Weight;
                _function = function;

                Init();
            }

            public override DifferentiableFunction Function
            {
                get { return EvalCore; }
            }

            /// <summary>
            /// This is the original differentiable function with the injected L1 term.
            /// </summary>
            private Float EvalCore(ref VBuffer<Float> input, ref VBuffer<Float> gradient, IProgressChannelProvider progress)
            {
                // REVIEW: Leverage Vector methods that use SSE.
                Float res = 0;

                if (!EnforceNonNegativity)
                {
                    if (_biasCount > 0)
                        VBufferUtils.ForEachDefined(ref input,
                            (ind, value) => { if (ind >= _biasCount) res += Math.Abs(value); });
                    else
                        VBufferUtils.ForEachDefined(ref input, (ind, value) => res += Math.Abs(value));
                }
                else
                {
                    if (_biasCount > 0)
                        VBufferUtils.ForEachDefined(ref input,
                            (ind, value) => { if (ind >= _biasCount) res += value; });
                    else
                        VBufferUtils.ForEachDefined(ref input, (ind, value) => res += value);
                }
                res = _l1weight * res + _function(ref input, ref gradient, progress);
                return res;
            }

            public override Float Eval(ref VBuffer<Float> input, ref VBuffer<Float> gradient)
            {
                return EvalCore(ref input, ref gradient, ProgressProvider);
            }

            private void MakeSteepestDescDir()
            {
                if (!EnforceNonNegativity)
                {
                    VBufferUtils.ApplyInto(ref _x, ref _grad, ref _steepestDescDir,
                        (ind, xVal, gradVal) =>
                        {
                            if (ind < _biasCount)
                                return -gradVal;
                            if (xVal < 0)
                                return -gradVal + _l1weight;
                            if (xVal > 0)
                                return -gradVal - _l1weight;
                            if (gradVal < -_l1weight)
                                return -gradVal - _l1weight;
                            if (gradVal > _l1weight)
                                return -gradVal + _l1weight;
                            return 0;
                        });
                }
                else
                {
                    VBufferUtils.ApplyInto(ref _x, ref _grad, ref _steepestDescDir,
                        (ind, xVal, gradVal) =>
                        {
                            if (ind < _biasCount)
                                return -gradVal;
                            if (xVal > 0)
                                return -gradVal - _l1weight;
                            return -Math.Min(gradVal + _l1weight, 0);
                        });
                }

                _steepestDescDir.CopyTo(ref _dir);
            }

            private void GetNextPoint(Float alpha)
            {
                VectorUtils.AddMultInto(ref _x, alpha, ref _dir, ref _newX);

                if (!EnforceNonNegativity)
                {
                    VBufferUtils.ApplyWith(ref _x, ref _newX,
                        delegate(int ind, Float xVal, ref Float newXval)
                        {
                            if (xVal*newXval < 0.0 && ind >= _biasCount)
                                newXval = 0;
                        });
                }
                else
                {
                    VBufferUtils.Apply(ref _newX, delegate(int ind, ref Float newXval)
                    {
                        if (newXval < 0.0 && ind >= _biasCount)
                            newXval = 0;
                    });
                }
            }

            internal override void UpdateDir()
            {
                MakeSteepestDescDir();
                MapDirByInverseHessian();
                FixDirZeros();
            }

            /// <summary>
            /// Backtracking line search with Armijo-like condition, from Andrew &amp; Gao
            /// </summary>
            internal override bool LineSearch(IChannel ch, bool force)
            {
                Float dirDeriv = -VectorUtils.DotProduct(ref _dir, ref _steepestDescDir);

                if (dirDeriv == 0)
                    throw ch.Process(new PrematureConvergenceException(this, "Directional derivative is zero. You may be sitting on the optimum."));

                // if a non-descent direction is chosen, the line search will break anyway, so throw here
                // The most likely reason for this is a bug in your function's gradient computation
                // It may also indicate that your function is not convex.
                ch.Check(dirDeriv < 0, "L-BFGS chose a non-descent direction.");

                Float alpha = (Iter == 1 ? (1 / VectorUtils.Norm(_dir)) : 1);
                GetNextPoint(alpha);
                Float unnormCos = VectorUtils.DotProduct(ref _steepestDescDir, ref _newX) - VectorUtils.DotProduct(ref _steepestDescDir, ref _x);
                if (unnormCos < 0)
                {
                    VBufferUtils.ApplyWith(ref _steepestDescDir, ref _dir,
                        (int ind, Float sdVal, ref Float dirVal) =>
                        {
                            if (sdVal * dirVal < 0 && ind >= _biasCount)
                                dirVal = 0;
                        });

                    GetNextPoint(alpha);
                    unnormCos = VectorUtils.DotProduct(ref _steepestDescDir, ref _newX) - VectorUtils.DotProduct(ref _steepestDescDir, ref _x);
                }

                int i = 0;
                while (true)
                {
                    Value = Eval(ref _newX, ref _newGrad);
                    GradientCalculations++;

                    if (Value <= LastValue - Gamma * unnormCos)
                        return true;

                    ++i;
                    if (!force && i == MaxLineSearch)
                    {
                        return false;
                    }

                    alpha *= (Float)0.25;
                    GetNextPoint(alpha);
                    unnormCos = VectorUtils.DotProduct(ref _steepestDescDir, ref _newX) - VectorUtils.DotProduct(ref _steepestDescDir, ref _x);
                }
            }
        }
    }
}
