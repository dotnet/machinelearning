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
    /// Line search that does not use derivatives
    /// </summary>
    internal interface ILineSearch : IDiffLineSearch
    {
        /// <summary>
        /// Finds a local minimum of the function
        /// </summary>
        /// <param name="func">Function to minimize</param>
        /// <returns>Minimizing value</returns>
        float Minimize(Func<float, float> func);
    }

    /// <summary>
    /// Delegate for differentiable 1-D functions
    /// </summary>
    /// <param name="x">Point to evaluate</param>
    /// <param name="deriv">Derivative at that point</param>
    /// <returns></returns>
    internal delegate float DiffFunc1D(float x, out float deriv);

    /// <summary>
    /// Line search that uses derivatives
    /// </summary>
    internal interface IDiffLineSearch
    {
        /// <summary>
        /// Finds a local minimum of the function
        /// </summary>
        /// <param name="func">Function to minimize</param>
        /// <param name="initValue">Value of function at 0</param>
        /// <param name="initDeriv">Derivative of function at 0</param>
        /// <returns>Minimizing value</returns>
        float Minimize(DiffFunc1D func, float initValue, float initDeriv);
    }

    /// <summary>
    /// Cubic interpolation line search
    /// </summary>
    internal sealed class CubicInterpLineSearch : IDiffLineSearch
    {
        private float _step;
        private const float _minProgress = (float)0.01;

        /// <summary>
        /// Gets or sets maximum number of steps.
        /// </summary>
        public int MaxNumSteps { get; set; }

        /// <summary>
        /// Gets or sets the minimum relative size of bounds around solution.
        /// </summary>
        public float MinWindow { get; set; }

        /// <summary>
        /// Gets or sets maximum step size
        /// </summary>
        public float MaxStep { get; set; }

        /// <summary>
        /// Makes a CubicInterpLineSearch
        /// </summary>
        /// <param name="maxNumSteps">Maximum number of steps before terminating</param>
        public CubicInterpLineSearch(int maxNumSteps)
        {
            MaxStep = float.PositiveInfinity;
            MaxNumSteps = maxNumSteps;
            _step = 1;
        }

        /// <summary>
        /// Makes a CubicInterpLineSearch
        /// </summary>
        /// <param name="minWindow">Minimum relative size of bounds around solution</param>
        public CubicInterpLineSearch(float minWindow)
        {
            MaxStep = float.PositiveInfinity;
            MinWindow = minWindow;
            _step = 1;
        }

        /// <summary>
        /// Cubic interpolation routine from Nocedal and Wright
        /// </summary>
        /// <param name="a">first point, with value and derivative</param>
        /// <param name="b">second point, with value and derivative</param>
        /// <returns>local minimum of interpolating cubic polynomial</returns>
        private static float CubicInterp(StepValueDeriv a, StepValueDeriv b)
        {
            float t1 = a.Deriv + b.Deriv - 3 * (a.Value - b.Value) / (a.Step - b.Step);
            float t2 = Math.Sign(b.Step - a.Step) * MathUtils.Sqrt(t1 * t1 - a.Deriv * b.Deriv);
            float num = b.Deriv + t2 - t1;
            float denom = b.Deriv - a.Deriv + 2 * t2;
            return b.Step - (b.Step - a.Step) * num / denom;
        }

        private sealed class StepValueDeriv
        {
            private readonly DiffFunc1D _func;

            public StepValueDeriv(DiffFunc1D func)
            {
                _func = func;
            }

            public StepValueDeriv(DiffFunc1D func, float initStep)
            {
                _func = func;
                Step = initStep;
            }

            public StepValueDeriv(DiffFunc1D func, float initStep, float initVal, float initDeriv)
            {
                _func = func;
                _step = initStep;
                _value = initVal;
                _deriv = initDeriv;
            }

            private float _step;
            private float _value;
            private float _deriv;

            public float Step
            {
                get { return _step; }
                set { _step = value; _value = _func(value, out _deriv); }
            }

            public float Value => _value;

            public float Deriv => _deriv;
        }

        private static void Swap<T>(ref T a, ref T b)
        {
            T t = a;
            a = b;
            b = t;
        }

        /// <summary>
        /// Finds a local minimum of the function
        /// </summary>
        /// <param name="func">Function to minimize</param>
        /// <param name="initValue">Value of function at 0</param>
        /// <param name="initDeriv">Derivative of function at 0</param>
        /// <returns>Minimizing value</returns>
        public float Minimize(DiffFunc1D func, float initValue, float initDeriv)
        {
            _step = FindMinimum(func, initValue, initDeriv);
            return Math.Min(_step, MaxStep);
        }

        private float FindMinimum(DiffFunc1D func, float initValue, float initDeriv)
        {
            Contracts.CheckParam(initDeriv < 0, nameof(initDeriv), "Cannot search in direction of ascent!");

            StepValueDeriv lo = new StepValueDeriv(func, 0, initValue, initDeriv);
            StepValueDeriv hi = new StepValueDeriv(func, _step);

            // bracket minimum
            while (hi.Deriv < 0)
            {
                Swap(ref lo, ref hi);
                if (lo.Step >= MaxStep)
                    return MaxStep;
                hi.Step = lo.Step * 2;
            }

            float window = 1;

            StepValueDeriv mid = new StepValueDeriv(func);
            for (int numSteps = 1; ; ++numSteps)
            {
                float interp = CubicInterp(lo, hi);
                if (window <= MinWindow || numSteps == MaxNumSteps)
                    return interp;

                // insure minimal progress to narrow interval
                float minProgressStep = _minProgress * (hi.Step - lo.Step);
                float maxMid = hi.Step - minProgressStep;
                if (interp > maxMid)
                    interp = maxMid;
                float minMid = lo.Step + minProgressStep;
                if (interp < minMid)
                    interp = minMid;

                mid.Step = interp;

                if (mid.Deriv == 0 || mid.Step == lo.Step || mid.Step == hi.Step)
                    return mid.Step;

                if (mid.Deriv < 0)
                    Swap(ref lo, ref mid);
                else
                    Swap(ref hi, ref mid);

                if (lo.Step >= MaxStep)
                    return MaxStep;

                window = (hi.Step - lo.Step) / hi.Step;
            }
        }
    }

    /// <summary>
    /// Finds local minimum with golden section search.
    /// </summary>
    internal sealed class GoldenSectionSearch : ILineSearch
    {
        private float _step;
        private static readonly float _phi = (1 + MathUtils.Sqrt(5)) / 2;

        /// <summary>
        /// Gets or sets maximum number of steps before terminating.
        /// </summary>
        public int MaxNumSteps { get; set; }

        /// <summary>
        /// Gets or sets minimum relative size of bounds around solution.
        /// </summary>
        public float MinWindow { get; set; }

        /// <summary>
        /// Gets or sets maximum step size.
        /// </summary>
        public float MaxStep { get; set; }

        /// <summary>
        /// Makes a new GoldenSectionSearch
        /// </summary>
        /// <param name="maxNumSteps">Maximum number of steps before terminating (not including bracketing)</param>
        public GoldenSectionSearch(int maxNumSteps)
        {
            MaxStep = float.PositiveInfinity;
            MaxNumSteps = maxNumSteps;
            _step = 1;
        }

        /// <summary>
        /// Makes a new GoldenSectionSearch
        /// </summary>
        /// <param name="minWindow">Minimum relative size of bounds around solution</param>
        public GoldenSectionSearch(float minWindow)
        {
            MaxStep = float.PositiveInfinity;
            MaxNumSteps = int.MaxValue;
            MinWindow = minWindow;
            _step = 1;
        }

        private static void Rotate<T>(ref T a, ref T b, ref T c)
        {
            T t = a;
            a = b;
            b = c;
            c = t;
        }

        private sealed class StepAndValue
        {
            private readonly Func<float, float> _func;
            private float _step;

            public float Step
            {
                get { return _step; }
                set
                {
                    _step = value;
                    Value = _func(value);
                }
            }

            public float Value { get; private set; }

            public StepAndValue(Func<float, float> func)
            {
                _func = func;
                _step = Value = float.NaN;
            }

            public StepAndValue(Func<float, float> func, float initStep)
                : this(func)
            {
                Step = initStep;
            }
        }

        /// <summary>
        /// Finds a local minimum of the function
        /// </summary>
        /// <param name="f">Function to minimize</param>
        /// <param name="initVal">Value of function at 0</param>
        /// <param name="initDeriv">Derivative of function at 0</param>
        /// <returns>Minimizing value</returns>
        public float Minimize(DiffFunc1D f, float initVal, float initDeriv)
        {
            return Minimize(f);
        }

        /// <summary>
        /// Finds a local minimum of the function
        /// </summary>
        /// <param name="func">Function to minimize</param>
        /// <returns>Minimizing value</returns>
        public float Minimize(DiffFunc1D func)
        {
            float d;
            return Minimize(x => func(x, out d));
        }

        /// <summary>
        /// Finds a local minimum of the function
        /// </summary>
        /// <param name="func">Function to minimize</param>
        /// <returns>Minimizing value</returns>
        public float Minimize(Func<float, float> func)
        {
            _step = FindMinimum(func);
            return Math.Min(_step, MaxStep);
        }

        private float FindMinimum(Func<float, float> func)
        {
            StepAndValue lo = new StepAndValue(func, _step / _phi);
            StepAndValue left = new StepAndValue(func, _step);

            StepAndValue hi = new StepAndValue(func);

            // bracket minimum
            if (lo.Value < left.Value)
            {
                do
                {
                    Rotate(ref hi, ref left, ref lo);
                    lo.Step = left.Step / _phi;
                } while (lo.Value < left.Value);
            }
            else
            {
                hi.Step = _step * _phi;
                while (hi.Value < left.Value)
                {
                    Rotate(ref lo, ref left, ref hi);
                    if (lo.Step >= MaxStep)
                        return MaxStep;
                    hi.Step = left.Step * _phi;
                }
            }

            float window = 1 - 1 / (_phi * _phi);
            int numSteps = 0;

            if (window <= MinWindow || numSteps == MaxNumSteps)
                return left.Step;

            StepAndValue right = new StepAndValue(func, lo.Step + (hi.Step - lo.Step) / _phi);
            do
            {
                if (left.Value == right.Value)
                    return left.Step;
                else if (left.Value < right.Value)
                {
                    Rotate(ref hi, ref right, ref left);
                    left.Step = hi.Step - (hi.Step - lo.Step) / _phi;
                }
                else
                {
                    Rotate(ref lo, ref left, ref right);
                    if (lo.Step >= MaxStep)
                        return MaxStep;
                    right.Step = lo.Step + (hi.Step - lo.Step) / _phi;
                }

                ++numSteps;
                window = (hi.Step - lo.Step) / hi.Step;
            } while (window > MinWindow && numSteps < MaxNumSteps);

            if (left.Value < right.Value)
                return left.Step;
            else
                return right.Step;
        }
    }

    /// <summary>
    /// Backtracking line search with Armijo condition
    /// </summary>
    internal sealed class BacktrackingLineSearch : IDiffLineSearch
    {
        private float _step;
        private float _c1;

        /// <summary>
        /// Makes a backtracking line search
        /// </summary>
        /// <param name="c1">Parameter for Armijo condition</param>
        public BacktrackingLineSearch(float c1 = (float)1e-4)
        {
            _step = 1;
            _c1 = c1;
        }

        /// <summary>
        /// Finds a local minimum of the function
        /// </summary>
        /// <param name="f">Function to minimize</param>
        /// <param name="initVal">Value of function at 0</param>
        /// <param name="initDeriv">Derivative of function at 0</param>
        /// <returns>Minimizing value</returns>
        public float Minimize(DiffFunc1D f, float initVal, float initDeriv)
        {
            Contracts.Check(initDeriv < 0, "Cannot search in direction of ascent!");

            float dummy;
            for (_step *= 2; ; _step /= 2)
            {
                float newVal = f(_step, out dummy);
                if (newVal <= initVal + _c1 * _step * initDeriv)
                    return _step;
            }
        }
    }

    // REVIEW: This is test code. Is this useless at this point, or
    // possibly something we should put into our unit tests?
    internal static class Test
    {
        private static VBuffer<float> _c1;
        private static VBuffer<float> _c2;
        private static VBuffer<float> _c3;

        private static float QuadTest(float x, out float deriv)
        {
            const float a = (float)1.32842;
            const float b = (float)(-28.38092);
            const float c = 93;
            deriv = a * x + b;
            return (float)0.5 * a * x * x + b * x + c;
        }

        private static float LogTest(float x, out float deriv)
        {
            double e = Math.Exp(x);
            deriv = (float)(-1.0 / (1.0 + e) + e / (1.0 + e) - 0.5);
            return (float)(Math.Log(1 + 1.0 / e) + Math.Log(1 + e) - 0.5 * x);
        }

        private static float QuadTest2D(in VBuffer<float> x, ref VBuffer<float> grad, IProgressChannelProvider progress = null)
        {
            float d1 = VectorUtils.DotProduct(in x, in _c1);
            float d2 = VectorUtils.DotProduct(in x, in _c2);
            float d3 = VectorUtils.DotProduct(in x, in _c3);
            _c3.CopyTo(ref grad);
            VectorUtils.AddMult(in _c1, d1, ref grad);
            VectorUtils.AddMult(in _c2, d2, ref grad);
            return (float)0.5 * (d1 * d1 + d2 * d2) + d3 + 55;
        }

        private static void StochasticQuadTest2D(in VBuffer<float> x, ref VBuffer<float> grad)
        {
            QuadTest2D(in x, ref grad);
        }

        private static void CreateWrapped(out VBuffer<float> vec, params float[] values)
        {
            vec = new VBuffer<float>(Utils.Size(values), values);
        }

        static Test()
        {
            CreateWrapped(out _c1, 1, 2);
            CreateWrapped(out _c2, -2, -3);
            CreateWrapped(out _c3, -1, 3);
        }

        private static void RunTest(DiffFunc1D f)
        {
            CubicInterpLineSearch cils = new CubicInterpLineSearch((float)1e-8);
            float val;
            float deriv;
            val = f(0, out deriv);
            float min = cils.Minimize(f, val, deriv);
            val = f(min, out deriv);
            Console.WriteLine(deriv);
            GoldenSectionSearch gss = new GoldenSectionSearch((float)1e-8);
            min = gss.Minimize(f);
            val = f(min, out deriv);
            Console.WriteLine(deriv);
        }

        public static void Main(string[] argv)
        {
            RunTest(QuadTest);
            RunTest(LogTest);

            VBuffer<float> grad = VBufferUtils.CreateEmpty<float>(2);
            int n = 0;
            bool print = false;
            DTerminate term =
                (in VBuffer<float> x) =>
                {
                    QuadTest2D(in x, ref grad);
                    float norm = VectorUtils.Norm(grad);
                    if (++n % 1000 == 0 || print)
                        Console.WriteLine("{0}\t{1}", n, norm);
                    return (norm < 1e-5);
                };
            SgdOptimizer sgdo = new SgdOptimizer(term, SgdOptimizer.RateScheduleType.Constant, false, 100, 1, (float)0.99);
            VBuffer<float> init;
            CreateWrapped(out init, 0, 0);
            VBuffer<float> ans = default(VBuffer<float>);
            sgdo.Minimize(StochasticQuadTest2D, ref init, ref ans);
            QuadTest2D(in ans, ref grad);
            Console.WriteLine(VectorUtils.Norm(grad));
            Console.WriteLine();
            Console.WriteLine();
            n = 0;
            GDOptimizer gdo = new GDOptimizer(term, null, true);
            print = true;
            CreateWrapped(out init, 0, 0);
            gdo.Minimize(QuadTest2D, in init, ref ans);
            QuadTest2D(in ans, ref grad);
            Console.WriteLine(VectorUtils.Norm(grad));
        }
    }
}
