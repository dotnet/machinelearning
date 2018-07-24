// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Numeric
{
    /// <summary>
    /// A delegate for functions with gradients.
    /// </summary>
    /// <param name="input">The point at which to evaluate the function</param>
    /// <param name="gradient">The gradient vector, which must be filled in (its initial contents are undefined)</param>
    /// <param name="progress">The progress channel provider that can be used to report calculation progress. Can be null.</param>
    /// <returns>The value of the function</returns>
    public delegate Float DifferentiableFunction(ref VBuffer<Float> input, ref VBuffer<Float> gradient, IProgressChannelProvider progress);

    /// <summary>
    /// A delegate for indexed sets of functions with gradients.
    ///
    /// REVIEW: I didn't add an <see cref="IProgressChannelProvider"/> here, since it looks like this code is not actually
    /// accessed from anywhere. Maybe it should go away?
    /// </summary>
    /// <param name="index">The index of the function</param>
    /// <param name="input">The point at which to evaluate the function</param>
    /// <param name="gradient">The gradient vector, which must be filled in (its initial contents are undefined)</param>
    /// <returns>The value of the function</returns>
    public delegate Float IndexedDifferentiableFunction(int index, ref VBuffer<Float> input, ref VBuffer<Float> gradient);

    /// <summary>
    /// Class to aggregate an indexed differentiable function into a single function, in parallel
    /// </summary>
    public class DifferentiableFunctionAggregator
    {
        private readonly IndexedDifferentiableFunction _func;
        private readonly int _maxIndex;
        private readonly int _threads;
        private readonly int _dim;

        private readonly VBuffer<Float>[] _tempGrads;
        private VBuffer<Float> _input;
        private readonly Float[] _tempVals;
        private readonly AutoResetEvent[] _threadFinished;

        /// <summary>
        /// Creates a DifferentiableFunctionAggregator
        /// </summary>
        /// <param name="func">Indexed function to use</param>
        /// <param name="dim">Dimensionality of the function</param>
        /// <param name="maxIndex">Max index of the function</param>
        /// <param name="threads">Number of threads to use</param>
        public DifferentiableFunctionAggregator(IndexedDifferentiableFunction func, int dim, int maxIndex, int threads = 64)
        {
            _func = func;
            _dim = dim;

            _maxIndex = maxIndex;

            if (threads > maxIndex)
                threads = maxIndex;
            if (threads > 64)
                threads = 64;
            _threads = threads;

            _tempGrads = new VBuffer<Float>[threads];
            _threadFinished = new AutoResetEvent[threads];
            for (int i = 0; i < threads; ++i)
                _threadFinished[i] = new AutoResetEvent(false);

            _tempVals = new Float[threads];
        }

        private void Eval(object chunkIndexObj)
        {
            int chunkIndex = (int)chunkIndexObj;
            int chunkSize = _maxIndex / _threads;
            int bigChunkSize = chunkSize + 1;
            int numBigChunks = _maxIndex % _threads;
            int from;
            int to;
            if (chunkIndex < numBigChunks)
            {
                from = bigChunkSize * chunkIndex;
                to = from + bigChunkSize;
            }
            else
            {
                from = bigChunkSize * numBigChunks + chunkSize * (chunkIndex - numBigChunks);
                to = from + chunkSize;
            }

            _tempVals[chunkIndex] = 0;
            VectorUtils.ScaleBy(ref _tempGrads[chunkIndex], 0);

            VBuffer<Float> tempGrad = default(VBuffer<Float>);
            for (int i = from; i < to; ++i)
            {
                tempGrad = new VBuffer<Float>(0, 0, tempGrad.Values, tempGrad.Indices);
                _tempVals[chunkIndex] += _func(i, ref _input, ref tempGrad);
                if (_tempGrads[chunkIndex].Length == 0)
                    tempGrad.CopyTo(ref _tempGrads[chunkIndex]);
                else
                    VectorUtils.Add(ref tempGrad, ref _tempGrads[chunkIndex]);
            }

            _threadFinished[chunkIndex].Set();
        }

        /// <summary>
        /// Evaluate and sum the function over all indices, in parallel
        /// </summary>
        /// <param name="input">The point at which to evaluate the function</param>
        /// <param name="gradient">The gradient vector, which must be filled in (its initial contents are undefined)</param>
        /// <returns>Function value</returns>
        public Float Eval(ref VBuffer<Float> input, ref VBuffer<Float> gradient)
        {
            _input = input;

            for (int c = 0; c < _threads; ++c)
            {
                ThreadPool.QueueUserWorkItem(Eval, c);
            }

            AutoResetEvent.WaitAll(_threadFinished);

            VectorUtils.ScaleBy(ref gradient, 0);
            Float value = 0;
            for (int c = 0; c < _threads; ++c)
            {
                if (gradient.Length == 0)
                    _tempGrads[c].CopyTo(ref gradient);
                else
                    VectorUtils.Add(ref _tempGrads[c], ref gradient);
                value += _tempVals[c];
            }

            return value;
        }
    }

    // REVIEW: The following GradientTester is very inefficient, and is not called
    // from anywhere. Consider deletion.
    /// <summary>
    /// A class for testing the gradient of DifferentiableFunctions, useful for debugging
    /// </summary>
    /// <remarks>
    /// Works by comparing the reported gradient to the numerically computed gradient.
    /// If the gradient is correct, the return value should be small (order of 1e-6).
    /// May have false negatives if extreme values cause the numeric gradient to be off,
    /// e.g. if the norm of x is very large, or if the gradient is changing rapidly at x.
    /// </remarks>
    public static class GradientTester
    {
        // approximately u^(1/3), where u is the unit roundoff ~ 1.1e-16.
        // the optimal value of eps for the central difference approximation, Nocedal & Wright
        private const Float Eps = (Float)4.79e-6;

        private static Random _r = new Random(5);

        /// <summary>
        /// Tests the gradient reported by f.
        /// </summary>
        /// <param name="f">function to test</param>
        /// <param name="x">point at which to test</param>
        /// <returns>maximum normalized difference between analytic and numeric directional derivative over multiple tests</returns>
        public static Float Test(DifferentiableFunction f, ref VBuffer<Float> x)
        {
            // REVIEW: Delete this method?
            return Test(f, ref x, false);
        }

        /// <summary>
        /// Tests the gradient reported by f.
        /// </summary>
        /// <param name="f">function to test</param>
        /// <param name="x">point at which to test</param>
        /// <param name="quiet">If false, outputs detailed info.</param>
        /// <returns>maximum normalized difference between analytic and numeric directional derivative over multiple tests</returns>
        public static Float Test(DifferentiableFunction f, ref VBuffer<Float> x, bool quiet)
        {
            // REVIEW: Delete this method?
            VBuffer<Float> grad = default(VBuffer<Float>);
            VBuffer<Float> newGrad = default(VBuffer<Float>);
            VBuffer<Float> newX = default(VBuffer<Float>);
            Float normX = VectorUtils.Norm(x);
            f(ref x, ref grad, null);

            if (!quiet)
                Console.WriteLine(Header);

            Float maxNormDiff = Float.NegativeInfinity;

            int numIters = Math.Min((int)x.Length, 10);
            int maxDirCount = Math.Min((int)x.Length / 2, 100);

            for (int n = 1; n <= numIters; n++)
            {
                int dirCount = Math.Min(n * 10, maxDirCount);
                List<int> indices = new List<int>(dirCount);
                List<Float> values = new List<Float>(dirCount);
                for (int i = 0; i < dirCount; i++)
                {
                    int index = _r.Next((int)x.Length);
                    while (indices.IndexOf(index) >= 0)
                        index = _r.Next((int)x.Length);
                    indices.Add(index);
                    values.Add(SampleFromGaussian(_r));
                }
                VBuffer<Float> dir = new VBuffer<Float>(x.Length, values.Count, values.ToArray(), indices.ToArray());

                Float norm = VectorUtils.Norm(dir);
                VectorUtils.ScaleBy(ref dir, 1 / norm);

                VectorUtils.AddMultInto(ref x, Eps, ref dir, ref newX);
                Float rVal = f(ref newX, ref newGrad, null);

                VectorUtils.AddMultInto(ref x, -Eps, ref dir, ref newX);
                Float lVal = f(ref newX, ref newGrad, null);

                Float dirDeriv = VectorUtils.DotProduct(ref grad, ref dir);
                Float numDeriv = (rVal - lVal) / (2 * Eps);

                Float normDiff = Math.Abs(1 - numDeriv / dirDeriv);
                Float diff = numDeriv - dirDeriv;
                if (!quiet)
                    Console.WriteLine("{0,-9}{1,-18:0.0000e0}{2,-18:0.0000e0}{3,-15:0.0000e0}{4,0:0.0000e0}", n, numDeriv, dirDeriv, diff, normDiff);

                maxNormDiff = Math.Max(maxNormDiff, normDiff);
            }

            return maxNormDiff;
        }

        /// <summary>
        /// The head of the test output
        /// </summary>
        public readonly static string Header = "Trial    Numeric deriv     Analytic deriv    Difference     Normalized";

        /// <summary>
        /// Tests the gradient using finite differences on each axis (appropriate for small functions)
        /// </summary>
        /// <param name="f"></param>
        /// <param name="x"></param>
        public static void TestAllCoords(DifferentiableFunction f, ref VBuffer<Float> x)
        {
            // REVIEW: Delete this method?
            VBuffer<Float> grad = default(VBuffer<Float>);
            VBuffer<Float> newGrad = default(VBuffer<Float>);
            VBuffer<Float> newX = default(VBuffer<Float>);
            Float val = f(ref x, ref grad, null);
            Float normX = VectorUtils.Norm(x);

            Console.WriteLine(Header);

            Random r = new Random(5);

            VBuffer<Float> dir = new VBuffer<Float>(x.Length, 1, new Float[] { 1 }, new int[] { 0 });
            for (int n = 0; n < x.Length; n++)
            {
                dir.Values[0] = n;
                VectorUtils.AddMultInto(ref x, Eps, ref dir, ref newX);
                Float rVal = f(ref newX, ref newGrad, null);

                VectorUtils.AddMultInto(ref x, -Eps, ref dir, ref newX);
                Float lVal = f(ref newX, ref newGrad, null);

                Float dirDeriv = VectorUtils.DotProduct(ref grad, ref dir);
                Float numDeriv = (rVal - lVal) / (2 * Eps);

                Float normDiff = Math.Abs(1 - numDeriv / dirDeriv);
                Float diff = numDeriv - dirDeriv;
                if (diff != 0)
                    Console.WriteLine("{0,-9}{1,-18:0.0000e0}{2,-18:0.0000e0}{3,-15:0.0000e0}{4,0:0.0000e0}", n, numDeriv, dirDeriv, diff, normDiff);
            }
        }

        /// <summary>
        /// Tests the gradient using finite differences on each axis in the list
        /// </summary>
        /// <param name="f">Function to test</param>
        /// <param name="x">Point at which to test</param>
        /// <param name="coords">List of coordinates to test</param>
        public static void TestCoords(DifferentiableFunction f, ref VBuffer<Float> x, IList<int> coords)
        {
            // REVIEW: Delete this method?
            VBuffer<Float> grad = default(VBuffer<Float>);
            VBuffer<Float> newGrad = default(VBuffer<Float>);
            VBuffer<Float> newX = default(VBuffer<Float>);
            Float val = f(ref x, ref grad, null);
            Float normX = VectorUtils.Norm(x);

            Console.WriteLine(Header);

            Random r = new Random(5);

            VBuffer<Float> dir = new VBuffer<Float>(x.Length, 1, new Float[] { 1 }, new int[] { 0 });
            foreach (int n in coords)
            {
                dir.Values[0] = n;
                VectorUtils.AddMultInto(ref x, Eps, ref dir, ref newX);
                Float rVal = f(ref newX, ref newGrad, null);

                VectorUtils.AddMultInto(ref x, -Eps, ref dir, ref newX);
                Float lVal = f(ref newX, ref newGrad, null);

                Float dirDeriv = VectorUtils.DotProduct(ref grad, ref dir);
                Float numDeriv = (rVal - lVal) / (2 * Eps);

                Float normDiff = Math.Abs(1 - numDeriv / dirDeriv);
                Float diff = numDeriv - dirDeriv;
                Console.WriteLine("{0,-9}{1,-18:0.0000e0}{2,-18:0.0000e0}{3,-15:0.0000e0}{4,0:0.0000e0}", n, numDeriv, dirDeriv, diff, normDiff);
            }
        }

        /// <summary>
        /// Tests the gradient reported by <paramref name="f"/>.
        /// </summary>
        /// <param name="f">Function to test</param>
        /// <param name="x">Point at which to test</param>
        /// <param name="dir">Direction to test derivative</param>
        /// <param name="quiet">Whether to disable output</param>
        /// <param name="newGrad">This is a reusable working buffer for intermediate calculations</param>
        /// <param name="newX">This is a reusable working buffer for intermediate calculations</param>
        /// <returns>Normalized difference between analytic and numeric directional derivative</returns>
        public static Float Test(DifferentiableFunction f, ref VBuffer<Float> x, ref VBuffer<Float> dir, bool quiet,
            ref VBuffer<Float> newGrad, ref VBuffer<Float> newX)
        {
            Float normDir = VectorUtils.Norm(dir);

            Float val = f(ref x, ref newGrad, null);
            Float dirDeriv = VectorUtils.DotProduct(ref newGrad, ref dir);

            Float scaledEps = Eps / normDir;

            VectorUtils.AddMultInto(ref x, scaledEps, ref dir, ref newX);
            Float rVal = f(ref newX, ref newGrad, null);

            VectorUtils.AddMultInto(ref x, -scaledEps, ref dir, ref newX);
            Float lVal = f(ref newX, ref newGrad, null);

            Float numDeriv = (rVal - lVal) / (2 * scaledEps);

            Float normDiff = Math.Abs(1 - numDeriv / dirDeriv);
            Float diff = numDeriv - dirDeriv;
            if (!quiet)
                Console.WriteLine("{0,-18:0.0000e0}{1,-18:0.0000e0}{2,-15:0.0000e0}{3,0:0.0000e0}", numDeriv, dirDeriv, diff, normDiff);

            return normDiff;
        }

        private static Float SampleFromGaussian(Random r)
        {
            double a = r.NextDouble();
            double b = r.NextDouble();
            return (Float)(Math.Sqrt(-2 * Math.Log(a)) * MathUtils.Cos(2 * Math.PI * b));
        }
    }
}
