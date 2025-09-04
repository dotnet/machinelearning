// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Microsoft.ML.IsolationForest
{
    /// <summary>
    /// Plain C# Isolation Forest model (training + scoring).
    /// netstandard2.0-friendly and style-clean.
    /// </summary>
    public sealed class IsolationForestModel
    {
        private readonly int _nTrees;
        private readonly int _sampleSize;
        private readonly int _maxHeight;
        private readonly int _seed;

        private readonly List<IsolationTree> _trees = [];
        private double _threshold = double.NaN;

        public IsolationForestModel(int nTrees = 100, int sampleSize = 256, int seed = 42)
        {
            if (nTrees <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(nTrees));
            }

            if (sampleSize <= 1)
            {
                throw new ArgumentOutOfRangeException(nameof(sampleSize));
            }

            _nTrees = nTrees;
            _sampleSize = sampleSize;
            _maxHeight = (int)Math.Ceiling(Math.Log(sampleSize, 2.0)); // no Math.Log2 in netstd2.0
            _seed = seed;
        }

        public void Fit(double[][] x, double? contamination = null, bool parallel = true)
        {
            if (x == null || x.Length == 0)
            {
                throw new ArgumentException("Empty dataset.", nameof(x));
            }

            var rng = new Random(_seed);
            _trees.Clear();
            var treeSeeds = new int[_nTrees];
            for (var i = 0; i < _nTrees; i++)
            {
                treeSeeds[i] = rng.Next();
            }

            if (parallel)
            {
                var built = new IsolationTree[_nTrees];
                Parallel.For(0, _nTrees, i =>
                {
                    var idx = ReservoirSampleIndices(x.Length, _sampleSize, new Random(treeSeeds[i]));
                    var sample = new double[idx.Length][];
                    for (var k = 0; k < idx.Length; k++)
                    {
                        sample[k] = x[idx[k]];
                    }
                    built[i] = IsolationTree.Build(sample, _maxHeight, new Random(treeSeeds[i]));
                });
                _trees.AddRange(built);
            }
            else
            {
                for (var i = 0; i < _nTrees; i++)
                {
                    var idx = ReservoirSampleIndices(x.Length, _sampleSize, new Random(treeSeeds[i]));
                    var sample = new double[idx.Length][];
                    for (var k = 0; k < idx.Length; k++)
                    {
                        sample[k] = x[idx[k]];
                    }

                    _trees.Add(IsolationTree.Build(sample, _maxHeight, new Random(treeSeeds[i])));
                }
            }

            if (contamination.HasValue && contamination.Value > 0 && contamination.Value < 1)
            {
                var scores = ScoreAll(x);
                Array.Sort(scores);
                var k = (int)Math.Floor((1.0 - contamination.Value) * scores.Length);
                if (k < 0)
                {
                    k = 0;
                }

                if (k > scores.Length - 1)
                {
                    k = scores.Length - 1;
                }

                _threshold = scores[k];
            }
        }

        public double Score(double[] row)
        {
            if (_trees.Count == 0)
            {
                throw new InvalidOperationException("Call Fit first.");
            }

            var sum = 0.0;
            for (var i = 0; i < _trees.Count; i++)
            {
                sum += _trees[i].PathLength(row);
            }

            var avgPath = sum / _trees.Count;
            var cn = C(_sampleSize);
            return Math.Pow(2.0, -avgPath / cn);
        }

        public double[] ScoreAll(double[][] x)
        {
            var res = new double[x.Length];
            for (var i = 0; i < x.Length; i++)
            {
                res[i] = Score(x[i]);
            }

            return res;
        }

        public bool Predict(double[] row, double? thresholdOverride = null)
        {
            var th = thresholdOverride ?? _threshold;
            if (double.IsNaN(th))
            {
                throw new InvalidOperationException("No threshold set.");
            }

            return Score(row) >= th;
        }

        private static double C(int n)
        {
            if (n <= 1)
            {
                return 0.0;
            }

            const double gamma = 0.57721566490153286060; // Euler–Mascheroni
            var hn1 = Math.Log(n - 1) + gamma;

            return 2.0 * hn1 - 2.0 * (n - 1.0) / n;
        }

        private static int[] ReservoirSampleIndices(int population, int k, Random rng)
        {
            k = Math.Min(k, population);
            var res = new int[k];
            for (var i = 0; i < k; i++)
            {
                res[i] = i;
            }

            for (var i = k; i < population; i++)
            {
                var j = rng.Next(i + 1);
                if (j < k)
                {
                    res[j] = i;
                }
            }

            return res;
        }

        // Inner tree
        private sealed class IsolationTree
        {
            private readonly Node _root;

            private IsolationTree(Node root)
            {
                _root = root;
            }

            public static IsolationTree Build(double[][] x, int maxHeight, Random rng)
            {
                return new IsolationTree(BuildNode(x, 0, maxHeight, rng));
            }

            public double PathLength(double[] row)
            {
                return Path(row, _root, 0);
            }

            private static Node BuildNode(double[][] x, int h, int maxH, Random rng)
            {
                var n = x.Length;
                var d = x[0].Length;

                if (n <= 1 || h >= maxH || IsConstant(x))
                {
                    return new Node { Size = n, IsExternal = true };
                }

                int feature;
                double min;
                double max;
                var tries = 0;

                do
                {
                    feature = rng.Next(d);
                    MinMax(x, feature, out min, out max);
                    tries++;
                    if (tries > d * 4)
                    {
                        return new Node { Size = n, IsExternal = true };
                    }
                } while (max <= min);

                var split = min + rng.NextDouble() * (max - min);
                var left = new List<double[]>(n);
                var right = new List<double[]>(n);
                for (var i = 0; i < x.Length; i++)
                {
                    if (x[i][feature] < split)
                    {
                        left.Add(x[i]);
                    }
                    else
                    {
                        right.Add(x[i]);
                    }
                }

                if (left.Count == 0 || right.Count == 0)
                {
                    return new Node { Size = n, IsExternal = true };
                }

                return new Node
                {
                    Feature = feature,
                    Split = split,
                    Left = BuildNode(left.ToArray(), h + 1, maxH, rng),
                    Right = BuildNode(right.ToArray(), h + 1, maxH, rng),
                    Size = n,
                    IsExternal = false
                };
            }

            private static void MinMax(double[][] x, int j, out double min, out double max)
            {
                min = double.PositiveInfinity;
                max = double.NegativeInfinity;
                for (var i = 0; i < x.Length; i++)
                {
                    var v = x[i][j];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }

            private static bool IsConstant(double[][] x)
            {
                const double epsilon = 1e-9; // tolerance for equality
                var d = x[0].Length;
                for (var j = 0; j < d; j++)
                {
                    var v0 = x[0][j];
                    for (var i = 1; i < x.Length; i++)
                    {
                        if (Math.Abs(x[i][j] - v0) > epsilon)
                        {
                            return false;
                        }
                    }
                }

                return true;
            }

            private static double ExternalPathLength(int n)
            {
                if (n <= 1)
                {
                    return 0;
                }

                const double gamma = 0.57721566490153286060;
                var hn1 = Math.Log(n - 1) + gamma;

                return 2.0 * hn1 - 2.0 * (n - 1.0) / n;
            }

            private static double Path(double[] row, Node node, int depth)
            {
                if (node.IsExternal)
                {
                    return depth + ExternalPathLength(node.Size);
                }

                if (row[node.Feature] < node.Split)
                {
                    return Path(row, node.Left, depth + 1);
                }
                else
                {
                    return Path(row, node.Right, depth + 1);
                }
            }

            private sealed class Node
            {
                public int Feature;
                public double Split;
                public Node Left;
                public Node Right;
                public int Size;
                public bool IsExternal;
            }
        }
    }
}
