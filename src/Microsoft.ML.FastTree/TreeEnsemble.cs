// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// A list of <see cref="RegressionTreeBase"/>'s derived class. To compute the output value of a
    /// <see cref="TreeEnsemble{T}"/>, we need to compute the output values of all trees in <see cref="Trees"/>,
    /// scale those values via <see cref="TreeWeights"/>, and finally sum the scaled values and <see cref="Bias"/> up.
    /// </summary>
    public abstract class TreeEnsemble<T> where T : RegressionTreeBase
    {
        /// <summary>
        /// When doing prediction, this is a value added to the weighted sum of all <see cref="Trees"/>' outputs.
        /// </summary>
        public double Bias { get; }

        /// <summary>
        /// <see cref="TreeWeights"/>[i] is the i-th <see cref="RegressionTreeBase"/>'s weight in <see cref="Trees"/>.
        /// </summary>
        public IReadOnlyList<double> TreeWeights { get; }

        /// <summary>
        /// <see cref="Trees"/>[i] is the i-th <see cref="RegressionTreeBase"/> in <see cref="Trees"/>.
        /// </summary>
        public IReadOnlyList<T> Trees { get; }

        private protected TreeEnsemble(IEnumerable<T> trees, IEnumerable<double> treeWeights, double bias)
        {
            Bias = bias;
            TreeWeights = treeWeights.ToList();
            Trees = trees.ToList();
        }
    }

    public sealed class RegressionTreeEnsemble : TreeEnsemble<RegressionTree>
    {
        internal RegressionTreeEnsemble(IEnumerable<RegressionTree> trees, IEnumerable<double> treeWeights, double bias)
            : base(trees, treeWeights, bias)
        {
        }
    }

    public sealed class QuantileRegressionTreeEnsemble : TreeEnsemble<QuantileRegressionTree>
    {
        internal QuantileRegressionTreeEnsemble(IEnumerable<QuantileRegressionTree> trees, IEnumerable<double> treeWeights, double bias)
            : base(trees, treeWeights, bias)
        {
        }
    }
}
