// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// A container base class for exposing <see cref="InternalRegressionTree"/>'s and
    /// <see cref="InternalQuantileRegressionTree"/>'s attributes to users.
    /// This class should not be mutable, so it contains a lot of read-only members.
    /// </summary>
    public abstract class RegressionTreeBase
    {
        /// <summary>
        /// <see cref="RegressionTreeBase"/> is an immutable wrapper over <see cref="_tree"/> for exposing some tree's
        /// attribute to users.
        /// </summary>
        private readonly InternalRegressionTree _tree;

        /// <summary>
        /// See <see cref="LeftChild"/>.
        /// </summary>
        private readonly ImmutableArray<int> _lteChild;
        /// <summary>
        /// See <see cref="RightChild"/>.
        /// </summary>
        private readonly ImmutableArray<int> _gtChild;
        /// <summary>
        /// See <see cref="NumericalSplitFeatureIndexes"/>.
        /// </summary>
        private readonly ImmutableArray<int> _numericalSplitFeatureIndexes;
        /// <summary>
        /// See <see cref="NumericalSplitThresholds"/>.
        /// </summary>
        private readonly ImmutableArray<float> _numericalSplitThresholds;
        /// <summary>
        /// See <see cref="CategoricalSplitFlags"/>.
        /// </summary>
        private readonly ImmutableArray<bool> _categoricalSplitFlags;
        /// <summary>
        /// See <see cref="LeafValues"/>.
        /// </summary>
        private readonly ImmutableArray<double> _leafValues;
        /// <summary>
        /// See <see cref="SplitGains"/>.
        /// </summary>
        private readonly ImmutableArray<double> _splitGains;

        /// <summary>
        /// <see cref="LeftChild"/>[i] is the i-th node's child index used when
        /// (1) the numerical feature indexed by <see cref="NumericalSplitFeatureIndexes"/>[i] is less than or equal
        /// to the threshold <see cref="NumericalSplitThresholds"/>[i], or
        /// (2) the categorical features indexed by <see cref="GetCategoricalCategoricalSplitFeatureRangeAt(int)"/>'s
        /// returned value with nodeIndex=i is NOT a sub-set of <see cref="GetCategoricalSplitFeaturesAt(int)"/> with
        /// nodeIndex=i.
        /// Note that the case (1) happens only when <see cref="CategoricalSplitFlags"/>[i] is true and otherwise (2)
        /// occurs. A non-negative returned value means a node (i.e., not a leaf); for example, 2 means the 3rd node in
        /// the underlying <see cref="_tree"/>. A negative returned value means a leaf; for example, -1 stands for the
        /// <see langword="~"/>(-1)-th leaf in the underlying <see cref="_tree"/>. Note that <see langword="~"/> is the
        /// bitwise complement operator in C#; for details, see
        /// https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/bitwise-complement-operator.
        /// </summary>
        public IReadOnlyList<int> LeftChild => _lteChild;

        /// <summary>
        /// <see cref="RightChild"/>[i] is the i-th node's child index used when the two conditions, (1) and (2),
        /// described in <see cref="LeftChild"/>'s document are not true. Its return value follows the format
        /// used in <see cref="LeftChild"/>.
        /// </summary>
        public IReadOnlyList<int> RightChild => _gtChild;

        /// <summary>
        /// <see cref="NumericalSplitFeatureIndexes"/>[i] is the feature index used the splitting function of the
        /// i-th node. This value is valid only if <see cref="CategoricalSplitFlags"/>[i] is false.
        /// </summary>
        public IReadOnlyList<int> NumericalSplitFeatureIndexes => _numericalSplitFeatureIndexes;

        /// <summary>
        /// <see cref="NumericalSplitThresholds"/>[i] is the threshold on feature indexed by
        /// <see cref="NumericalSplitFeatureIndexes"/>[i], where i is the i-th node's index
        /// (for example, i is 1 for the 2nd node in <see cref="_tree"/>).
        /// </summary>
        public IReadOnlyList<float> NumericalSplitThresholds => _numericalSplitThresholds;

        /// <summary>
        /// Determine the types of splitting function. If <see cref="CategoricalSplitFlags"/>[i] is true, the i-th
        /// node's uses categorical splitting function. Otherwise, traditional numerical split is used.
        /// </summary>
        public IReadOnlyList<bool> CategoricalSplitFlags => _categoricalSplitFlags;

        /// <summary>
        /// <see cref="LeafValues"/>[i] is the learned value at the i-th leaf.
        /// </summary>
        public IReadOnlyList<double> LeafValues => _leafValues;

        /// <summary>
        /// Return categorical thresholds used at node indexed by nodeIndex. If the considered input feature does NOT
        /// matche any of values returned by <see cref="GetCategoricalSplitFeaturesAt(int)"/>, we call it a
        /// less-than-threshold event and therefore <see cref="LeftChild"/>[nodeIndex] is the child node that input
        /// should go next. The returned value is valid only if <see cref="CategoricalSplitFlags"/>[nodeIndex] is true.
        /// </summary>
        public IReadOnlyList<int> GetCategoricalSplitFeaturesAt(int nodeIndex)
        {
            if (nodeIndex < 0 || nodeIndex >= NumberOfNodes)
                throw Contracts.Except($"The input index, {nodeIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumberOfNodes} (exclusive).");

            if (_tree.CategoricalSplitFeatures == null || _tree.CategoricalSplitFeatures[nodeIndex] == null)
                return new List<int>(); // Zero-length vector.
            else
                return _tree.CategoricalSplitFeatures[nodeIndex];
        }
        /// <summary>
        /// Return categorical thresholds' range used at node indexed by nodeIndex. A categorical split at node indexed
        /// by nodeIndex can consider multiple consecutive input features at one time; their range is specified by
        /// <see cref="GetCategoricalCategoricalSplitFeatureRangeAt(int)"/>. The returned value is always a 2-element
        /// array; its 1st element is the starting index and its 2nd element is the endining index of a feature segment.
        /// The returned value is valid only if <see cref="CategoricalSplitFlags"/>[nodeIndex] is true.
        /// </summary>
        public IReadOnlyList<int> GetCategoricalCategoricalSplitFeatureRangeAt(int nodeIndex)
        {
            if (nodeIndex < 0 || nodeIndex >= NumberOfNodes)
                throw Contracts.Except($"The input node index, {nodeIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumberOfNodes} (exclusive).");

            if (_tree.CategoricalSplitFeatureRanges == null || _tree.CategoricalSplitFeatureRanges[nodeIndex] == null)
                return new List<int>(); // Zero-length vector.
            else
                return _tree.CategoricalSplitFeatureRanges[nodeIndex];
        }

        /// <summary>
        /// The gains obtained by splitting data at nodes. Its i-th value is computed from to the split at the i-th node.
        /// </summary>
        public IReadOnlyList<double> SplitGains => _splitGains;

        /// <summary>
        /// Number of leaves in the tree. Note that <see cref="NumberOfLeaves"/> does not take non-leaf nodes into account.
        /// </summary>
        public int NumberOfLeaves => _tree.NumLeaves;

        /// <summary>
        /// Number of nodes in the tree. This doesn't include any leaves. For example, a tree with node0->node1,
        /// node0->leaf3, node1->leaf1, node1->leaf2, <see cref="NumberOfNodes"/> and <see cref="NumberOfLeaves"/> should
        /// be 2 and 3, respectively.
        /// </summary>
        // A visualization of the example mentioned in this doc string.
        //         node0
        //         /  \
        //     node1 leaf3
        //     /  \
        // leaf1 leaf2
        // The index of leaf starts with 1 because interally we use "-1" as the 1st leaf's index, "-2" for the 2nd leaf's index, and so on.
        public int NumberOfNodes => _tree.NumNodes;

        internal RegressionTreeBase(InternalRegressionTree tree)
        {
            _tree = tree;

            _lteChild = ImmutableArray.Create(_tree.LteChild, 0, _tree.NumNodes);
            _gtChild = ImmutableArray.Create(_tree.GtChild, 0, _tree.NumNodes);

            _numericalSplitFeatureIndexes = ImmutableArray.Create(_tree.SplitFeatures, 0, _tree.NumNodes);
            _numericalSplitThresholds = ImmutableArray.Create(_tree.RawThresholds, 0, _tree.NumNodes);
            _categoricalSplitFlags = ImmutableArray.Create(_tree.CategoricalSplit, 0, _tree.NumNodes);
            _leafValues = ImmutableArray.Create(_tree.LeafValues, 0, _tree.NumLeaves);
            _splitGains = ImmutableArray.Create(_tree.SplitGains, 0, _tree.NumNodes);
        }
    }

    /// <summary>
    /// A container class for exposing <see cref="InternalRegressionTree"/>'s attributes to users.
    /// This class should not be mutable, so it contains a lot of read-only members. Note that
    /// <see cref="RegressionTree"/> is identical to <see cref="RegressionTreeBase"/> but in
    /// another derived class <see cref="QuantileRegressionTree"/> some attributes are added.
    /// </summary>
    public sealed class RegressionTree : RegressionTreeBase
    {
        internal RegressionTree(InternalRegressionTree tree) : base(tree) { }
    }

    /// <summary>
    /// A container class for exposing <see cref="InternalQuantileRegressionTree"/>'s attributes to users.
    /// This class should not be mutable, so it contains a lot of read-only members. In addition to
    /// things inherited from <see cref="RegressionTreeBase"/>, we add <see cref="GetLeafSamplesAt(int)"/>
    /// and <see cref="GetLeafSampleWeightsAt(int)"/> to expose (sub-sampled) training labels falling into
    /// the leafIndex-th leaf and their weights.
    /// </summary>
    public sealed class QuantileRegressionTree : RegressionTreeBase
    {
        /// <summary>
        /// Sample labels from training data. <see cref="_leafSamples"/>[i] stores the labels falling into the
        /// i-th leaf.
        /// </summary>
        private readonly double[][] _leafSamples;

        /// <summary>
        /// Sample labels' weights from training data. <see cref="_leafSampleWeights"/>[i] stores the weights for
        /// labels falling into the i-th leaf. <see cref="_leafSampleWeights"/>[i][j] is the weight of
        /// <see cref="_leafSamples"/>[i][j].
        /// </summary>
        private readonly double[][] _leafSampleWeights;

        /// <summary>
        /// Return the training labels falling into the specified leaf.
        /// </summary>
        /// <param name="leafIndex">The index of the specified leaf.</param>
        /// <returns>Training labels</returns>
        public IReadOnlyList<double> GetLeafSamplesAt(int leafIndex)
        {
            if (leafIndex < 0 || leafIndex >= NumberOfLeaves)
                throw Contracts.Except($"The input leaf index, {leafIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumberOfLeaves} (exclusive).");

            // _leafSample always contains valid values assigned in constructor.
            return _leafSamples[leafIndex];
        }

        /// <summary>
        /// Return the weights for training labels falling into the specified leaf. If <see cref="GetLeafSamplesAt"/>
        /// and <see cref="GetLeafSampleWeightsAt"/> use the same input, the i-th returned value of this function is
        /// the weight of the i-th label in <see cref="GetLeafSamplesAt"/>.
        /// </summary>
        /// <param name="leafIndex">The index of the specified leaf.</param>
        /// <returns>Training labels' weights</returns>
        public IReadOnlyList<double> GetLeafSampleWeightsAt(int leafIndex)
        {
            if (leafIndex < 0 || leafIndex >= NumberOfLeaves)
                throw Contracts.Except($"The input leaf index, {leafIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumberOfLeaves} (exclusive).");

            // _leafSampleWeights always contains valid values assigned in constructor.
            return _leafSampleWeights[leafIndex];
        }

        internal QuantileRegressionTree(InternalQuantileRegressionTree tree) : base(tree)
        {
            tree.ExtractLeafSamplesAndTheirWeights(out _leafSamples, out _leafSampleWeights);
        }
    }
}
