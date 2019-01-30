using System;
using System.Linq;
using Microsoft.ML.Trainers.FastTree.Internal;

namespace Microsoft.ML.FastTree.Representation
{
    /// <summary>
    /// A container class for exposing <see cref="RegressionTree"/>'s attributes to users.
    /// This class should not be mutable, so it contains a lot of read-only members.
    /// </summary>
    public class TreeRegressor
    {
        /// <summary>
        /// <see cref="TreeRegressor"/> is an immutable wrapper over <see cref="_tree"/> for exposing some tree's
        /// attribute to users.
        /// </summary>
        private readonly RegressionTree _tree;

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
        /// <see cref="LteChild"/>[i] is the i-th node's child index used when
        /// (1) the numerical feature indexed by <see cref="NumericalSplitFeatureIndexes"/>[i] is less than the
        /// threshold <see cref="NumericalSplitThresholds"/>[i], or
        /// (2) the categorical features indexed by <see cref="GetCategoricalCategoricalSplitFeatureRangeAt(int)"/>'s
        /// returned value with nodeIndex=i is NOT a sub-set of <see cref="GetCategoricalSplitFeaturesAt(int)"/> with
        /// nodeIndex=i.
        /// Note that the case (1) happens only when <see cref="CategoricalSplitFlags"/>[i] is true and otherwise (2)
        /// occurs. A non-negative returned value means a node (i.e., not a leaf); for example, 2 means the 3rd node in
        /// the underlying <see cref="_tree"/>. A negative returned value means a leaf; for example, -1 stands for the
        /// first leaf in the underlying <see cref="_tree"/>.
        /// </summary>
        public ReadOnlySpan<int> LteChild => new ReadOnlySpan<int>(_tree.LteChild, 0, _tree.NumNodes);
        /// <summary>
        /// <see cref="GtChild"/>[i] is the i-th node's child index used when the two conditions, (1) and (2),
        /// described in <see cref="LteChild"/>'s document are not true. Its return value follows the format
        /// used in <see cref="LteChild"/>.
        /// </summary>
        public ReadOnlySpan<int> GtChild => new ReadOnlySpan<int>(_tree.GtChild, 0, _tree.NumNodes);
        /// <summary>
        /// <see cref="NumericalSplitFeatureIndexes"/>[i] is the feature index used the splitting function of the
        /// i-th node. This value is valid only if <see cref="CategoricalSplitFlags"/>[i] is false.
        /// </summary>
        public ReadOnlySpan<int> NumericalSplitFeatureIndexes => new ReadOnlySpan<int>(_tree.SplitFeatures, 0, _tree.NumNodes);
        /// <summary>
        /// <see cref="NumericalSplitThresholds"/>[i] is the threshold on feature indexed by
        /// <see cref="NumericalSplitFeatureIndexes"/>[i], where i is the i-th node's index
        /// (for example, i is 1 for the 2nd node in <see cref="_tree"/>).
        /// </summary>
        public ReadOnlySpan<float> NumericalSplitThresholds => new ReadOnlySpan<float>(_tree.RawThresholds, 0, _tree.NumNodes);
        /// <summary>
        /// Determine the types of splitting function. If <see cref="CategoricalSplitFlags"/>[i] is true, the i-th
        /// node's uses categorical splitting function. Otherwise, traditional numerical split is used.
        /// </summary>
        public ReadOnlySpan<bool> CategoricalSplitFlags => new ReadOnlySpan<bool>(_tree.CategoricalSplit, 0, _tree.NumNodes);
        /// <summary>
        /// <see cref="LeafValues"/>[i] is the learned value at the i-th leaf.
        /// </summary>
        public ReadOnlySpan<double> LeafValues => new ReadOnlySpan<double>(_tree.LeafValues, 0, _tree.NumLeaves);
        /// <summary>
        /// Return categorical thresholds used at node indexed by nodeIndex. If the considered input feature does NOT
        /// matche any of values returned by <see cref="GetCategoricalSplitFeaturesAt(int)"/>, we call it a
        /// less-than-threshold event and therefore <see cref="LteChild"/>[nodeIndex] is the child node that input
        /// should go next. The returned value is valid only if <see cref="CategoricalSplitFlags"/>[nodeIndex] is true.
        /// </summary>
        public ReadOnlySpan<int> GetCategoricalSplitFeaturesAt(int nodeIndex)
        {
            if (nodeIndex < 0 || nodeIndex >= NumNodes)
                throw Contracts.Except($"The input index, {nodeIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumNodes} (exclusive).");

            if (_tree.CategoricalSplitFeatures == null || _tree.CategoricalSplitFeatures[nodeIndex] == null)
                return new ReadOnlySpan<int>(); // Zero-length vector.
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
        public ReadOnlySpan<int> GetCategoricalCategoricalSplitFeatureRangeAt(int nodeIndex)
        {
            if (nodeIndex < 0 || nodeIndex >= NumNodes)
                throw Contracts.Except($"The input node index, {nodeIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumNodes} (exclusive).");

            if (_tree.CategoricalSplitFeatureRanges == null || _tree.CategoricalSplitFeatureRanges[nodeIndex] == null)
                return new ReadOnlySpan<int>(); // Zero-length vector.
            else
                return new ReadOnlySpan<int>(_tree.CategoricalSplitFeatureRanges[nodeIndex]);
        }

        /// <summary>
        /// Return the training labels falling into the specified leaf.
        /// </summary>
        /// <param name="leafIndex">The index of the specified leaf.</param>
        /// <returns>Training labels</returns>
        public ReadOnlySpan<double> GetLeafSamplesAt(int leafIndex)
        {
            if (leafIndex < 0 || leafIndex >= NumLeaves)
                throw Contracts.Except($"The input leaf index, {leafIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumLeaves} (exclusive).");

            // _leafSample always contains valid values assigned in constructor.
            return new ReadOnlySpan<double>(_leafSamples[leafIndex]);
        }

        /// <summary>
        /// Return the weights for training labels falling into the specified leaf. If <see cref="GetLeafSamplesAt"/>
        /// and <see cref="GetLeafSampleWeightsAt"/> use the same input, the i-th returned value of this function is
        /// the weight of the i-th label in <see cref="GetLeafSamplesAt"/>.
        /// </summary>
        /// <param name="leafIndex">The index of the specified leaf.</param>
        /// <returns>Training labels' weights</returns>
        public ReadOnlySpan<double> GetLeafSampleWeightsAt(int leafIndex)
        {
            if (leafIndex < 0 || leafIndex >= NumLeaves)
                throw Contracts.Except($"The input leaf index, {leafIndex}, is invalid. Its valid range is from 0 (inclusive) to {NumLeaves} (exclusive).");

            // _leafSampleWeights always contains valid values assigned in constructor.
            return new ReadOnlySpan<double>(_leafSampleWeights[leafIndex]);
        }

        /// <summary>
        /// Number of leaves in the tree. Note that <see cref="NumLeaves"/> does not take non-leaf nodes into account.
        /// </summary>
        public int NumLeaves => _tree.NumLeaves;

        /// <summary>
        /// Number of nodes in the tree. This doesn't include any leaves. For example, a tree with node0->node1,
        /// node0->leaf3, node1->leaf1, node1->leaf2, <see cref="NumNodes"/> and <see cref="NumLeaves"/> should
        /// be 2 and 3, respectively.
        /// </summary>
        // A visualization of the example mentioned in this doc string.
        //         node0
        //         /  \
        //     node1 leaf3
        //     /  \
        // leaf1 leaf2
        // The index of leaf starts with 1 because interally we use "-1" as the 1st leaf's index, "-2" for the 2nd leaf's index, and so on.
        public int NumNodes => _tree.NumNodes;

        internal TreeRegressor(RegressionTree tree)
        {
            _tree = tree;
            _leafSamples = null;
            _leafSampleWeights = null;

            if (tree is QuantileRegressionTree)
                ((QuantileRegressionTree)tree).ExtractLeafSamplesAndTheirWeights(out _leafSamples, out _leafSampleWeights);
            else
            {
                _leafSamples = tree.LeafValues.Select(value => new double[] { value }).ToArray();
                _leafSampleWeights = tree.LeafValues.Select(value => new double[] { 1.0 }).ToArray();
            }
        }
    }

}
