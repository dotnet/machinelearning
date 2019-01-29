using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Trainers.FastTree.Internal;

namespace Microsoft.ML.FastTree.Representation
{
    /// <summary>
    /// A container class for exposing <see cref="RegressionTree"/>'s attributes to users.
    /// </summary>
    public class TreeRegressor
    {
        /// <summary>
        /// <see cref="TreeRegressor"/> is an immutable wrapper over <see cref="_tree"/> for exposing some tree's attribute to users.
        /// </summary>
        private readonly RegressionTree _tree;

        /// <summary>
        /// Sample labels from training data. <see cref="_leafSamples"/>[i] stores the labels falling into the i-th leaf.
        /// </summary>
        private readonly List<double[]> _leafSamples;
        /// <summary>
        /// Sample labels' weights from training data. <see cref="_leafSampleWeights"/>[i] stores the weights for labels falling into the i-th leaf.
        /// <see cref="_leafSampleWeights"/>[i][j] is the weight of <see cref="_leafSamples"/>[i][j].
        /// </summary>
        private readonly List<double[]> _leafSampleWeights;

        // Immutable accessors to the underlying tree structure. They help users to inspect tree learned using, for example,
        // gradient-boosting decision tree and random forest.
        public ReadOnlySpan<int> LteChild => new ReadOnlySpan<int>(_tree.LteChild, 0, _tree.NumNodes);
        public ReadOnlySpan<int> GtChild => new ReadOnlySpan<int>(_tree.GtChild, 0, _tree.NumNodes);
        public ReadOnlySpan<int> NumericalSplitFeatureIndexes => new ReadOnlySpan<int>(_tree.SplitFeatures, 0, _tree.NumNodes);
        public ReadOnlySpan<float> NumericalSplitThresholds => new ReadOnlySpan<float>(_tree.RawThresholds, 0, _tree.NumNodes);
        public ReadOnlySpan<bool> CategoricalSplitFlags => new ReadOnlySpan<bool>(_tree.CategoricalSplit, 0, _tree.NumNodes);
        public ReadOnlySpan<double> LeafValues => new ReadOnlySpan<double>(_tree.LeafValues, 0, _tree.NumLeaves);
        public ReadOnlySpan<int> GetCategoricalSplitFeaturesAt(int nodeIndex) => new ReadOnlySpan<int>(_tree.CategoricalSplitFeatures[nodeIndex]);
        public ReadOnlySpan<int> GetCategoricalCategoricalSplitFeatureRangeAt(int nodeIndex) => new ReadOnlySpan<int>(_tree.CategoricalSplitFeatureRanges[nodeIndex]);

        /// <summary>
        /// Return the training labels falling into the specified leaf.
        /// </summary>
        /// <param name="leafIndex">The index of the specified leaf.</param>
        /// <returns>Training labels</returns>
        public ReadOnlySpan<double> GetLeafSamplesAt(int leafIndex) => new ReadOnlySpan<double>(_leafSamples[leafIndex]);

        /// <summary>
        /// Return the weights for training labels falling into the specified leaf. If <see cref="GetLeafSamplesAt"/> and <see cref="GetLeafSampleWeightsAt"/>
        /// use the same input, the i-th returned value of this function is the weight of the i-th label in <see cref="GetLeafSamplesAt"/>.
        /// </summary>
        /// <param name="leafIndex">The index of the specified leaf.</param>
        /// <returns>Training labels' weights</returns>
        public ReadOnlySpan<double> GetLeafSampleWeightsAt(int leafIndex) => new ReadOnlySpan<double>(_leafSampleWeights[leafIndex]);

        /// <summary>
        /// Number of leaves in the tree. Note that <see cref="NumLeaves"/> does not take non-leaf nodes into account.
        /// </summary>
        public int NumLeaves => _tree.NumLeaves;

        /// <summary>
        /// Number of nodes in the tree. This doesn't include any leaves. For example, a tree with node0->node1, node0->leaf3,
        /// node1->leaf1, node1->leaf2, <see cref="NumNodes"/> and <see cref="NumLeaves"/> should be 2 and 3, respectively.
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
                _leafSamples = tree.LeafValues.Select(value => new double[] { value }).ToList();
                _leafSampleWeights = tree.LeafValues.Select(value => new double[] { 1.0 }).ToList();
            }
        }
    }

}
