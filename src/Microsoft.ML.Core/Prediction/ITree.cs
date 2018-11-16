// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.TreePredictor
{
    // The interfaces contained herein are meant to allow tree visualizer to run without an explicit dependency
    // on FastTree, so as to allow it greater generality. These should probably be moved somewhere else, but where?
    // FastTree itself is not a good candidate since their entire purpose was to avoid tying the tree visualizer
    // to FastTree itself. They are semi-tolerable though as a set of internal types here.

    /// <summary>
    /// Predictor that has ensemble tree structures and returns collection of trees.
    /// </summary>
    [BestFriend]
    internal interface ITreeEnsemble
    {
        /// <summary>
        /// Returns the number of trees in the ensemble.
        /// </summary>
        int NumTrees { get; }

        /// <summary>
        /// Returns the collection of trees.
        /// </summary>
        /// <returns>Collection of trees</returns>
        ITree[] GetTrees();
    }

    /// <summary>
    /// Type of tree used in ensemble of tree based predictors
    /// </summary>
    [BestFriend]
    internal interface ITree
    {
        /// <summary>
        /// Returns the array of right(Greater than) child nodes of every interior nodes
        /// </summary>
        int[] GtChild { get; }

        /// <summary>
        /// Returns the array of left(Leser than or equal to) nodes of every interior nodes
        /// </summary>
        int[] LteChild { get; }

        /// <summary>
        /// returns the number of interior nodes.
        /// </summary>
        int NumNodes { get; }

        /// <summary>
        /// Returns the number of leaf nodes.
        /// </summary>
        int NumLeaves { get; }

        /// <summary>
        /// Returns node structure for the given node
        /// </summary>
        /// <param name="nodeId">Node id</param>
        /// <param name="isLeaf">Flag to denote whether the node is leaf or not</param>
        /// <param name="featureNames">Feature names collection</param>
        /// <returns>Node structure</returns>
        INode GetNode(int nodeId, bool isLeaf, IEnumerable<string> featureNames);
    }

    /// <summary>
    /// Type of tree used in ensemble of tree based predictors
    /// </summary>
    /// <typeparam name="TFeatures">Type of features container (instance) on which to make predictions</typeparam>
    [BestFriend]
    internal interface ITree<TFeatures> : ITree
    {
        /// <summary>
        /// Returns the leaf node for the given instance.
        /// </summary>
        /// <param name="features">Type of features container (instance) on which to make predictions</param>
        /// <returns>node id</returns>
        /// <typeparamref name="TFeatures">Type of features container (instance) on which to make predictions</typeparamref>
        int GetLeaf(in TFeatures features);
    }

    /// <summary>
    /// Type to represent the structure of node
    /// </summary>
    [BestFriend]
    internal interface INode
    {
        /// <summary>
        /// Returns Key value pairs representing the properties of the node.
        /// </summary>
        Dictionary<string, object> KeyValues { get; }
    }

    /// <summary>
    /// Keys to represent the properties of node.
    /// </summary>
    [BestFriend]
    internal static class NodeKeys
    {
        /// <summary>
        /// Name of the the interior node. It is Feature name if it is fasttree. Type is string for default trees.
        /// </summary>
        public const string SplitName = "SplitName";

        /// <summary>
        /// Split gain of the interior node. Type is double for default trees.
        /// </summary>
        public const string SplitGain = "SplitGain";

        /// <summary>
        /// Threshold value of the interior node. Type is string for default trees.
        /// It is expected that the string has exactly two space separated components.
        /// i. The first one should be the operator
        /// ii. The second one should be the actual threshold
        /// For ex., for a split like f1 &lt;= 10, expected Threshold is "&lt;= 10"
        /// For a split like color not-in { blue, green }, expected Threshold is "not-in { blue, green }"
        /// </summary>
        public const string Threshold = "Threshold";

        /// <summary>
        /// Gain value (specific to fasttree) of the interior node. Type is double for default trees.
        /// </summary>
        public const string GainValue = "GainValue";

        /// <summary>
        /// Previous leaf value(specific to fasttree) of the interior node. Type is double for default trees.
        /// </summary>
        public const string PreviousLeafValue = "PreviousLeafValue";

        /// <summary>
        /// Leaf value of the leaf node. Type is double for default trees.
        /// </summary>
        public const string LeafValue = "LeafValue";

        /// <summary>
        /// Extra items that will be displayed in the tool-tip. Type is string for default trees.
        /// </summary>
        public const string Extras = "Extras";
    }
}
