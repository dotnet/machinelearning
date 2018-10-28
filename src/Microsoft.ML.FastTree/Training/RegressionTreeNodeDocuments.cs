// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    // RegressionTreeNodeDocuments represents an association between a node in a regression
    // tree and documents belonging to that node.
    // In other word it is <RegressionTree,DocumentPartitioning, NodeIndex> tuple along with operations
    // defined for the tuple
    // It also hides the fact that node can be either leaf on an interior node behind the hood.
    // The caller can treat interior node (entire subtree) in a same fashion as a leaf node.
    public class RegressionTreeNodeDocuments
    {
        public readonly RegressionTree Tree;
        public readonly DocumentPartitioning Partitioning;
        public readonly int NodeIndex; //Index to a node or leaf within the tree
        private int _documentCount;
        public bool IsLeaf => NodeIndex < 0;

        public RegressionTreeNodeDocuments(RegressionTree tree, DocumentPartitioning partitioning, int nodeIndex)
        {
            Tree = tree;
            Partitioning = partitioning;
            NodeIndex = nodeIndex;
            _documentCount = -1;
        }
        public int GetDocumentCount()
        {
            if (_documentCount == -1)
                _documentCount = Tree.GetNodesLeaves(NodeIndex).Sum(leaf => Partitioning.NumDocsInLeaf(leaf));
            return _documentCount;
        }

        // Adds delta to a node output (if node is a subtree all leafs are modified)
        public void UpdateOutputsWithDelta(double delta)
        {
            if (IsLeaf)
                Tree.UpdateOutputWithDelta(~NodeIndex, delta);
            else
                foreach (var node in GetLeaves())
                {
                    node.UpdateOutputsWithDelta(delta);
                }
        }

        // Returns collection of leaves of a interior node (or itslef if it is a leaf)
        public IEnumerable<RegressionTreeNodeDocuments> GetLeaves()
        {
            if (IsLeaf)
                return Enumerable.Repeat(this, 1);
            else
                return Tree.GetNodesLeaves(NodeIndex).Select(l => new RegressionTreeNodeDocuments(Tree, Partitioning, ~l));
        }

        // Returns a Hashset of leaf indexes of this node
        public HashSet<int> GetDocuments()
        {
            HashSet<int> documentsInNode;
            if (NodeIndex < 0) //It is a leaf
            {
                documentsInNode = new HashSet<int>(Partitioning.DocumentsInLeaf(~NodeIndex));
            }
            else
            {
                documentsInNode = new HashSet<int>();
                foreach (var leaf in Tree.GetNodesLeaves(NodeIndex))
                {
                    var allDocsInLeaf = Partitioning.DocumentsInLeaf(leaf);
                    documentsInNode.UnionWith(allDocsInLeaf);
                }

            }
            return documentsInNode;
        }
    }

    //RecursiveRegressionTree captures a recursive representation of a tree
    //and inherits from RegressionTreeNodeDocuments (a non-recursive node with documents)
    //The class in most cases would be contructed with node index of 0 and would create
    //entire structure of a full tree accessible with LTENode and GTNode
    //
    //Curently only used for smoothing and defines operations defined in recursive fashion
    //GetWeightedOutput and SmoothLeafOutputs used for smoothing the trees
    public class RecursiveRegressionTree : RegressionTreeNodeDocuments
    {
        //Left and right children on a regression tree
        public readonly RecursiveRegressionTree LteNode;
        public readonly RecursiveRegressionTree GtNode;
        private double _weightedOutput;
        private int _nodeCount;

        public RecursiveRegressionTree(RegressionTree t, DocumentPartitioning p, int n)
            : base(t, p, n)
        {
            _weightedOutput = double.NaN;
            _nodeCount = int.MaxValue;
            if (!IsLeaf)
            {
                LteNode = new RecursiveRegressionTree(Tree, Partitioning, Tree.GetLteChildForNode(NodeIndex));
                GtNode = new RecursiveRegressionTree(Tree, Partitioning, Tree.GetGtChildForNode(NodeIndex));
            }
        }

        // Smoothing of leafs outputs:
        // 0 - no smoothing
        // 1 - maximal smoothing
        public void SmoothLeafOutputs(double parentOutput, double smoothing)
        {
            double myOutput = (1 - smoothing) * GetWeightedOutput() + smoothing * parentOutput;
            if (IsLeaf)
            {
                Tree.SetOutput(~NodeIndex, myOutput);
            }
            else
            {
                LteNode.SmoothLeafOutputs(myOutput, smoothing);
                GtNode.SmoothLeafOutputs(myOutput, smoothing);
            }
        }

        // Implementation for cached computation of weighted output
        // (used by smoothing)
        public double GetWeightedOutput()
        {
            if (!double.IsNaN(_weightedOutput))
                return _weightedOutput;
            if (NodeIndex < 0)
                return Tree.GetOutput(~NodeIndex);

            int lteDocCount = LteNode.GetDocumentCount();
            int gtCount = GtNode.GetDocumentCount();

            _weightedOutput = (lteDocCount * LteNode.GetWeightedOutput() + gtCount * GtNode.GetWeightedOutput())
                 / (lteDocCount + gtCount);
            return _weightedOutput;
        }

        // Smoothing of leafs outputs:
        // 0 - no smoothing
        // 1 - maximal smoothing
        public void SmoothLeafOutputs(double parentOutput, double smoothing, int[] documentCount)
        {
            int nodeCount = int.MaxValue;
            double myOutput = (1 - smoothing) * GetWeightedOutput(documentCount, out nodeCount) + smoothing * parentOutput;
            if (IsLeaf)
            {
                Tree.SetOutput(~NodeIndex, myOutput);
            }
            else
            {
                LteNode.SmoothLeafOutputs(myOutput, smoothing);
                GtNode.SmoothLeafOutputs(myOutput, smoothing);
            }
        }

        // set document count outside instead of calculated from local partition
        public double GetWeightedOutput(int[] documentCount, out int nodeCount)
        {
            if (!double.IsNaN(_weightedOutput) && _nodeCount != int.MaxValue)
            {
                nodeCount = _nodeCount;
                return _weightedOutput;
            }

            if (NodeIndex < 0)
            {
                nodeCount = documentCount[~NodeIndex];
                return Tree.GetOutput(~NodeIndex);
            }

            int lteDocCount = int.MaxValue;
            double lteWeight = LteNode.GetWeightedOutput(documentCount, out lteDocCount);
            int gtDocCount = int.MaxValue;
            double gtWeight = GtNode.GetWeightedOutput(documentCount, out gtDocCount);

            _weightedOutput = (lteDocCount * lteWeight + gtDocCount * gtWeight) / (lteDocCount + gtDocCount);
            _nodeCount = lteDocCount + gtDocCount;

            nodeCount = _nodeCount;
            return _weightedOutput;
        }

    }

}
