// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using System;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public abstract class TreeLearner
    {
        public readonly Dataset TrainData;
        public readonly int NumLeaves;
        // REVIEW: Needs to be assignable due to the way bagging is implemented. :P Imagine something less stupid and fragile.
        public DocumentPartitioning Partitioning;

        protected TreeLearner(Dataset trainData, int numLeaves)
        {
            TrainData = trainData;
            NumLeaves = numLeaves;
            Partitioning = new DocumentPartitioning(TrainData.NumDocs, numLeaves);
        }

        public static string TargetWeightsDatasetName { get { return "TargetWeightsDataset"; } }

        public abstract RegressionTree FitTargets(IChannel ch, bool[] activeFeatures, double[] targets);

        /// <summary>
        /// Get size of reserved memory for the tree learner.
        /// The default implementation returns 0 directly, and the subclasses can return
        /// different value if it reserves memory for training.
        /// </summary>
        /// <returns>size of reserved memory</returns>
        public virtual long GetSizeOfReservedMemory()
        {
            return 0L;
        }
    }

    /// <summary>
    /// An exception class for an error which occurs in the midst of learning a tree.
    /// </summary>
    public class TreeLearnerException : Exception
    {
        public TreeLearnerException(string message) : base(message) { }
    }
}
