using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Trainers.FastTree.Internal;

namespace Microsoft.ML.FastTree.Representation
{
    /// <summary>
    /// A list of <see cref="RegressionTree"/>. To compute the output value of a <see cref="TreeEnsemble"/>, we need to compute
    /// the output values of all trees in <see cref="Trees"/>, scale those values via <see cref="TreeWeights"/>, and finally sum the scaled
    /// values and <see cref="Bias"/> up.
    /// </summary>
    public class TreeEnsemble
    {
        /// <summary>
        /// It's a best friend for being accessed from LightGBM.
        /// </summary>
        [BestFriend]
        internal readonly InternalTreeEnsemble UnderlyingTreeEnsemble;

        /// <summary>
        /// When doing prediction, this is a value added to the weighted sum of all <see cref="Trees"/>' outputs.
        /// </summary>
        public double Bias { get; }

        /// <summary>
        /// <see cref="TreeWeights"/>[i] is the i-th <see cref="RegressionTree"/>'s weight in this <see cref="TreeEnsemble"/>.
        /// </summary>
        public IReadOnlyList<double> TreeWeights { get; }

        /// <summary>
        /// <see cref="Trees"/>[i] is the i-th <see cref="RegressionTree"/> in this <see cref="TreeEnsemble"/>.
        /// </summary>
        public IReadOnlyList<RegressionTree> Trees { get; }

        internal TreeEnsemble(InternalTreeEnsemble treeEnsemble)
        {
            UnderlyingTreeEnsemble = treeEnsemble;
            Bias = treeEnsemble.Bias;
            TreeWeights = treeEnsemble.Trees.Select(tree => tree.Weight).ToList();
            Trees = treeEnsemble.Trees.Select(tree => new RegressionTree(tree)).ToList();
        }
    }

}
