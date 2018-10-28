// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Float = System.Single;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public class QuantileRegressionTree : RegressionTree
    {
        // Holds the labels of samped instances for this tree
        private double[] _labelsDistribution;

        // Holds the weights of sampled instances for this tree
        private double[] _instanceWeights;

        public bool IsWeightedTargets { get { return _instanceWeights != null; } }

        private const uint VerWithWeights = 0x00010002;

        public QuantileRegressionTree(int maxLeaves)
            : base(maxLeaves)
        {
        }

        internal QuantileRegressionTree(ModelLoadContext ctx, bool usingDefaultValue, bool categoricalSplits)
            : base(ctx, usingDefaultValue, categoricalSplits)
        {
            // *** Binary format ***
            // double[]: Labels Distribution.
            // double[]: Weights for the Distribution.
            _labelsDistribution = ctx.Reader.ReadDoubleArray();

            if (ctx.Header.ModelVerWritten >= VerWithWeights)
                _instanceWeights = ctx.Reader.ReadDoubleArray();
        }

        // REVIEW: Do we need this method? I am seeing in many places in tree code
        public QuantileRegressionTree(byte[] buffer, ref int position)
            : base(buffer, ref position)
        {
            _labelsDistribution = buffer.ToDoubleArray(ref position);
            _instanceWeights = buffer.ToDoubleArray(ref position);
        }

        public override void Save(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // double[]: Labels Distribution.
            base.Save(ctx, TreeType.FastForest);
            ctx.Writer.WriteDoubleArray(_labelsDistribution);
            ctx.Writer.WriteDoubleArray(_instanceWeights);
        }

        /// <summary>
        /// Loads the sampled labels of this tree to the distribution array for the sparse instance type.
        /// By calling for all the trees, the distribution array will have all the samples from all the trees
        /// </summary>
        public void LoadSampledLabels(ref VBuffer<Float> feat, Float[] distribution, Float[] weights, int sampleCount, int destinationIndex)
        {
            int leaf = GetLeaf(ref feat);
            LoadSampledLabels(distribution, weights, sampleCount, destinationIndex, leaf);
        }

        private void LoadSampledLabels(Float[] distribution, Float[] weights, int sampleCount, int destinationIndex, int leaf)
        {
            Contracts.Check(sampleCount == _labelsDistribution.Length / NumLeaves, "Bad quantile sample count");
            Contracts.Check(_instanceWeights == null || sampleCount == _instanceWeights.Length / NumLeaves, "Bad quantile weight count");

            if (weights != null)
            {
                for (int i = 0, j = sampleCount * leaf, k = destinationIndex; i < sampleCount; i++, j++, k++)
                {
                    distribution[k] = (Float)_labelsDistribution[j];
                    weights[k] = (Float)_instanceWeights[j];
                }
            }
            else
            {
                for (int i = 0, j = sampleCount * leaf, k = destinationIndex; i < sampleCount; i++, j++, k++)
                    distribution[k] = (Float)_labelsDistribution[j];
            }
        }

        public void SetLabelsDistribution(double[] labelsDistribution, double[] weights)
        {
            _labelsDistribution = labelsDistribution;
            _instanceWeights = weights;
        }

        public override int SizeInBytes()
        {
            return base.SizeInBytes() + _labelsDistribution.SizeInBytes() + (_instanceWeights != null ? _instanceWeights.SizeInBytes() : 0);
        }

        public override void ToByteArray(byte[] buffer, ref int position)
        {
            base.ToByteArray(buffer, ref position);
            _labelsDistribution.ToByteArray(buffer, ref position);
            if (_instanceWeights != null)
                _instanceWeights.ToByteArray(buffer, ref position);
        }
    }
}
