// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree
{
    internal class InternalQuantileRegressionTree : InternalRegressionTree
    {
        /// <summary>
        /// Holds the labels of sampled instances for this tree. This value can be null when training, for example, random forest (FastForest).
        /// </summary>
        private double[] _labelsDistribution;

        /// <summary>
        /// Holds the weights of sampled instances for this tree. This value can be null when training, for example, random forest (FastForest).
        /// </summary>
        private double[] _instanceWeights;

        public bool IsWeightedTargets { get { return _instanceWeights != null; } }

        private const uint VerWithWeights = 0x00010002;

        public InternalQuantileRegressionTree(int maxLeaves)
            : base(maxLeaves)
        {
        }

        internal InternalQuantileRegressionTree(ModelLoadContext ctx, bool usingDefaultValue, bool categoricalSplits)
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
        public InternalQuantileRegressionTree(byte[] buffer, ref int position)
            : base(buffer, ref position)
        {
            _labelsDistribution = buffer.ToDoubleArray(ref position);
            _instanceWeights = buffer.ToDoubleArray(ref position);
        }

        internal override void Save(ModelSaveContext ctx)
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
        public void LoadSampledLabels(in VBuffer<float> feat, float[] distribution, float[] weights, int sampleCount, int destinationIndex)
        {
            int leaf = GetLeaf(in feat);
            LoadSampledLabels(distribution, weights, sampleCount, destinationIndex, leaf);
        }

        private void LoadSampledLabels(float[] distribution, float[] weights, int sampleCount, int destinationIndex, int leaf)
        {
            Contracts.Check(sampleCount == _labelsDistribution.Length / NumLeaves, "Bad quantile sample count");
            Contracts.Check(_instanceWeights == null || sampleCount == _instanceWeights.Length / NumLeaves, "Bad quantile weight count");

            if (weights != null)
            {
                for (int i = 0, j = sampleCount * leaf, k = destinationIndex; i < sampleCount; i++, j++, k++)
                {
                    distribution[k] = (float)_labelsDistribution[j];
                    weights[k] = (float)_instanceWeights[j];
                }
            }
            else
            {
                for (int i = 0, j = sampleCount * leaf, k = destinationIndex; i < sampleCount; i++, j++, k++)
                    distribution[k] = (float)_labelsDistribution[j];
            }
        }

        public void SetLabelsDistribution(double[] labelsDistribution, double[] weights)
        {
            _labelsDistribution = labelsDistribution;
            _instanceWeights = weights;
        }

        /// <summary>
        /// Copy training examples' labels and their weights to external variables.
        /// </summary>
        /// <param name="leafSamples">List of label collections. The type of a collection is a double array. The i-th label collection contains training examples' labels falling into the i-th leaf.</param>
        /// <param name="leafSampleWeights">List of labels' weight collections. The type of a collection is a double array. The i-th collection contains weights of labels falling into the i-th leaf.
        /// Specifically, leafSampleWeights[i][j] is the weight of leafSamples[i][j].</param>
        internal void ExtractLeafSamplesAndTheirWeights(out double[][] leafSamples, out double[][] leafSampleWeights)
        {
            leafSamples = new double[NumLeaves][];
            leafSampleWeights = new double[NumLeaves][];
            // If there is no training labels stored, we view the i-th leaf value as the only label stored at the i-th leaf.
            var sampleCountPerLeaf = _labelsDistribution != null ? _labelsDistribution.Length / NumLeaves : 1;
            for (int i = 0; i < NumLeaves; ++i)
            {
                leafSamples[i] = new double[sampleCountPerLeaf];
                leafSampleWeights[i] = new double[sampleCountPerLeaf];
                for (int j = 0; j < sampleCountPerLeaf; ++j)
                {
                    if (_labelsDistribution != null)
                        leafSamples[i][j] = _labelsDistribution[i * sampleCountPerLeaf + j];
                    else
                        // No training label is available, so the i-th leaf's value is used directly. Note that sampleCountPerLeaf must be 1 in this case.
                        leafSamples[i][j] = LeafValues[i];
                    if (_instanceWeights != null)
                        leafSampleWeights[i][j] = _instanceWeights[i * sampleCountPerLeaf + j];
                    else
                        leafSampleWeights[i][j] = 1.0;
                }
            }
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
