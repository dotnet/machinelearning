// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Float = System.Single;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public class RegressionTree
    {
        private double _maxOutput;

        // Weight of this tree in the ensemble

        // for each non-leaf, we keep the following data
        public Float[] DefaultValueForMissing;
        private double[] _splitGain;
        private double[] _gainPValue;
        // The value of this non-leaf node, prior to split when it was a leaf.
        private double[] _previousLeafValue;

        public bool[] ActiveFeatures { get; set; }

        public int[] LteChild { get; }
        public int[] GtChild { get; }
        public int[] SplitFeatures { get; }
        /// <summary>
        /// Indicates if a node's split feature was categorical.
        /// </summary>
        public bool[] CategoricalSplit { get; }
        /// <summary>
        /// Array of categorical values for the categorical feature that might be chosen as
        /// a split feature for a node.
        /// </summary>
        public int[][] CategoricalSplitFeatures;
        /// <summary>
        /// For a given categorical feature that is chosen as a split feature for a node, this
        /// array contains its start and end range in the input feature vector at prediction time.
        /// </summary>
        public int[][] CategoricalSplitFeatureRanges;
        // These are the thresholds based on the binned values of the raw features.
        public UInt32[] Thresholds { get; }
        // These are the thresholds based on the raw feature values. Populated after training.
        public Float[] RawThresholds { get; private set; }
        public double[] SplitGains { get { return _splitGain; } }
        public double[] GainPValues { get { return _gainPValue; } }
        public double[] PreviousLeafValues { get { return _previousLeafValue; } }
        public double[] LeafValues { get; }

        /// <summary>
        /// Code to identify the type of tree in binary serialization. These values are
        /// persisted, so they should remain consistent for the sake of deserialization
        /// backwards compatibility.
        /// </summary>
        protected enum TreeType : byte
        {
            Regression = 0,
            Affine = 1,
            FastForest = 2
        }

        private RegressionTree()
        {
            Weight = 1.0;
        }

        /// <summary>
        /// constructs a regression tree with an upper bound on depth
        /// </summary>
        public RegressionTree(int maxLeaves)
            : this()
        {
            SplitFeatures = new int[maxLeaves - 1];
            CategoricalSplit = new bool[maxLeaves - 1];
            _splitGain = new double[maxLeaves - 1];
            _gainPValue = new double[maxLeaves - 1];
            _previousLeafValue = new double[maxLeaves - 1];
            Thresholds = new UInt32[maxLeaves - 1];
            DefaultValueForMissing = null;
            LteChild = new int[maxLeaves - 1];
            GtChild = new int[maxLeaves - 1];
            LeafValues = new double[maxLeaves];
            NumLeaves = 1;
        }

        public RegressionTree(byte[] buffer, ref int position)
            : this()
        {
            NumLeaves = buffer.ToInt(ref position);
            _maxOutput = buffer.ToDouble(ref position);
            Weight = buffer.ToDouble(ref position);
            LteChild = buffer.ToIntArray(ref position);
            GtChild = buffer.ToIntArray(ref position);
            SplitFeatures = buffer.ToIntArray(ref position);
            byte[] categoricalSplitAsBytes = buffer.ToByteArray(ref position);
            CategoricalSplit = categoricalSplitAsBytes.Select(b => b > 0).ToArray();
            if (CategoricalSplit.Any(b => b))
            {
                CategoricalSplitFeatures = new int[NumNodes][];
                CategoricalSplitFeatureRanges = new int[NumNodes][];
                for (int index = 0; index < NumNodes; index++)
                {
                    CategoricalSplitFeatures[index] = buffer.ToIntArray(ref position);
                    CategoricalSplitFeatureRanges[index] = buffer.ToIntArray(ref position);
                }
            }

            Thresholds = buffer.ToUIntArray(ref position);
            RawThresholds = buffer.ToFloatArray(ref position);
            _splitGain = buffer.ToDoubleArray(ref position);
            _gainPValue = buffer.ToDoubleArray(ref position);
            _previousLeafValue = buffer.ToDoubleArray(ref position);
            LeafValues = buffer.ToDoubleArray(ref position);
        }

        private bool[] GetCategoricalSplitFromIndices(int[] indices)
        {
            bool[] categoricalSplit = new bool[NumNodes];
            if (indices == null)
                return categoricalSplit;

            Contracts.Assert(indices.Length <= NumNodes);

            foreach (int index in indices)
            {
                Contracts.Assert(index >= 0 && index < NumNodes);
                categoricalSplit[index] = true;
            }

            return categoricalSplit;
        }

        private bool[] GetCategoricalSplitFromBytes(byte[] indices)
        {
            bool[] categoricalSplit = new bool[NumNodes];
            if (indices == null)
                return categoricalSplit;

            Contracts.Assert(indices.Length <= NumNodes);

            foreach (int index in indices)
            {
                Contracts.Assert(index >= 0 && index < NumNodes);
                categoricalSplit[index] = true;
            }

            return categoricalSplit;
        }

        /// <summary>
        /// Create a Regression Tree object from raw tree contents.
        /// </summary>
        public static RegressionTree Create(int numLeaves, int[] splitFeatures, Double[] splitGain,
            Float[] rawThresholds, Float[] defaultValueForMissing, int[] lteChild, int[] gtChild, Double[] leafValues,
            int[][] categoricalSplitFeatures, bool[] categoricalSplit)
        {
            if (numLeaves <= 1)
            {
                // Create a dummy tree.
                RegressionTree tree = new RegressionTree(2);
                tree.SetOutput(0, 0.0);
                tree.SetOutput(1, 0.0);
                return tree;
            }
            else
            {
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(splitFeatures), nameof(splitFeatures), "Size error, should equal to numLeaves - 1.");
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(splitGain), nameof(splitGain), "Size error, should equal to numLeaves - 1.");
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(rawThresholds), nameof(rawThresholds), "Size error, should equal to numLeaves - 1.");
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(lteChild), nameof(lteChild), "Size error, should equal to numLeaves - 1.");
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(gtChild), nameof(gtChild), "Size error, should equal to numLeaves - 1.");
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(defaultValueForMissing), nameof(defaultValueForMissing), "Size error, should equal to numLeaves - 1.");
                Contracts.CheckParam(numLeaves == Utils.Size(leafValues), nameof(leafValues), "Size error, should equal to numLeaves.");
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(categoricalSplitFeatures), nameof(categoricalSplitFeatures), "Size error, should equal to numLeaves - 1.");
                Contracts.CheckParam(numLeaves - 1 == Utils.Size(categoricalSplit), nameof(categoricalSplit), "Size error, should equal to numLeaves - 1.");
                return new RegressionTree(splitFeatures, splitGain, null, rawThresholds, defaultValueForMissing, lteChild, gtChild, leafValues, categoricalSplitFeatures, categoricalSplit);
            }
        }

        internal RegressionTree(int[] splitFeatures, Double[] splitGain, Double[] gainPValue,
            Float[] rawThresholds, Float[] defaultValueForMissing, int[] lteChild, int[] gtChild, Double[] leafValues,
            int[][] categoricalSplitFeatures, bool[] categoricalSplit)
            : this()
        {
            Contracts.CheckParam(Utils.Size(splitFeatures) > 0, nameof(splitFeatures), "Number of split features must be positive");

            NumLeaves = Utils.Size(splitFeatures) + 1;
            SplitFeatures = splitFeatures;
            _splitGain = splitGain;
            _gainPValue = gainPValue;
            RawThresholds = rawThresholds;
            DefaultValueForMissing = defaultValueForMissing;
            LteChild = lteChild;
            GtChild = gtChild;
            LeafValues = leafValues;
            CategoricalSplitFeatures = categoricalSplitFeatures;
            CategoricalSplitFeatureRanges = new int[CategoricalSplitFeatures.Length][];
            for (int i = 0; i < CategoricalSplitFeatures.Length; ++i)
            {
                if (CategoricalSplitFeatures[i] != null && CategoricalSplitFeatures[i].Length > 0)
                {
                    CategoricalSplitFeatureRanges[i] = new int[2];
                    CategoricalSplitFeatureRanges[i][0] = CategoricalSplitFeatures[i].First();
                    CategoricalSplitFeatureRanges[i][1] = CategoricalSplitFeatures[i].Last();
                    Contracts.Assert(categoricalSplit[i]);
                }
            }
            CategoricalSplit = categoricalSplit;

            CheckValid(Contracts.Check);

            if (DefaultValueForMissing != null)
            {
                bool allZero = true;
                foreach (var val in DefaultValueForMissing)
                {
                    if (val != 0.0f)
                    {
                        allZero = false;
                        break;
                    }
                }
                if (allZero)
                    DefaultValueForMissing = null;
            }
        }

        internal RegressionTree(ModelLoadContext ctx, bool usingDefaultValue, bool categoricalSplits)
            : this()
        {
            // *** Binary format ***
            // Four convenient quantities to keep in mind:
            // -- L and N (number of leaves and nodes, L = N+1)
            // -- ML and MN (maximum number of leaves and nodes this can support, ML = MN+1)
            // -- C (Number of nodes that have categorical split feature)
            // -- CT (Number of categorical feature values for a chosen split feature)
            // Some arrays can be null if they do not effect prediction, or redundant.
            // All arrays, despite having prescribed sizes, are prefixed with size
            //
            // byte: tree type code, 0 if regression, 1 if affine (unsupported), 2 if fast forest
            // int: number of leaves currently in the tree, L
            // double: maxoutput
            // double: weight
            // int[MN]: lte child, MN is inferred from the length of this array
            // int[MN]: gt child
            // int[MN]: split feature index
            // int[C]: categorical node indices.
            // int[C*(CT + 2)]: categorical feature values and categorical feature range in the input feature vector.
            // int[MN]: threshold bin index (can be null of raw thresholds are not null)
            // Float[MN]: raw threshold (can be not null if threshold bin indices are not null)
            // Float[MN]: default value For missing
            // double[ML]: leaf value
            // double[MN]: gain of this split (can be null)
            // double[MN]: p-value of this split (can be null)
            // double[MN]: previous value of a node before it made the transition from leaf to node (can be null)

            BinaryReader reader = ctx.Reader;
            NumLeaves = reader.ReadInt32();
            _maxOutput = reader.ReadDouble();
            Weight = reader.ReadDouble();
            // Tree structure...
            LteChild = reader.ReadIntArray();
            GtChild = reader.ReadIntArray();
            SplitFeatures = reader.ReadIntArray();

            if (categoricalSplits)
            {
                int[] categoricalNodeIndices = reader.ReadIntArray();
                CategoricalSplit = GetCategoricalSplitFromIndices(categoricalNodeIndices);
                if (categoricalNodeIndices?.Length > 0)
                {
                    CategoricalSplitFeatures = new int[NumNodes][];
                    CategoricalSplitFeatureRanges = new int[NumNodes][];
                    foreach (var index in categoricalNodeIndices)
                    {
                        Contracts.Assert(CategoricalSplit[index]);
                        Contracts.Assert(index >= 0 && index < NumNodes);

                        CategoricalSplitFeatures[index] = reader.ReadIntArray();
                        CategoricalSplitFeatureRanges[index] = reader.ReadIntArray(2);
                    }
                }
            }
            else
                CategoricalSplit = new bool[NumNodes];

            Thresholds = reader.ReadUIntArray();
            RawThresholds = reader.ReadFloatArray();

            DefaultValueForMissing = null;
            if (usingDefaultValue)
                DefaultValueForMissing = reader.ReadFloatArray();

            LeafValues = reader.ReadDoubleArray();
            // Informational...
            _splitGain = reader.ReadDoubleArray();
            _gainPValue = reader.ReadDoubleArray();
            _previousLeafValue = reader.ReadDoubleArray();

            CheckValid(Contracts.CheckDecode);

            // Check the need of _defaultValueForMissing
            if (DefaultValueForMissing != null)
            {
                bool allZero = true;
                foreach (var val in DefaultValueForMissing)
                {
                    if (val != 0.0f)
                    {
                        allZero = false;
                        break;
                    }
                }
                if (allZero)
                    DefaultValueForMissing = null;
            }
        }

        protected void Save(ModelSaveContext ctx, TreeType code)
        {
#if DEBUG
            // This must be compiled only in the debug case, since you can't
            // have delegates on functions with conditional attributes.
            CheckValid((t, s) => Contracts.Assert(t, s));
#endif

            // *** Binary format ***
            // Four convenient quantities to keep in mind:
            // -- L and N (number of leaves and nodes, L = N+1)
            // -- ML and MN (maximum number of leaves and nodes this can support, ML = MN+1)
            // -- C (Number of nodes that have categorical split feature)
            // -- CT (Number of categorical feature values for a chosen split feature)
            // Some arrays can be null if they do not effect prediction, or redundant.
            // All arrays, despite having prescribed sizes, are prefixed with size
            //
            // byte: tree type code, 0 if regression, 1 if affine (unsupported), 2 if fast forest
            // int: number of leaves currently in the tree, L
            // double: maxoutput
            // double: weight
            // int[MN]: lte child, MN is inferred from the length of this array
            // int[MN]: gt child
            // int[MN]: split feature index
            // int[C]: categorical node indices.
            // int[C*(CT + 2)]: categorical feature values and categorical feature range in the input feature vector.
            // int[MN]: threshold bin index (can be null if raw thresholds are not null)
            // Float[MN]: raw threshold (can be not null if threshold bin indices are not null)
            // Float[MN]: default value For missing
            // double[ML]: leaf value
            // double[MN]: gain of this split (can be null)
            // double[MN]: p-value of this split (can be null)
            // double[MN]: previous value of a node before it made the transition from leaf to node (can be null)

            BinaryWriter writer = ctx.Writer;
            writer.Write((byte)code);
            writer.Write(NumLeaves);
            writer.Write(_maxOutput);
            writer.Write(Weight);

            writer.WriteIntArray(LteChild);
            writer.WriteIntArray(GtChild);
            writer.WriteIntArray(SplitFeatures);

            Contracts.Assert(CategoricalSplit != null);
            Contracts.Assert(CategoricalSplit.Length >= NumNodes);

            List<int> categoricalNodeIndices = new List<int>();
            for (int index = 0; index < NumNodes; index++)
            {
                if (CategoricalSplit[index])
                    categoricalNodeIndices.Add(index);
            }

            writer.WriteIntArray(categoricalNodeIndices.ToArray());

            for (int index = 0; index < categoricalNodeIndices.Count; index++)
            {
                int indexLocal = categoricalNodeIndices[index];

                Contracts.Assert(indexLocal >= 0 && indexLocal < NumNodes);
                Contracts.Assert(CategoricalSplitFeatures[indexLocal] != null &&
                                 CategoricalSplitFeatures[indexLocal].Length > 0);

                Contracts.Assert(CategoricalSplitFeatureRanges[indexLocal] != null &&
                                 CategoricalSplitFeatureRanges[indexLocal].Length == 2);

                writer.WriteIntArray(CategoricalSplitFeatures[indexLocal]);
                writer.WriteIntsNoCount(CategoricalSplitFeatureRanges[indexLocal].AsSpan(0, 2));
            }

            writer.WriteUIntArray(Thresholds);
            writer.WriteFloatArray(RawThresholds);
            writer.WriteFloatArray(DefaultValueForMissing);
            writer.WriteDoubleArray(LeafValues);

            writer.WriteDoubleArray(_splitGain);
            writer.WriteDoubleArray(_gainPValue);
            writer.WriteDoubleArray(_previousLeafValue);
        }

        public virtual void Save(ModelSaveContext ctx)
        {
            Save(ctx, TreeType.Regression);
        }

        public static RegressionTree Load(ModelLoadContext ctx, bool usingDefaultValues, bool categoricalSplits)
        {
            TreeType code = (TreeType)ctx.Reader.ReadByte();
            switch (code)
            {
            case TreeType.Regression:
                return new RegressionTree(ctx, usingDefaultValues, categoricalSplits);
            case TreeType.Affine:
                // Affine regression trees do not actually work, nor is it clear how they ever
                // could have worked within TLC, so the chance of this happening seems remote.
                throw Contracts.ExceptNotSupp("Affine regression trees unsupported");
            case TreeType.FastForest:
                return new QuantileRegressionTree(ctx, usingDefaultValues, categoricalSplits);
            default:
                throw Contracts.ExceptDecode();
            }
        }

        private void CheckValid(Action<bool, string> checker)
        {
            int numMaxNodes = Utils.Size(LteChild);
            int numMaxLeaves = numMaxNodes + 1;
            checker(NumLeaves > 1, "non-positive number of leaves");
            checker(Weight >= 0, "negative tree weight");
            checker(numMaxLeaves >= NumLeaves, "inconsistent number of leaves with maximum leaf capacity");
            checker(GtChild != null && GtChild.Length == numMaxNodes, "bad gtchild");
            checker(LteChild != null && LteChild.Length == numMaxNodes, "bad ltechild");
            checker(SplitFeatures != null && SplitFeatures.Length == numMaxNodes, "bad split feature length");
            checker(CategoricalSplit != null &&
                (CategoricalSplit.Length == numMaxNodes || CategoricalSplit.Length == NumNodes), "bad categorical split length");

            if (CategoricalSplit.Any(x => x))
            {
                checker(CategoricalSplitFeatures != null &&
                        (CategoricalSplitFeatures.Length == NumNodes ||
                         CategoricalSplitFeatures.Length == numMaxNodes),
                    "bad categorical split features length");

                checker(CategoricalSplitFeatureRanges != null &&
                        (CategoricalSplitFeatureRanges.Length == NumNodes ||
                         CategoricalSplitFeatureRanges.Length == numMaxNodes),
                    "bad categorical split feature ranges length");

                checker(CategoricalSplitFeatureRanges.All(x => x == null || x.Length == 0 || x.Length == 2),
                    "bad categorical split feature ranges values");

                for (int index = 0; index < CategoricalSplit.Length; index++)
                {
                    if (CategoricalSplit[index])
                    {
                        checker(CategoricalSplitFeatures[index] != null, "Categorical split features is null");
                        checker(CategoricalSplitFeatures[index].Length > 0,
                            "Categorical split features is zero length");

                        checker(CategoricalSplitFeatureRanges[index] != null,
                            "Categorical split feature ranges is null");

                        checker(CategoricalSplitFeatureRanges[index].Length == 2,
                            "Categorical split feature range length is not two.");

                        int previous = -1;
                        for (int featureIndex = 0; featureIndex < CategoricalSplitFeatures[index].Length; featureIndex++)
                        {
                            checker(CategoricalSplitFeatures[index][featureIndex] > previous,
                                "categorical split features is not sorted");

                            checker(CategoricalSplitFeatures[index][featureIndex] >=
                                    CategoricalSplitFeatureRanges[index][0] &&
                                    CategoricalSplitFeatures[index][featureIndex] <=
                                    CategoricalSplitFeatureRanges[index][1],
                                "categorical split features values are out of range.");

                            previous = CategoricalSplitFeatures[index][featureIndex];
                        }
                    }
                }
            }

            checker(Utils.Size(Thresholds) == 0 || Thresholds.Length == numMaxNodes, "bad threshold length");
            checker(Utils.Size(RawThresholds) == 0 || RawThresholds.Length == NumLeaves - 1, "bad rawthreshold length");
            checker(RawThresholds != null || Thresholds != null,
                "at most one of raw or indexed thresholds can be null");
            checker(Utils.Size(_splitGain) == 0 || _splitGain.Length == numMaxNodes, "bad splitgain length");
            checker(Utils.Size(_gainPValue) == 0 || _gainPValue.Length == numMaxNodes, "bad gainpvalue length");
            checker(Utils.Size(_previousLeafValue) == 0 || _previousLeafValue.Length == numMaxNodes, "bad previous leaf value length");
            checker(LeafValues != null && LeafValues.Length == numMaxLeaves, "bad leaf value length");
        }

        public virtual int SizeInBytes()
        {
            return NumLeaves.SizeInBytes() +
                _maxOutput.SizeInBytes() +
                Weight.SizeInBytes() +
                LteChild.SizeInBytes() +
                GtChild.SizeInBytes() +
                SplitFeatures.SizeInBytes() +
                (CategoricalSplitFeatures != null ? CategoricalSplitFeatures.Select(thresholds => thresholds.SizeInBytes()).Sum() : 0) +
                (CategoricalSplitFeatureRanges != null ? CategoricalSplitFeatureRanges.Select(ranges => ranges.SizeInBytes()).Sum() : 0) +
                NumNodes * sizeof(int) +
                CategoricalSplit.Length * sizeof(bool) +
                Thresholds.SizeInBytes() +
                RawThresholds.SizeInBytes() +
                _splitGain.SizeInBytes() +
                _gainPValue.SizeInBytes() +
                _previousLeafValue.SizeInBytes() +
                LeafValues.SizeInBytes();
        }

        public virtual void ToByteArray(byte[] buffer, ref int position)
        {
            NumLeaves.ToByteArray(buffer, ref position);
            _maxOutput.ToByteArray(buffer, ref position);
            Weight.ToByteArray(buffer, ref position);
            LteChild.ToByteArray(buffer, ref position);
            GtChild.ToByteArray(buffer, ref position);
            SplitFeatures.ToByteArray(buffer, ref position);
            CategoricalSplit.Length.ToByteArray(buffer, ref position);
            foreach (var split in CategoricalSplit)
                Convert.ToByte(split).ToByteArray(buffer, ref position);

            if (CategoricalSplitFeatures != null)
            {
                Contracts.AssertValue(CategoricalSplitFeatureRanges);
                for (int i = 0; i < CategoricalSplitFeatures.Length; i++)
                {
                    CategoricalSplitFeatures[i].ToByteArray(buffer, ref position);
                    CategoricalSplitFeatureRanges[i].ToByteArray(buffer, ref position);
                }
            }

            Thresholds.ToByteArray(buffer, ref position);
            RawThresholds.ToByteArray(buffer, ref position);
            _splitGain.ToByteArray(buffer, ref position);
            _gainPValue.ToByteArray(buffer, ref position);
            _previousLeafValue.ToByteArray(buffer, ref position);
            LeafValues.ToByteArray(buffer, ref position);
        }

        public void SumupValue(IChannel ch, RegressionTree tree)
        {
            if (LeafValues.Length != tree.LeafValues.Length)
            {
                throw Contracts.Except("cannot sumup value with different lengths");
            }

            for (int node = 0; node < LeafValues.Length; ++node)
            {
                if (node < LeafValues.Length - 1 &&
                    (LteChild[node] != tree.LteChild[node] ||
                    GtChild[node] != tree.GtChild[node] ||
                    Thresholds[node] != tree.Thresholds[node] ||
                    SplitFeatures[node] != tree.SplitFeatures[node] ||
                    Math.Abs(_splitGain[node] - tree._splitGain[node]) > 1e-6))
                {
                    ch.Warning(
                        "mismatch @ {10}: {0}/{1}, {2}/{3}, {4}/{5}, {6}/{7}, {8}/{9}",
                        LteChild[node],
                        tree.LteChild[node],
                        GtChild[node],
                        tree.GtChild[node],
                        Thresholds[node],
                        tree.Thresholds[node],
                        SplitFeatures[node],
                        tree.SplitFeatures[node],
                        _splitGain[node],
                        tree._splitGain[node],
                        node);

                    throw Contracts.Except("trees from different workers do not match");
                }

                LeafValues[node] += tree.LeafValues[node];
            }

        }

        /// <summary>
        /// The current number of leaves in the tree.
        /// </summary>
        public int NumLeaves { get; private set; }

        /// <summary>
        /// The current number of nodes in the tree.
        /// </summary>
        public int NumNodes => NumLeaves - 1;

        /// <summary>
        /// The maximum number of leaves the internal structure of this tree can support.
        /// </summary>
        public int MaxNumLeaves => LeafValues.Length;

        /// <summary>
        /// The maximum number of nodes this tree can support.
        /// </summary>
        public int MaxNumNodes => LteChild.Length;

        public double MaxOutput => _maxOutput;

        public double Weight { get; set; }

        public int GetLteChildForNode(int node)
        {
            return LteChild[node];
        }

        public int GetGtChildForNode(int node)
        {
            return GtChild[node];
        }

        public int SplitFeature(int node)
        {
            return SplitFeatures[node];
        }

        public UInt32 Threshold(int node)
        {
            return Thresholds[node];
        }

        public Float RawThreshold(int node)
        {
            return RawThresholds[node];
        }

        public double LeafValue(int leaf)
        {
            return LeafValues[leaf];
        }

        public void SetLeafValue(int leaf, double newValue)
        {
            LeafValues[leaf] = newValue;
        }

        // adds value to all of the tree outputs
        public void ShiftOutputs(double value)
        {
            for (int node = 0; node < LeafValues.Length; ++node)
                LeafValues[node] += value;
        }

        /// <summary>
        /// Scales all of the output values at the leaves of the tree by a given scalar
        /// </summary>
        public virtual void ScaleOutputsBy(double scalar)
        {
            for (int i = 0; i < LeafValues.Length; ++i)
            {
                LeafValues[i] *= scalar;
            }
        }

        public void UpdateOutputWithDelta(int leafIndex, double delta)
        {
            LeafValues[leafIndex] += delta;
        }

        /// <summary>
        /// Evaluates the regression tree on a given document.
        /// </summary>
        /// <param name="featureBins"></param>
        /// <returns>the real-valued regression tree output</returns>
        public virtual double GetOutput(Dataset.RowForwardIndexer.Row featureBins)
        {
            if (LteChild[0] == 0)
                return 0;
            int leaf = GetLeaf(featureBins);
            return GetOutput(leaf);
        }

        /// <summary>
        /// evaluates the regression tree on a given binnedinstance.
        /// </summary>
        /// <param name="binnedInstance">A previously binned instance/document</param>
        /// <returns>the real-valued regression tree output</returns>
        public virtual double GetOutput(int[] binnedInstance)
        {
            if (LteChild[0] == 0)
                return 0;
            int leaf = GetLeaf(binnedInstance);
            return GetOutput(leaf);
        }

        public virtual double GetOutput(in VBuffer<Float> feat)
        {
            if (LteChild[0] == 0)
                return 0;
            int leaf = GetLeaf(in feat);
            return GetOutput(leaf);
        }

        public double GetOutput(int leaf)
        {
            return LeafValues[leaf];
        }

        public void SetOutput(int leaf, double value)
        {
            LeafValues[leaf] = value;
        }

        // Returns index to a leaf a document belongs to
        // For empty tree returns 0
        public int GetLeaf(Dataset.RowForwardIndexer.Row featureBins)
        {
            // check for an empty tree
            if (NumLeaves == 1)
                return 0;

            int node = 0;
            while (node >= 0)
            {
                if (featureBins[SplitFeatures[node]] <= Thresholds[node])
                    node = LteChild[node];
                else
                    node = GtChild[node];
            }
            return ~node;
        }

        // Returns index to a leaf an instance/document belongs to.
        // For empty tree returns 0.
        public int GetLeaf(int[] binnedInstance)
        {
            // check for an empty tree
            if (NumLeaves == 1)
                return 0;

            int node = 0;
            while (node >= 0)
            {
                if (binnedInstance[SplitFeatures[node]] <= Thresholds[node])
                    node = LteChild[node];
                else
                    node = GtChild[node];
            }
            return ~node;
        }

        // Returns index to a leaf an instance/document belongs to.
        // Input are the raw feature values in dense format.
        // For empty tree returns 0.
        public int GetLeaf(in VBuffer<Float> feat)
        {
            // REVIEW: This really should validate feat.Length!
            if (feat.IsDense)
                return GetLeafCore(feat.Values);
            return GetLeafCore(feat.GetIndices(), feat.GetValues());
        }

        /// <summary>
        /// Returns leaf index the instance falls into, if we start the search from the <paramref name="root"/> node.
        /// </summary>
        private int GetLeafFrom(in VBuffer<Float> feat, int root)
        {
            if (root < 0)
            {
                // This is already a leaf.
                return ~root;
            }

            if (feat.IsDense)
                return GetLeafCore(feat.Values, root: root);
            return GetLeafCore(feat.GetIndices(), feat.GetValues(), root: root);
        }

        /// <summary>
        /// Returns the leaf node for the given feature vector, and populates 'path' with the list of internal nodes in the
        /// path from the root to that leaf. If 'path' is null a new list is initialized. All elements in 'path' are cleared
        /// before filling in the current path nodes.
        /// </summary>
        public int GetLeaf(in VBuffer<Float> feat, ref List<int> path)
        {
            // REVIEW: This really should validate feat.Length!
            if (path == null)
                path = new List<int>();
            else
                path.Clear();

            if (feat.IsDense)
                return GetLeafCore(feat.Values, path);
            return GetLeafCore(feat.GetIndices(), feat.GetValues(), path);
        }

        private Float GetFeatureValue(Float x, int node)
        {
            // Not need to convert missing vaules.
            if (DefaultValueForMissing == null)
                return x;

            if (Double.IsNaN(x))
            {
                return DefaultValueForMissing[node];
            }
            else
            {
                return x;
            }
        }

        private int GetLeafCore(Float[] nonBinnedInstance, List<int> path = null, int root = 0)
        {
            Contracts.AssertValue(nonBinnedInstance);
            Contracts.Assert(path == null || path.Count == 0);
            Contracts.Assert(root >= 0);

            // Check for an empty tree.
            if (NumLeaves == 1)
                return 0;

            int node = root;
            if (path == null)
            {
                while (node >= 0)
                {
                    //REVIEW: Think about optimizing dense case for performance.
                    if (CategoricalSplit[node])
                    {
                        Contracts.Assert(CategoricalSplitFeatures != null);

                        int newNode = LteChild[node];
                        foreach (var indices in CategoricalSplitFeatures[node])
                        {
                            Float fv = GetFeatureValue(nonBinnedInstance[indices], node);
                            if (fv > 0.0f)
                            {
                                newNode = GtChild[node];
                                break;
                            }
                        }

                        node = newNode;
                    }
                    else
                    {

                        Float fv = GetFeatureValue(nonBinnedInstance[SplitFeatures[node]], node);
                        if (fv <= RawThresholds[node])
                            node = LteChild[node];
                        else
                            node = GtChild[node];
                    }
                }
            }
            else
            {
                while (node >= 0)
                {
                    path.Add(node);
                    if (CategoricalSplit[node])
                    {
                        Contracts.Assert(CategoricalSplitFeatures != null);

                        int newNode = LteChild[node];
                        foreach (var index in CategoricalSplitFeatures[node])
                        {
                            Float fv = GetFeatureValue(nonBinnedInstance[index], node);
                            if (fv > 0.0f)
                            {
                                newNode = GtChild[node];
                                break;
                            }
                        }

                        node = newNode;
                    }
                    else
                    {
                        Float fv = GetFeatureValue(nonBinnedInstance[SplitFeatures[node]], node);
                        if (fv <= RawThresholds[node])
                            node = LteChild[node];
                        else
                            node = GtChild[node];
                    }
                }
            }
            return ~node;
        }

        private int GetLeafCore(ReadOnlySpan<int> featIndices, ReadOnlySpan<Float> featValues, List<int> path = null, int root = 0)
        {
            Contracts.Assert(featIndices.Length == featValues.Length);
            Contracts.Assert(path == null || path.Count == 0);
            Contracts.Assert(root >= 0);

            int count = featValues.Length;

            // check for an empty tree
            if (NumLeaves == 1)
                return 0;

            int node = root;

            while (node >= 0)
            {
                if (path != null)
                    path.Add(node);

                if (CategoricalSplit[node])
                {
                    Contracts.Assert(CategoricalSplitFeatures != null);
                    Contracts.Assert(CategoricalSplitFeatureRanges != null);

                    //REVIEW: Consider experimenting with bitmap instead of doing log(n) binary search.
                    int newNode = LteChild[node];
                    int end = featIndices.FindIndexSorted(0, count, CategoricalSplitFeatureRanges[node][1]);
                    for (int i = featIndices.FindIndexSorted(0, count, CategoricalSplitFeatureRanges[node][0]);
                         i < count && i <= end;
                         ++i)
                    {
                        int index = featIndices[i];
                        if (CategoricalSplitFeatures[node].TryFindIndexSorted(0, CategoricalSplitFeatures[node].Length, index, out int ii))
                        {
                            Float val = GetFeatureValue(featValues[i], node);
                            if (val > 0.0f)
                            {
                                newNode = GtChild[node];
                                break;
                            }
                        }
                    }

                    node = newNode;
                }
                else
                {
                    Float val = 0;
                    int ifeat = SplitFeatures[node];

                    int ii = featIndices.FindIndexSorted(0, count, ifeat);
                    if (ii < count && featIndices[ii] == ifeat)
                        val = featValues[ii];
                    val = GetFeatureValue(val, node);
                    if (val <= RawThresholds[node])
                        node = LteChild[node];
                    else
                        node = GtChild[node];
                }
            }
            return ~node;
        }

        //Returns all leafs indexes belonging for a given node.
        //If node<0 it treats it node as being a leaf and returns ~node
        public IEnumerable<int> GetNodesLeaves(int node)
        {
            if (NumLeaves == 1)
                return Enumerable.Range(0, NumLeaves);
            if (node < 0)
                return Enumerable.Range(~node, 1);
            return GetNodesLeaves(LteChild[node]).Concat(GetNodesLeaves(GtChild[node]));
        }

        /// <summary>
        /// returns the hypothesis output for an entire dataset
        /// </summary>
        public double[] GetOutputs(Dataset dataset)
        {
            var featureBinRows = dataset.GetFeatureBinRowwiseIndexer();
            double[] outputs = new double[dataset.NumDocs];
            for (int d = 0; d < dataset.NumDocs; ++d)
                outputs[d] = GetOutput(featureBinRows[d]);
            return outputs;
        }

        /// <summary>
        /// Turns a leaf of the tree into an interior node with two leaf-children.
        /// </summary>
        /// <param name="leaf">The index of the leaf to split.</param>
        /// <param name="feature">The index of the feature used to split this leaf (as
        /// it indexes the array of DerivedFeature instances passed to the to tree ensemble format).</param>
        /// <param name="categoricalSplitFeatures">Thresholds for categorical split.</param>
        /// <param name="categoricalSplitRange"></param>
        /// <param name="categoricalSplit"></param>
        /// <param name="threshold">The </param>
        /// <param name="lteValue">The value of the leaf on the LTE side.</param>
        /// <param name="gtValue">The value of the leaf on the GT side.</param>
        /// <param name="gain">The splitgain of this split. This does not
        /// affect the logic of the tree evaluation.</param>
        /// <param name="gainPValue">The p-value associated with this split,
        /// indicating confidence that this is a better than random split.
        /// This does not affect the logic of the tree evaluation.</param>
        /// <returns>Returns the node index</returns>
        public virtual int Split(int leaf, int feature, int[] categoricalSplitFeatures, int[]
            categoricalSplitRange, bool categoricalSplit, uint threshold, double lteValue, double gtValue, double gain, double gainPValue)
        {
            int indexOfNewNonLeaf = NumLeaves - 1;

            // find the leaf's parent, and update its info
            int parent = Array.FindIndex(LteChild, x => x == ~leaf);
            if (parent >= 0)
                LteChild[parent] = indexOfNewNonLeaf;
            else
            {
                parent = Array.FindIndex(GtChild, x => x == ~leaf);
                if (parent >= 0)
                    GtChild[parent] = indexOfNewNonLeaf;
            }

            // define a new non-leaf, set its info and the info of its two new children
            SplitFeatures[indexOfNewNonLeaf] = feature;

            //Lazily initialize categorical split data structures.
            if (categoricalSplit && CategoricalSplitFeatures == null)
            {
                Contracts.Assert(CategoricalSplitFeatureRanges == null);
                CategoricalSplitFeatures = new int[MaxNumNodes][];
                CategoricalSplitFeatureRanges = new int[MaxNumNodes][];
            }

            if (categoricalSplit)
            {
                CategoricalSplitFeatures[indexOfNewNonLeaf] = categoricalSplitFeatures;
                CategoricalSplitFeatureRanges[indexOfNewNonLeaf] = categoricalSplitRange;
            }

            CategoricalSplit[indexOfNewNonLeaf] = categoricalSplit;
            _splitGain[indexOfNewNonLeaf] = gain;
            _gainPValue[indexOfNewNonLeaf] = gainPValue;
            Thresholds[indexOfNewNonLeaf] = threshold;
            LteChild[indexOfNewNonLeaf] = ~leaf;
            _previousLeafValue[indexOfNewNonLeaf] = LeafValues[leaf];
            LeafValues[leaf] = lteValue;
            GtChild[indexOfNewNonLeaf] = ~NumLeaves;
            LeafValues[NumLeaves] = gtValue;

            // set the maxOutput of this tree
            if (lteValue > _maxOutput)
                _maxOutput = lteValue;
            if (gtValue > _maxOutput)
                _maxOutput = gtValue;

            // increment counters
            ++NumLeaves;

            // return index of new guy
            return indexOfNewNonLeaf;
        }

        // returns a unique hash code that represents this tree
        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        public void PopulateRawThresholds(Dataset dataset)
        {
            var features = dataset.Flocks;
            if (RawThresholds != null)
                return;

            int numNodes = NumLeaves - 1;

            RawThresholds = new Float[numNodes];
            for (int n = 0; n < numNodes; n++)
            {
                int flock;
                int subfeature;
                dataset.MapFeatureToFlockAndSubFeature(SplitFeatures[n], out flock, out subfeature);
                if (CategoricalSplit[n] == false)
                    RawThresholds[n] = (Float)dataset.Flocks[flock].BinUpperBounds(subfeature)[Thresholds[n]];
                else
                    RawThresholds[n] = 0.5f;
            }
        }

        public void RemapFeatures(int[] oldToNewFeatures)
        {
            Contracts.AssertValue(oldToNewFeatures);
            int numNodes = NumLeaves - 1;

            for (int n = 0; n < numNodes; n++)
            {
                Contracts.Assert(0 <= SplitFeatures[n] && SplitFeatures[n] < oldToNewFeatures.Length);
                SplitFeatures[n] = oldToNewFeatures[SplitFeatures[n]];
                if (CategoricalSplit[n])
                {
                    Contracts.Assert(CategoricalSplitFeatures[n] != null);
                    Contracts.Assert(CategoricalSplitFeatureRanges[n] != null &&
                        (CategoricalSplitFeatureRanges[n].Length == 2 || CategoricalSplitFeatureRanges[n].Length == 0));

                    for (int i = 0; i < CategoricalSplitFeatures[n].Length; i++)
                        CategoricalSplitFeatures[n][i] = oldToNewFeatures[CategoricalSplitFeatures[n][i]];

                    for (int i = 0; i < CategoricalSplitFeatureRanges[n].Length; i++)
                        CategoricalSplitFeatureRanges[n][i] = oldToNewFeatures[CategoricalSplitFeatureRanges[n][i]];
                }
            }
        }

        /// <summary>
        /// Returns a representation of the tree in the production
        /// decision tree format (SHAREDDYNAMICRANKROOT\TreeEnsembleRanker\Tree.h).
        /// The intent is that this
        /// </summary>
        /// <param name="sbEvaluator">Append the new evaluator to this stringbuilder.</param>
        /// <param name="sbInput">Append any hitherto unused [Input:#] sections
        /// to this stringbuilder.</param>
        /// <param name="featureContents">The feature to content map.</param>
        /// <param name="evaluatorCounter">A running count of evaluators. When
        /// this method returns it should have one more entry.</param>
        /// <param name="featureToId">A map of feature index (in the features array)
        /// to the ID as it will be written in the file. This instance should be
        /// used for all </param>
        public void ToTreeEnsembleFormat(StringBuilder sbEvaluator, StringBuilder sbInput, FeaturesToContentMap featureContents,
            ref int evaluatorCounter, Dictionary<int, int> featureToId)
        {
            Contracts.AssertValue(sbEvaluator);
            Contracts.AssertValue(sbInput);
            Contracts.AssertValue(featureContents);

            Dictionary<int, int> categoricalSplitNodeToId = new Dictionary<int, int>();
            ToTreeEnsembleFormatForCategoricalSplit(sbEvaluator, sbInput, featureContents, ref evaluatorCounter,
                featureToId, categoricalSplitNodeToId);

            ++evaluatorCounter;

            int numNonLeaves = NumLeaves - 1;

            sbEvaluator.AppendFormat("\n[Evaluator:{0}]\n", evaluatorCounter);
            sbEvaluator.AppendFormat("EvaluatorType=DecisionTree\nNumInternalNodes={0}\n", numNonLeaves);

            StringBuilder sbFeatures = new StringBuilder("SplitFeatures=");
            StringBuilder sbSplitGain = new StringBuilder("\nSplitGain=");
            StringBuilder sbGainPValue = _gainPValue != null ? new StringBuilder("\nGainPValue=") : null;
            StringBuilder sbLteChild = new StringBuilder("\nLTEChild=");
            StringBuilder sbGtChild = new StringBuilder("\nGTChild=");
            StringBuilder sbOutput = new StringBuilder("\nOutput=");
            StringBuilder sbThreshold = new StringBuilder("\nThreshold=");

            for (int n = 0; n < numNonLeaves; ++n)
            {
                string toAppend = (n < numNonLeaves - 1 ? "\t" : "");

                if (CategoricalSplit[n])
                    sbFeatures.Append("E:" + categoricalSplitNodeToId[n] + toAppend);
                else
                {
                    if (!featureToId.ContainsKey(SplitFeatures[n]))
                    {
                        sbInput.AppendFormat("\n[Input:{0}]\n", featureToId.Count + 1);
                        sbInput.Append(featureContents.GetContent(SplitFeatures[n]));
                        sbInput.Append("\n");

                        featureToId.Add(SplitFeatures[n], featureToId.Count + 1);
                    }

                    sbFeatures.Append("I:" + featureToId[SplitFeatures[n]] + toAppend);
                }

                sbSplitGain.Append(_splitGain[n].ToString() + toAppend);
                sbGainPValue?.Append(_gainPValue[n].ToString("0.000e00") + toAppend);

                int lteChildCorrected = LteChild[n];
                if (lteChildCorrected < 0)
                    lteChildCorrected = -1 - (~lteChildCorrected);
                sbLteChild.Append(lteChildCorrected.ToString() + toAppend);

                int gtChildCorrected = GtChild[n];
                if (gtChildCorrected < 0)
                    gtChildCorrected = -1 - (~gtChildCorrected);
                sbGtChild.Append(gtChildCorrected.ToString() + toAppend);

                sbOutput.Append(LeafValues[n].ToString() + "\t");

                double threshold = RawThresholds[n];
                sbThreshold.Append(threshold.ToString("R") + toAppend);
            }

            sbOutput.Append(LeafValues[numNonLeaves].ToString());

            sbEvaluator.AppendFormat("{0}{1}{2}{3}{4}{5}{6}\n",
                sbFeatures.ToString(), sbSplitGain.ToString(), sbGainPValue?.ToString(),
                sbLteChild.ToString(), sbGtChild.ToString(), sbThreshold.ToString(),
                sbOutput.ToString());

        }

        private void ToTreeEnsembleFormatForCategoricalSplit(StringBuilder sbEvaluator, StringBuilder sbInput, FeaturesToContentMap featureContents,
            ref int evaluatorCounter, Dictionary<int, int> featureToId, Dictionary<int, int> categoricalSplitNodeToId)
        {
            //REVIEW: Can all these conditions even be true?
            if (CategoricalSplitFeatures == null ||
                CategoricalSplitFeatures.Length == 0 ||
                CategoricalSplitFeatures.All(val => val == null))
            {
                return;
            }

            Contracts.AssertValue(sbEvaluator);
            Contracts.AssertValue(sbInput);
            Contracts.AssertValue(featureContents);
            Contracts.Assert(CategoricalSplitFeatures.Length == NumNodes);

            for (int i = 0; i < CategoricalSplitFeatures.Length; ++i)
            {
                if (!CategoricalSplit[i])
                    continue;

                ++evaluatorCounter;
                categoricalSplitNodeToId.Add(i, evaluatorCounter);

                //REVIEW: What happens when the threshold length is zero in the case of -Infinity gain? Is empty tree ok?
                int numNonLeaves = CategoricalSplitFeatures[i].Length;

                sbEvaluator.AppendFormat("\n[Evaluator:{0}]\n", evaluatorCounter);
                sbEvaluator.AppendFormat("EvaluatorType=DecisionTree\nNumInternalNodes={0}\n", numNonLeaves);

                StringBuilder sbFeatures = new StringBuilder("SplitFeatures=");
                StringBuilder sbLteChild = new StringBuilder("\nLTEChild=");
                StringBuilder sbGtChild = new StringBuilder("\nGTChild=");
                StringBuilder sbOutput = new StringBuilder("\nOutput=");
                StringBuilder sbThreshold = new StringBuilder("\nThreshold=");

                for (int n = 0; n < numNonLeaves; ++n)
                {
                    string toAppend = (n < numNonLeaves - 1 ? "\t" : "");
                    int categoricalSplitFeature = CategoricalSplitFeatures[i][n];
                    if (!featureToId.ContainsKey(categoricalSplitFeature))
                    {
                        sbInput.AppendFormat("\n[Input:{0}]\n", featureToId.Count + 1);
                        sbInput.Append(featureContents.GetContent(categoricalSplitFeature));
                        sbInput.Append("\n");

                        featureToId.Add(categoricalSplitFeature, featureToId.Count + 1);
                    }

                    sbFeatures.Append("I:" + featureToId[categoricalSplitFeature] + toAppend);
                    sbLteChild.Append((n + 1) + toAppend);
                    sbGtChild.Append(~n + toAppend);
                    sbOutput.Append("1\t");
                    sbThreshold.Append(((double)0.5).ToString("R") + toAppend);
                }

                sbOutput.Append("0");

                sbEvaluator.AppendFormat("{0}{1}{2}{3}{4}\n",
                    sbFeatures.ToString(), sbLteChild.ToString(), sbGtChild.ToString(), sbThreshold.ToString(),
                    sbOutput.ToString());
            }

        }

        // prints the tree out as a string (in old Bing format used by LambdaMART and AdIndex)
        public string ToOldIni(FeatureNameCollection featureNames)
        {
            // print the root node
            StringBuilder output = new StringBuilder();
            output.Append("Name=AnchorMostFrequent\nTransform=DecisionTree");

            int numNonLeaves = NumLeaves - 1;
            for (int n = 0; n < numNonLeaves; ++n)
            {
                string name = featureNames[SplitFeatures[n]];
                double currentThreshold = RawThresholds[n];

                int lteChildCorrected = LteChild[n];
                if (lteChildCorrected < 0)
                    lteChildCorrected = numNonLeaves + (~lteChildCorrected);
                int gtChildCorrected = GtChild[n];
                if (gtChildCorrected < 0)
                    gtChildCorrected = numNonLeaves + (~gtChildCorrected);

                output.AppendFormat("\nNodeType:{0}=Branch\nNodeDecision:{0}={1}\nNodeThreshold:{0}={2}\nNodeLTE:{0}={3}\nNodeGT:{0}={4}\n", n, name, currentThreshold, lteChildCorrected, gtChildCorrected);
            }

            for (int n = 0; n < NumLeaves; ++n)
            {
                output.AppendFormat("\nNodeType:{0}=Value\nNodeValue:{0}={1}\n", numNonLeaves + n, LeafValues[n]);
            }

            return output.ToString();
        }

        internal JToken AsPfa(JToken feat)
        {
            return AsPfaCore(feat, 0);
        }

        private JToken AsPfaCore(JToken feat, int node)
        {
            // REVIEW: This function works through explicit recursion, which is
            // dangerous due to stack overflow issues. If it becomes an issue we should
            // switch to an iterative and non-recursive approach.
            Contracts.Assert(-NumLeaves <= node && node < NumNodes);
            if (node < 0)
                return LeafValues[~node];
            JToken lte = AsPfaCore(feat, GetLteChildForNode(node));
            JToken gt = AsPfaCore(feat, GetGtChildForNode(node));
            return PfaUtils.If(PfaUtils.Call("<=", PfaUtils.Index(feat, SplitFeatures[node]), RawThresholds[node]), lte, gt);
        }

        public FeatureToGainMap GainMap
        {
            get
            {
                var result = new FeatureToGainMap();
                int numNonLeaves = NumLeaves - 1;
                for (int n = 0; n < numNonLeaves; ++n)
                    result[SplitFeatures[n]] += _splitGain[n];
                return result;
            }
        }

        // adds the outputs of this hypothesis to an existing scores vector
        // returns the mean output of weak hypothesis on the dataset
        public void AddOutputsToScores(Dataset dataset, double[] scores, double multiplier)
        {
            if (multiplier == 1.0)
            {
                AddOutputsToScores(dataset, scores);
                return;
            }

            // Just break it up into NumThreads chunks. This minimizes the number of recomputations
            //  neccessary in the rowwise indexer.
            int innerLoopSize = 1 + dataset.NumDocs / BlockingThreadPool.NumThreads;   // +1 is to make sure we don't have a few left over at the end
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * dataset.NumDocs / innerLoopSize)];
            var actionIndex = 0;
            for (int d = 0; d < dataset.NumDocs; d += innerLoopSize)
            {
                var fromDoc = d;
                var toDoc = Math.Min(d + innerLoopSize, dataset.NumDocs);
                actions[actionIndex++] = () =>
                  {
                      var featureBins = dataset.GetFeatureBinRowwiseIndexer();
                      for (int doc = fromDoc; doc < toDoc; doc++)
                          scores[doc] += multiplier * GetOutput(featureBins[doc]);
                  };
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
        }

        // adds the outputs of this hypothesis to an existing scores vector
        // returns the mean output of weak hypothesis on the dataset
        public void AddOutputsToScores(Dataset dataset, double[] scores)
        {
            // Just break it up into NumThreads chunks. This minimizes the number of recomputations
            //  neccessary in the rowwise indexer.
            int innerLoopSize = 1 + dataset.NumDocs / BlockingThreadPool.NumThreads;   // +1 is to make sure we don't have a few left over at the end
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * dataset.NumDocs / innerLoopSize)];
            var actionIndex = 0;
            for (int d = 0; d < dataset.NumDocs; d += innerLoopSize)
            {
                var fromDoc = d;
                var toDoc = Math.Min(d + innerLoopSize, dataset.NumDocs);
                actions[actionIndex++] = () =>
                  {
                      var featureBins = dataset.GetFeatureBinRowwiseIndexer(ActiveFeatures);
                      for (int doc = fromDoc; doc < toDoc; doc++)
                          scores[doc] += GetOutput(featureBins[doc]);
                  };
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
        }

        internal void AddOutputsToScores(Dataset dataset, double[] scores, int[] docIndices)
        {
            // Just break it up into NumThreads chunks. This minimizes the number of recomputations
            //  neccessary in the rowwise indexer.
            int innerLoopSize = 1 + docIndices.Length / BlockingThreadPool.NumThreads;   // +1 is to make sure we don't have a few left over at the end
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * docIndices.Length / innerLoopSize)];
            var actionIndex = 0;
            for (int d = 0; d < docIndices.Length; d += innerLoopSize)
            {
                var fromDoc = d;
                var toDoc = Math.Min(d + innerLoopSize, docIndices.Length);
                actions[actionIndex++] = () =>
                  {
                      var featureBins = dataset.GetFeatureBinRowwiseIndexer();
                      for (int doc = fromDoc; doc < toDoc; doc++)
                          scores[docIndices[doc]] += GetOutput(featureBins[docIndices[doc]]);
                  };
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
        }

        // -- tree optimization code
        /// <summary>
        /// Sets the path to a leaf to be indexed by 0,1,2,3,... and sets the leaf index to 0
        /// </summary>
        public void OptimizePathToLeaf(int leafIndex)
        {
            int i = 1;
            foreach (int nodeIndex in PathToLeaf(leafIndex))
            {
                SwapNodePositions(nodeIndex, i);
                ++i;
            }
        }

        /// <summary>
        ///  swaps the positions of two nodes in the tree, without any functional change to the tree
        /// </summary>
        public void SwapNodePositions(int pos1, int pos2)
        {
            if (pos1 == pos2)
                return;
            if (pos1 <= 0 || pos2 <= 0)
                throw Contracts.Except("Cannot swap root or leaves");

            int parentOfLteChild1 = Array.IndexOf(LteChild, pos1);
            int parentOfGtChild1 = Array.IndexOf(GtChild, pos1);
            int parentOfLteChild2 = Array.IndexOf(LteChild, pos2);
            int parentOfGtChild2 = Array.IndexOf(GtChild, pos2);

            if (parentOfLteChild1 >= 0)
                LteChild[parentOfLteChild1] = pos2;
            else
                GtChild[parentOfGtChild1] = pos2;
            if (parentOfLteChild2 >= 0)
                LteChild[parentOfLteChild2] = pos1;
            else
                GtChild[parentOfGtChild2] = pos1;

            int lteChild1 = LteChild[pos1];
            int gtChild1 = GtChild[pos1];
            uint threshold1 = Thresholds[pos1];
            int splitFeature1 = SplitFeatures[pos1];

            LteChild[pos1] = LteChild[pos2];
            GtChild[pos1] = GtChild[pos2];
            Thresholds[pos1] = Thresholds[pos2];
            SplitFeatures[pos1] = SplitFeatures[pos2];

            LteChild[pos2] = lteChild1;
            GtChild[pos2] = gtChild1;
            Thresholds[pos2] = threshold1;
            SplitFeatures[pos2] = splitFeature1;
        }

        public int[] PathToLeaf(int leafIndex)
        {
            List<int> path = new List<int>();
            if (!PathToLeaf(LteChild[0], leafIndex, path))
                PathToLeaf(GtChild[0], leafIndex, path);

            return path.ToArray();
        }

        private bool PathToLeaf(int currentNodeIndex, int leafIndex, List<int> path)
        {
            if (currentNodeIndex < 0)
            {
                if (~currentNodeIndex == leafIndex)
                    return true;
                return false;
            }

            path.Add(currentNodeIndex);
            if (PathToLeaf(LteChild[currentNodeIndex], leafIndex, path))
                return true;
            if (PathToLeaf(GtChild[currentNodeIndex], leafIndex, path))
                return true;
            path.RemoveAt(path.Count - 1);

            return false;
        }

        public void AppendFeatureContributions(in VBuffer<Float> src, BufferBuilder<Float> contributions)
        {
            if (LteChild[0] == 0)
            {
                // There is no root split, so no contributions.
                return;
            }

            // Walk down to the leaf, to get the true output.
            var mainLeaf = GetLeaf(in src);
            var trueOutput = GetOutput(mainLeaf);

            // Now walk down again, spawning ghost instances to calculate deltas.
            // This largely repeats GetLeafCore.
            int node = 0;
            while (node >= 0)
            {
                int ifeat = SplitFeatures[node];
                var val = src.GetItemOrDefault(ifeat);
                val = GetFeatureValue(val, node);
                int otherWay;
                if (val <= RawThresholds[node])
                {
                    otherWay = GtChild[node];
                    node = LteChild[node];
                }
                else
                {
                    otherWay = LteChild[node];
                    node = GtChild[node];
                }

                // What if we went the other way?
                var ghostLeaf = GetLeafFrom(in src, otherWay);
                var ghostOutput = GetOutput(ghostLeaf);

                // If the ghost got a smaller output, the contribution of the feature is positive, so
                // the contribution is true minus ghost.
                contributions.AddFeature(ifeat, (Float)(trueOutput - ghostOutput));
            }
        }
    }
}
