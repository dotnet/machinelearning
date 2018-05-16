// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public class Ensemble
    {
        private readonly string _firstInputInitializationContent;
        private readonly List<RegressionTree> _trees;

        public IEnumerable<RegressionTree> Trees => _trees;

        public double Bias { get; set; }

        public int NumTrees => _trees.Count;

        public Ensemble()
        {
            _trees = new List<RegressionTree>();
        }

        public Ensemble(ModelLoadContext ctx, bool usingDefaultValues, bool categoricalSplits)
        {
            // REVIEW: Verify the contents of the ensemble, both during building,
            // and during deserialization.

            // *** Binary format ***
            // int: Number of trees
            // Regression trees (num trees of these)
            // double: Bias
            // int: Id to InputInitializationContent string, currently ignored

            _trees = new List<RegressionTree>();
            int numTrees = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(numTrees >= 0);
            for (int t = 0; t < numTrees; ++t)
                AddTree(RegressionTree.Load(ctx, usingDefaultValues, categoricalSplits));
            Bias = ctx.Reader.ReadDouble();
            _firstInputInitializationContent = ctx.LoadStringOrNull();
        }

        public void Save(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // int: Number of trees
            // Regression trees (num trees of these)
            // double: Bias
            // int: Id to InputInitializationContent string

            BinaryWriter writer = ctx.Writer;
            writer.Write(NumTrees);
            foreach (RegressionTree tree in Trees)
                tree.Save(ctx);
            writer.Write(Bias);
            ctx.SaveStringOrNull(_firstInputInitializationContent);
        }

        public void AddTree(RegressionTree tree) => _trees.Add(tree);
        public void AddTreeAt(RegressionTree tree, int index) => _trees.Insert(index, tree);
        public void RemoveTree(int index) => _trees.RemoveAt(index);
        public void RemoveAfter(int index) => _trees.RemoveRange(index, NumTrees - index);
        public RegressionTree GetTreeAt(int index) => _trees[index];

        /// <summary>
        /// Converts the bin based thresholds to the raw real-valued thresholds.
        /// To be called after training the ensemble.
        /// </summary>
        /// <param name="dataset">The dataset from which to get the bin upper bounds per feature</param>
        public void PopulateRawThresholds(Dataset dataset)
        {
            for (int i = 0; i < NumTrees; i++)
                GetTreeAt(i).PopulateRawThresholds(dataset);
        }

        /// <summary>
        /// Remaps the features, to a new feature space. Is called in the event that the features
        /// in the training <see cref="Dataset"/> structure are different from the ones in the
        /// original pipeline (possibly due to trivialization of input features), and so need to
        /// be remapped back to the original space. Note that the tree once modified in this way
        /// will no longer have features pointing to the original training <see cref="Dataset"/>,
        /// so this should be called only after <see cref="PopulateRawThresholds"/> is called.
        /// </summary>
        /// <param name="oldToNewFeatures">The mapping from old original features, into the new features</param>
        public void RemapFeatures(int[] oldToNewFeatures)
        {
            Contracts.AssertValue(oldToNewFeatures);
            for (int i = 0; i < NumTrees; i++)
                GetTreeAt(i).RemapFeatures(oldToNewFeatures);
        }

        /// <summary>
        /// returns the ensemble in the production TreeEnsemble format
        /// </summary>
        public string ToTreeEnsembleIni(FeaturesToContentMap fmap,
            string trainingParams, bool appendFeatureGain, bool includeZeroGainFeatures = true)
        {
            StringBuilder sbEvaluator = new StringBuilder();
            StringBuilder sbInput = new StringBuilder();
            StringBuilder sb = new StringBuilder();

            Dictionary<int, int> featureToID = new Dictionary<int, int>();  //Mapping from feature to ini input id
            int numNodes = 0;

            // Append the pretrained input
            if (_firstInputInitializationContent != null)
            {
                numNodes++;
                featureToID[-1] = 0;
                sbInput.AppendFormat("\n[Input:1]\n{0}\n", _firstInputInitializationContent);
            }

            int evaluatorCounter = 0;
            for (int w = 0; w < NumTrees; ++w)
                _trees[w].ToTreeEnsembleFormat(sbEvaluator, sbInput, fmap, ref evaluatorCounter, featureToID);

            numNodes += evaluatorCounter;

            sb.AppendFormat("[TreeEnsemble]\nInputs={0}\nEvaluators={1}\n", featureToID.Count, evaluatorCounter + 1);

            sb.Append(sbInput);
            sb.Append(sbEvaluator);

            // Append the final aggregator
            sb.AppendFormat("\n[Evaluator:{0}]\nEvaluatorType=Aggregator\nNumNodes={1}\nNodes=", evaluatorCounter + 1, numNodes);

            // Nodes
            if (_firstInputInitializationContent != null)
            {
                sb.Append("I:1");
            }
            if (NumTrees > 0)
            {
                if (_firstInputInitializationContent != null)
                    sb.Append("\t");
                sb.Append("E:1");
            }
            for (int w = 1; w < NumTrees; ++w)
            {
                sb.AppendFormat("\tE:{0}", w + 1);
            }

            // weights
            sb.Append("\nWeights=");
            if (_firstInputInitializationContent != null)
            {
                sb.AppendFormat("1");
            }

            if (NumTrees > 0)
            {
                if (_firstInputInitializationContent != null)
                    sb.Append("\t");
                sb.AppendFormat("{0}", _trees[0].Weight);
            }

            for (int w = 1; w < NumTrees; ++w)
            {
                if (w > 0)
                {
                    sb.Append("\t");
                }
                sb.Append(_trees[w].Weight);
            }

            sb.AppendFormat("\nBias={0}", Bias);
            sb.Append("\nType=Linear");

            // Add comments section with training parameters

            int commentsWritten = AppendComments(sb, trainingParams);
            if (appendFeatureGain)
            {
                var gainSummary = ToGainSummary(fmap, featureToID, NumTrees, includeZeroGainFeatures,
                    normalize: false, startingCommentNumber: commentsWritten);
                sb.Append(gainSummary);
            }

            return sb.ToString();
        }

        protected int AppendComments(StringBuilder sb, string trainingParams)
        {
            sb.AppendFormat("\n\n[Comments]\nC:0=Regression Tree Ensemble\nC:1=Generated using FastTree\nC:2=Created on {0}\n", DateTime.UtcNow);

            string[] trainingParamsList = trainingParams.Split(new char[] { '\n' });
            int i = 0;
            for (; i < trainingParamsList.Length; ++i)
            {
                if (trainingParamsList[i].Length > 0)
                {
                    sb.AppendFormat("C:{0}=PARAM:{1}\n", i + 3, trainingParamsList[i]);
                }
            }
            return i + 3;
        }

        internal JToken AsPfa(BoundPfaContext ctx, JToken feat)
        {
            var toAdd = new JArray();
            foreach (var tree in Trees)
                toAdd.Add(tree.AsPfa(feat));
            if (Bias != 0)
                toAdd.Add(Bias);
            switch (toAdd.Count)
            {
                case 0:
                    return 0.0;
                case 1:
                    return toAdd[0];
                default:
                    return PfaUtils.Call("a.sum", ((JObject)null)
                        .AddReturn("type", PfaUtils.Type.Array(PfaUtils.Type.Double)).AddReturn("new", toAdd));
            }
        }

        public double[] MaxOutputs()
        {
            double[] values = new double[_trees.Count];
            for (int w = 0; w < _trees.Count; ++w)
                values[w] = _trees[w].MaxOutput;
            return values;
        }

        public double GetOutput(Dataset.RowForwardIndexer.Row featureBins, int prefix)
        {
            double output = 0.0;
            for (int h = 0; h < prefix; h++)
                output += _trees[h].GetOutput(featureBins);
            return output;
        }

        public double GetOutput(int[] binnedInstance)
        {
            double output = 0.0;
            for (int h = 0; h < NumTrees; h++)
                output += _trees[h].GetOutput(binnedInstance);
            return output;
        }

        public double GetOutput(ref VBuffer<float> feat)
        {
            double output = 0.0;
            for (int h = 0; h < NumTrees; h++)
                output += _trees[h].GetOutput(ref feat);
            return output;
        }

        public float[] GetDistribution(ref VBuffer<float> feat, int sampleCount, out float[] weights)
        {
            var distribution = new float[sampleCount * NumTrees];

            if (((QuantileRegressionTree)_trees[0]).IsWeightedTargets)
                weights = new float[sampleCount * NumTrees];
            else
                weights = null;

            for (int h = 0; h < NumTrees; h++)
            {
                ((QuantileRegressionTree)_trees[h]).LoadSampledLabels(ref feat, distribution,
                    weights, sampleCount, h * sampleCount);
            }
            return distribution;
        }

        public double GetOutput(Dataset.RowForwardIndexer.Row featureBins)
        {
            return GetOutput(featureBins, _trees.Count);
        }
        public void GetOutputs(Dataset dataset, double[] outputs) { GetOutputs(dataset, outputs, -1); }
        public void GetOutputs(Dataset dataset, double[] outputs, int prefix)
        {
            if (prefix > _trees.Count || prefix < 0)
                prefix = _trees.Count;

            int innerLoopSize = 1 + dataset.NumDocs / BlockingThreadPool.NumThreads;  // minimize number of times we have to skip forward in the sparse arrays
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * dataset.NumDocs / innerLoopSize)];
            var actionIndex = 0;
            for (int d = 0; d < dataset.NumDocs; d += innerLoopSize)
            {
                actions[actionIndex++] = () =>
                  {
                      var featureBins = dataset.GetFeatureBinRowwiseIndexer();
                      var toDoc = Math.Min(d + innerLoopSize, dataset.NumDocs);
                      for (int doc = d; doc < toDoc; doc++)
                          outputs[doc] = GetOutput(featureBins[doc], prefix);
                  };
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
        }

        public string ToGainSummary(FeaturesToContentMap fmap, Dictionary<int, int> featureToID, int prefix, bool includeZeroGainFeatures, bool normalize, int startingCommentNumber)
        {
            if (_trees.Count == 0)
                return string.Empty;

            StringBuilder output = new StringBuilder();

            // use only first prefix trees
            if (prefix > _trees.Count || prefix < 0)
                prefix = _trees.Count;
            FeatureToGainMap gainMap = new FeatureToGainMap(_trees.Take(prefix).ToList(), normalize);

            if (includeZeroGainFeatures)
            {
                for (int ifeat = 0; ifeat < fmap.Count; ++ifeat)
                    gainMap[ifeat++] += 0.0;
            }

            var sortedByGain = gainMap.OrderByDescending(pair => pair.Value).AsEnumerable();
            var maxValue = sortedByGain.First().Value;
            double normalizingFactor = normalize && maxValue != 0 ? Math.Sqrt(maxValue) : 1.0;
            double power = normalize ? 0.5 : 1.0;
            foreach (var pair in sortedByGain)
            {
                int outputInputId = featureToID.ContainsKey(pair.Key) ? featureToID[pair.Key] : 0;
                output.Append(string.Format("C:{0}=FG:I{1}:{2}:{3}\n", startingCommentNumber++, outputInputId,
                    fmap.GetName(pair.Key), Math.Pow(pair.Value, power) / normalizingFactor));
            }
            return output.ToString();
        }

        /// <summary>
        /// Returns a vector of feature contributions for a given example.
        /// <paramref name="builder"/> is used as a buffer to accumulate the contributions across trees. 
        /// If <paramref name="builder"/> is null, it will be created, otherwise it will be reused.
        /// </summary>
        internal void GetFeatureContributions(ref VBuffer<float> features, ref VBuffer<float> contribs, ref BufferBuilder<float> builder)
        {
            // The feature contributions are equal to the sum of per-tree contributions.

            // REVIEW: it might make sense to accumulate as doubles.
            if (builder == null)
                builder = new BufferBuilder<float>(R4Adder.Instance);
            builder.Reset(features.Length, false);

            foreach (var tree in _trees)
                tree.AppendFeatureContributions(ref features, builder);

            builder.GetResult(ref contribs);
        }
    }

    public class FeatureToGainMap : Dictionary<int, double>
    {
        public FeatureToGainMap() { }
        // Override default Dictionary to return 0.0 for non-eisting keys
        public new double this[int key] {
            get {
                TryGetValue(key, out double retval);
                return retval;
            }
            set {
                base[key] = value;
            }
        }

        public FeatureToGainMap(IList<RegressionTree> trees, bool normalize)
        {
            if (trees.Count == 0)
                return;

            IList<int> combinedKeys = null;
            for (int iteration = 0; iteration < trees.Count; iteration++)
            {
                FeatureToGainMap currentGains = trees[iteration].GainMap;
                combinedKeys = Keys.Union(currentGains.Keys).Distinct().ToList();
                foreach (var k in combinedKeys)
                    this[k] += currentGains[k];
            }
            if (normalize)
            {
                foreach (var k in combinedKeys)
                    this[k] = this[k] / trees.Count;
            }
        }
    }

    /// <summary>
    /// A class that given either a <see cref="RoleMappedSchema"/>
    /// provides a mechanism for getting the corresponding input INI content for the features.
    /// </summary>
    public sealed class FeaturesToContentMap
    {
        private readonly VBuffer<DvText> _content;
        private readonly VBuffer<DvText> _names;

        public int Count => _names.Length;

        /// <summary>
        /// Maps input features names to their input INI content based on the metadata of the
        /// features column. If the <c>IniContent</c> slotwise string metadata is present, that
        /// is used, or else default content is derived from the slot names.
        /// </summary>
        /// <seealso cref="MetadataUtils.Kinds.SlotNames"/>
        public FeaturesToContentMap(RoleMappedSchema schema)
        {
            Contracts.AssertValue(schema);
            var feat = schema.Feature;
            Contracts.AssertValue(feat);
            Contracts.Assert(feat.Type.ValueCount > 0);

            var sch = schema.Schema;
            if (sch.HasSlotNames(feat.Index, feat.Type.ValueCount))
                sch.GetMetadata(MetadataUtils.Kinds.SlotNames, feat.Index, ref _names);
            else
                _names = VBufferUtils.CreateEmpty<DvText>(feat.Type.ValueCount);
#if !CORECLR
            var type = sch.GetMetadataTypeOrNull(BingBinLoader.IniContentMetadataKind, feat.Index);
            if (type != null && type.IsVector && type.VectorSize == feat.Type.ValueCount && type.ItemType.IsText)
                sch.GetMetadata(BingBinLoader.IniContentMetadataKind, feat.Index, ref _content);
            else
                _content = VBufferUtils.CreateEmpty<DvText>(feat.Type.ValueCount);
#else
                _content = VBufferUtils.CreateEmpty<DvText>(feat.Type.ValueCount);
#endif
            Contracts.Assert(_names.Length == _content.Length);
        }

        public string GetName(int ifeat)
        {
            Contracts.Assert(0 <= ifeat && ifeat < Count);
            DvText name = _names.GetItemOrDefault(ifeat);
            return name.HasChars ? name.ToString() : string.Format("f{0}", ifeat);
        }

        public string GetContent(int ifeat)
        {
            Contracts.Assert(0 <= ifeat && ifeat < Count);
            DvText content = _content.GetItemOrDefault(ifeat);
            return content.HasChars ? content.ToString() : DatasetUtils.GetDefaultTransform(GetName(ifeat));
        }
    }
}
