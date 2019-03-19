// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(GamModelParametersBase.VisualizationCommand), typeof(GamModelParametersBase.VisualizationCommand.Arguments), typeof(SignatureCommand),
    "GAM Vizualization Command", GamModelParametersBase.VisualizationCommand.LoadName, "gamviz", DocName = "command/GamViz.md")]

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// The base class for GAM Model Parameters.
    /// </summary>
    public abstract class GamModelParametersBase : ModelParametersBase<float>, IValueMapper, ICalculateFeatureContribution,
        IFeatureContributionMapper, ICanSaveInTextFormat, ICanSaveSummary, ICanSaveInIniFormat
    {
        /// <summary>
        /// The model intercept. Also known as bias or mean effect.
        /// </summary>
        public readonly double Bias;
        /// <summary>
        /// The number of shape functions used in the model.
        /// </summary>
        public readonly int NumberOfShapeFunctions;

        private readonly double[][] _binUpperBounds;
        private readonly double[][] _binEffects;
        private readonly VectorType _inputType;
        private readonly DataViewType _outputType;
        // These would be the bins for a totally sparse input.
        private readonly int[] _binsAtAllZero;
        // The output value for all zeros
        private readonly double _valueAtAllZero;
        private readonly int[] _shapeToInputMap;
        private readonly int _numInputFeatures;
        private readonly Dictionary<int, int> _inputFeatureToShapeFunctionMap;

        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => _outputType;

        /// <summary>
        /// Used to determine the contribution of each feature to the score of an example by <see cref="FeatureContributionCalculatingTransformer"/>.
        /// For Generalized Additive Models (GAM), the contribution of a feature is equal to the shape function for the given feature evaluated at
        /// the feature value.
        /// </summary>
        FeatureContributionCalculator ICalculateFeatureContribution.FeatureContributionCalculator => new FeatureContributionCalculator(this);

        private protected GamModelParametersBase(IHostEnvironment env, string name,
            double[][] binUpperBounds, double[][] binEffects, double intercept, int numInputFeatures = -1, int[] shapeToInputMap = null)
            : base(env, name)
        {
            Host.CheckValue(binEffects, nameof(binEffects), "May not be null.");
            Host.CheckValue(binUpperBounds, nameof(binUpperBounds), "May not be null.");
            Host.CheckParam(binUpperBounds.Length == binEffects.Length, nameof(binUpperBounds), "Must have same number of features as binEffects");
            Host.CheckParam(binEffects.Length > 0, nameof(binEffects), "Must have at least one entry");
            Host.CheckParam(numInputFeatures == -1 || numInputFeatures > 0, nameof(numInputFeatures), "Must be greater than zero");
            Host.CheckParam(shapeToInputMap == null || shapeToInputMap.Length == binEffects.Length, nameof(shapeToInputMap), "Must have same number of features as binEffects");

            // Define the model basics
            Bias = intercept;
            _binUpperBounds = binUpperBounds;
            _binEffects = binEffects;
            NumberOfShapeFunctions = binEffects.Length;

            // For sparse inputs we have a fast lookup
            _binsAtAllZero = new int[NumberOfShapeFunctions];
            _valueAtAllZero = 0;

            // Walk through each feature and perform checks / updates
            for (int i = 0; i < NumberOfShapeFunctions; i++)
            {
                // Check data validity
                Host.CheckValue(binEffects[i], nameof(binEffects), "Array contained null entries");
                Host.CheckParam(binUpperBounds[i].Length == binEffects[i].Length, nameof(binEffects), "Array contained wrong number of effect values");
                Host.CheckParam(Utils.IsMonotonicallyIncreasing(binUpperBounds[i]), nameof(binUpperBounds), "Array must be monotonically increasing");

                // Update the value at zero
                _valueAtAllZero += GetBinEffect(i, 0, out _binsAtAllZero[i]);
            }

            // Define the sparse mappings from/to input to/from shape functions
            _shapeToInputMap = shapeToInputMap;
            if (_shapeToInputMap == null)
                _shapeToInputMap = Utils.GetIdentityPermutation(NumberOfShapeFunctions);

            _numInputFeatures = numInputFeatures;
            if (_numInputFeatures == -1)
                _numInputFeatures = NumberOfShapeFunctions;
            _inputFeatureToShapeFunctionMap = new Dictionary<int, int>(_shapeToInputMap.Length);
            for (int i = 0; i < _shapeToInputMap.Length; i++)
            {
                Host.CheckParam(0 <= _shapeToInputMap[i] && _shapeToInputMap[i] < _numInputFeatures, nameof(_shapeToInputMap), "Contains out of range feature value");
                Host.CheckParam(!_inputFeatureToShapeFunctionMap.ContainsValue(_shapeToInputMap[i]), nameof(_shapeToInputMap), "Contains duplicate mappings");
                _inputFeatureToShapeFunctionMap[_shapeToInputMap[i]] = i;
            }

            _inputType = new VectorType(NumberDataViewType.Single, _numInputFeatures);
            _outputType = NumberDataViewType.Single;
        }

        private protected GamModelParametersBase(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name)
        {
            Host.CheckValue(ctx, nameof(ctx));

            BinaryReader reader = ctx.Reader;

            NumberOfShapeFunctions = reader.ReadInt32();
            Host.CheckDecode(NumberOfShapeFunctions >= 0);
            _numInputFeatures = reader.ReadInt32();
            Host.CheckDecode(_numInputFeatures >= 0);
            Bias = reader.ReadDouble();
            if (ctx.Header.ModelVerWritten == 0x00010001)
                using (var ch = env.Start("GamWarningChannel"))
                    ch.Warning("GAMs models written prior to ML.NET 0.6 are loaded with an incorrect Intercept. For these models, subtract the value of the intercept from the prediction.");

            _binEffects = new double[NumberOfShapeFunctions][];
            _binUpperBounds = new double[NumberOfShapeFunctions][];
            _binsAtAllZero = new int[NumberOfShapeFunctions];
            for (int i = 0; i < NumberOfShapeFunctions; i++)
            {
                _binEffects[i] = reader.ReadDoubleArray();
                Host.CheckDecode(Utils.Size(_binEffects[i]) >= 1);
            }
            for (int i = 0; i < NumberOfShapeFunctions; i++)
            {
                _binUpperBounds[i] = reader.ReadDoubleArray(_binEffects[i].Length);
                _valueAtAllZero += GetBinEffect(i, 0, out _binsAtAllZero[i]);
            }
            int len = reader.ReadInt32();
            Host.CheckDecode(len >= 0);

            _inputFeatureToShapeFunctionMap = new Dictionary<int, int>(len);
            _shapeToInputMap = Utils.CreateArray(NumberOfShapeFunctions, -1);
            for (int i = 0; i < len; i++)
            {
                int key = reader.ReadInt32();
                Host.CheckDecode(0 <= key && key < _numInputFeatures);
                int val = reader.ReadInt32();
                Host.CheckDecode(0 <= val && val < NumberOfShapeFunctions);
                Host.CheckDecode(!_inputFeatureToShapeFunctionMap.ContainsKey(key));
                Host.CheckDecode(_shapeToInputMap[val] == -1);
                _inputFeatureToShapeFunctionMap[key] = val;
                _shapeToInputMap[val] = key;
            }

            _inputType = new VectorType(NumberDataViewType.Single, _numInputFeatures);
            _outputType = NumberDataViewType.Single;
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.Writer.Write(NumberOfShapeFunctions);
            Host.Assert(NumberOfShapeFunctions >= 0);
            ctx.Writer.Write(_numInputFeatures);
            Host.Assert(_numInputFeatures >= 0);
            ctx.Writer.Write(Bias);
            for (int i = 0; i < NumberOfShapeFunctions; i++)
                ctx.Writer.WriteDoubleArray(_binEffects[i]);
            int diff = _binEffects.Sum(e => e.Take(e.Length - 1).Select((ef, i) => ef != e[i + 1] ? 1 : 0).Sum());
            int bound = _binEffects.Sum(e => e.Length - 1);

            for (int i = 0; i < NumberOfShapeFunctions; i++)
            {
                ctx.Writer.WriteDoublesNoCount(_binUpperBounds[i]);
                Host.Assert(_binUpperBounds[i].Length == _binEffects[i].Length);
            }
            ctx.Writer.Write(_inputFeatureToShapeFunctionMap.Count);
            foreach (KeyValuePair<int, int> kvp in _inputFeatureToShapeFunctionMap)
            {
                ctx.Writer.Write(kvp.Key);
                ctx.Writer.Write(kvp.Value);
            }
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>), "Input type does not match.");
            Host.Check(typeof(TOut) == typeof(float), "Output type does not match.");

            ValueMapper<VBuffer<float>, float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private void Map(in VBuffer<float> features, ref float response)
        {
            Host.CheckParam(features.Length == _numInputFeatures, nameof(features), "Bad length of input");

            double value = Bias;
            var featuresValues = features.GetValues();

            if (features.IsDense)
            {
                for (int i = 0; i < featuresValues.Length; ++i)
                {
                    if (_inputFeatureToShapeFunctionMap.TryGetValue(i, out int j))
                        value += GetBinEffect(j, featuresValues[i]);
                }
            }
            else
            {
                var featuresIndices = features.GetIndices();
                // Add in the precomputed results for all features
                value += _valueAtAllZero;
                for (int i = 0; i < featuresValues.Length; ++i)
                {
                    if (_inputFeatureToShapeFunctionMap.TryGetValue(featuresIndices[i], out int j))
                        // Add the value and subtract the value at zero that was previously accounted for
                        value += GetBinEffect(j, featuresValues[i]) - GetBinEffect(j, 0);
                }
            }

            response = (float)value;
        }

        internal double GetFeatureBinsAndScore(in VBuffer<float> features, int[] bins)
        {
            Host.CheckParam(features.Length == _numInputFeatures, nameof(features));
            Host.CheckParam(Utils.Size(bins) == NumberOfShapeFunctions, nameof(bins));

            double value = Bias;
            var featuresValues = features.GetValues();
            if (features.IsDense)
            {
                for (int i = 0; i < featuresValues.Length; ++i)
                {
                    if (_inputFeatureToShapeFunctionMap.TryGetValue(i, out int j))
                        value += GetBinEffect(j, featuresValues[i], out bins[j]);
                }
            }
            else
            {
                var featuresIndices = features.GetIndices();
                // Add in the precomputed results for all features
                value += _valueAtAllZero;
                Array.Copy(_binsAtAllZero, bins, NumberOfShapeFunctions);

                // Update the results for features we have
                for (int i = 0; i < featuresValues.Length; ++i)
                {
                    if (_inputFeatureToShapeFunctionMap.TryGetValue(featuresIndices[i], out int j))
                        // Add the value and subtract the value at zero that was previously accounted for
                        value += GetBinEffect(j, featuresValues[i], out bins[j]) - GetBinEffect(j, 0);
                }
            }
            return value;
        }

        private double GetBinEffect(int featureIndex, double featureValue)
        {
            Host.Assert(0 <= featureIndex && featureIndex < NumberOfShapeFunctions, "Index out of range.");
            int index = Algorithms.FindFirstGE(_binUpperBounds[featureIndex], featureValue);
            return _binEffects[featureIndex][index];
        }

        private double GetBinEffect(int featureIndex, double featureValue, out int binIndex)
        {
            Host.Check(0 <= featureIndex && featureIndex < NumberOfShapeFunctions, "Index out of range.");
            binIndex = Algorithms.FindFirstGE(_binUpperBounds[featureIndex], featureValue);
            return _binEffects[featureIndex][binIndex];
        }

        /// <summary>
        /// Get the bin upper bounds for each feature.
        /// </summary>
        /// <param name="featureIndex">The index of the feature (in the training vector) to get.</param>
        /// <returns>The bin upper bounds. May be zero length if this feature has no bins.</returns>
        public IReadOnlyList<double> GetBinUpperBounds(int featureIndex)
        {
            Host.Check(0 <= featureIndex && featureIndex < NumberOfShapeFunctions, "Index out of range.");
            if (!_inputFeatureToShapeFunctionMap.TryGetValue(featureIndex, out int j))
                return new double[0];

            var binUpperBounds = new double[_binUpperBounds[j].Length];
            _binUpperBounds[j].CopyTo(binUpperBounds, 0);
            return binUpperBounds;
        }

        /// <summary>
        /// Get all the bin upper bounds.
        /// </summary>
        [BestFriend]
        internal double[][] GetBinUpperBounds()
        {
            double[][] binUpperBounds = new double[NumberOfShapeFunctions][];
            for (int i = 0; i < NumberOfShapeFunctions; i++)
            {
                if (_inputFeatureToShapeFunctionMap.TryGetValue(i, out int j))
                {
                    binUpperBounds[i] = new double[_binUpperBounds[j].Length];
                    _binUpperBounds[j].CopyTo(binUpperBounds[i], 0);
                }
                else
                {
                    binUpperBounds[i] = new double[0];
                }
            }
            return binUpperBounds;
        }

        /// <summary>
        /// Get the binned weights for each feature.
        /// </summary>
        /// <param name="featureIndex">The index of the feature (in the training vector) to get.</param>
        /// <returns>The binned effects for each feature. May be zero length if this feature has no bins.</returns>
        public IReadOnlyList<double> GetBinEffects(int featureIndex)
        {
            Host.Check(0 <= featureIndex && featureIndex < NumberOfShapeFunctions, "Index out of range.");
            if (!_inputFeatureToShapeFunctionMap.TryGetValue(featureIndex, out int j))
                return new double[0];

            var binEffects = new double[_binEffects[j].Length];
            _binEffects[j].CopyTo(binEffects, 0);
            return binEffects;
        }

        /// <summary>
        /// Get all the binned effects.
        /// </summary>
        [BestFriend]
        internal double[][] GetBinEffects()
        {
            double[][] binEffects = new double[NumberOfShapeFunctions][];
            for (int i = 0; i < NumberOfShapeFunctions; i++)
            {
                if (_inputFeatureToShapeFunctionMap.TryGetValue(i, out int j))
                {
                    binEffects[i] = new double[_binEffects[j].Length];
                    _binEffects[j].CopyTo(binEffects[i], 0);
                }
                else
                {
                    binEffects[i] = new double[0];
                }
            }
            return binEffects;
        }

        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer), "writer must not be null.");
            Host.CheckValueOrNull(schema);

            writer.WriteLine("\xfeffFeature index table"); // add BOM to tell excel this is UTF-8
            writer.WriteLine($"Number of features:\t{NumberOfShapeFunctions + 1:D}");
            writer.WriteLine("Feature Index\tFeature Name");

            // REVIEW: We really need some unit tests around text exporting (for this, and other learners).
            // A useful test in this case would be a model trained with:
            // maml.exe train data=Samples\breast-cancer-withheader.txt loader=text{header+ col=Label:0 col=F1:1-4 col=F2:4 col=F3:5-*}
            //    xf =expr{col=F2 expr=x:0.0} xf=concat{col=Features:F1,F2,F3} tr=gam out=bubba2.zip
            // Write out the intercept
            writer.WriteLine("-1\tIntercept");

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            AnnotationUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, _numInputFeatures, ref names);

            for (int internalIndex = 0; internalIndex < NumberOfShapeFunctions; internalIndex++)
            {
                int featureIndex = _shapeToInputMap[internalIndex];
                var name = names.GetItemOrDefault(featureIndex);
                writer.WriteLine(!name.IsEmpty ? "{0}\t{1}" : "{0}\tFeature {0}", featureIndex, name);
            }

            writer.WriteLine();
            writer.WriteLine("Per feature binned effects:");
            writer.WriteLine("Feature Index\tFeature Value Bin Upper Bound\tOutput (effect on label)");
            writer.WriteLine($"{-1:D}\t{float.MaxValue:R}\t{Bias:R}");
            for (int internalIndex = 0; internalIndex < NumberOfShapeFunctions; internalIndex++)
            {
                int featureIndex = _shapeToInputMap[internalIndex];

                double[] effects = _binEffects[internalIndex];
                double[] boundaries = _binUpperBounds[internalIndex];
                for (int i = 0; i < effects.Length; ++i)
                    writer.WriteLine($"{featureIndex:D}\t{boundaries[i]:R}\t{effects[i]:R}");
            }
        }

        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            ((ICanSaveInTextFormat)this).SaveAsText(writer, schema);
        }

        ValueMapper<TSrc, VBuffer<float>> IFeatureContributionMapper.GetFeatureContributionMapper<TSrc, TDstContributions>
            (int top, int bottom, bool normalize)
        {
            Host.Check(typeof(TSrc) == typeof(VBuffer<float>), "Source type does not match.");
            Host.Check(typeof(TDstContributions) == typeof(VBuffer<float>), "Destination type does not match.");

            ValueMapper<VBuffer<float>, VBuffer<float>> del =
                (in VBuffer<float> srcFeatures, ref VBuffer<float> dstContributions) =>
                {
                    GetFeatureContributions(in srcFeatures, ref dstContributions, top, bottom, normalize);
                };
            return (ValueMapper<TSrc, VBuffer<float>>)(Delegate)del;
        }

        private void GetFeatureContributions(in VBuffer<float> features, ref VBuffer<float> contributions,
                        int top, int bottom, bool normalize)
        {
            var editor = VBufferEditor.Create(ref contributions, features.Length);

            // We need to use dense value of features, b/c the feature contributions could be significant
            // even for features with value 0.
            var featureIndex = 0;
            foreach (var featureValue in features.DenseValues())
            {
                float contribution = 0;
                if (_inputFeatureToShapeFunctionMap.TryGetValue(featureIndex, out int j))
                    contribution = (float)GetBinEffect(j, featureValue);
                editor.Values[featureIndex] = contribution;
                featureIndex++;
            }
            contributions = editor.Commit();
            Numeric.VectorUtils.SparsifyNormalize(ref contributions, top, bottom, normalize);
        }

        void ICanSaveInIniFormat.SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator)
        {
            Host.CheckValue(writer, nameof(writer), "writer must not be null");
            var ensemble = new InternalTreeEnsemble();

            for (int featureIndex = 0; featureIndex < NumberOfShapeFunctions; featureIndex++)
            {
                var effects = _binEffects[featureIndex];
                var binThresholds = _binUpperBounds[featureIndex];

                Host.Check(effects.Length == binThresholds.Length, "Effects array must be same length as binUpperBounds array.");
                var numLeaves = effects.Length;
                var numInternalNodes = numLeaves - 1;

                var splitFeatures = Enumerable.Repeat(featureIndex, numInternalNodes).ToArray();
                var (treeThresholds, lteChild, gtChild) = CreateBalancedTree(numInternalNodes, binThresholds);
                var tree = CreateRegressionTree(numLeaves, splitFeatures, treeThresholds, lteChild, gtChild, effects);
                ensemble.AddTree(tree);
            }

            // Adding the intercept as a dummy tree with the output values being the model intercept,
            // works for reaching parity.
            var interceptTree = CreateRegressionTree(
                numLeaves: 2,
                splitFeatures: new[] { 0 },
                rawThresholds: new[] { 0f },
                lteChild: new[] { ~0 },
                gtChild: new[] { ~1 },
                leafValues: new[] { Bias, Bias });
            ensemble.AddTree(interceptTree);

            var ini = FastTreeIniFileUtils.TreeEnsembleToIni(
                Host, ensemble, schema, calibrator, string.Empty, false, false);

            // Remove the SplitGain values which are all 0.
            // It's eaiser to remove them here, than to modify the FastTree code.
            var goodLines = ini.Split(new[] { '\n' }).Where(line => !line.StartsWith("SplitGain="));
            ini = string.Join("\n", goodLines);
            writer.WriteLine(ini);
        }

        // GAM bins should be converted to balanced trees / binary search trees
        // so that scoring takes O(log(n)) instead of O(n). The following utility
        // creates a balanced tree.
        private (float[], int[], int[]) CreateBalancedTree(int numInternalNodes, double[] binThresholds)
        {
            var binIndices = Enumerable.Range(0, numInternalNodes).ToArray();
            var internalNodeIndices = new List<int>();
            var lteChild = new List<int>();
            var gtChild = new List<int>();
            var internalNodeId = numInternalNodes;

            CreateBalancedTreeRecursive(
                0, binIndices.Length - 1, internalNodeIndices, lteChild, gtChild, ref internalNodeId);
            // internalNodeId should have been counted all the way down to 0 (root node)
            Host.Assert(internalNodeId == 0);

            var tree = (
                thresholds: internalNodeIndices.Select(x => (float)binThresholds[binIndices[x]]).ToArray(),
                lteChild: lteChild.ToArray(),
                gtChild: gtChild.ToArray());
            return tree;
        }

        private int CreateBalancedTreeRecursive(int lower, int upper,
            List<int> internalNodeIndices, List<int> lteChild, List<int> gtChild, ref int internalNodeId)
        {
            if (lower > upper)
            {
                // Base case: we've reached a leaf node
                Host.Assert(lower == upper + 1);
                return ~lower;
            }
            else
            {
                // This is postorder traversal algorithm and populating the internalNodeIndices/lte/gt lists in reverse.
                // Preorder is the only option, because we need the results of both left/right recursions for populating the lists.
                // As a result, lists are populated in reverse, because the root node should be the first item on the lists.
                // Binary search tree algorithm (recursive splitting to half) is used for creating balanced tree.
                var mid = (lower + upper) / 2;
                var left = CreateBalancedTreeRecursive(
                    lower, mid - 1, internalNodeIndices, lteChild, gtChild, ref internalNodeId);
                var right = CreateBalancedTreeRecursive(
                    mid + 1, upper, internalNodeIndices, lteChild, gtChild, ref internalNodeId);
                internalNodeIndices.Insert(0, mid);
                lteChild.Insert(0, left);
                gtChild.Insert(0, right);
                return --internalNodeId;
            }
        }

        private static InternalRegressionTree CreateRegressionTree(
            int numLeaves, int[] splitFeatures, float[] rawThresholds, int[] lteChild, int[] gtChild, double[] leafValues)
        {
            var numInternalNodes = numLeaves - 1;
            return InternalRegressionTree.Create(
                numLeaves: numLeaves,
                splitFeatures: splitFeatures,
                rawThresholds: rawThresholds,
                lteChild: lteChild,
                gtChild: gtChild.ToArray(),
                leafValues: leafValues,
                // Ignored arguments
                splitGain: new double[numInternalNodes],
                defaultValueForMissing: new float[numInternalNodes],
                categoricalSplitFeatures: new int[numInternalNodes][],
                categoricalSplit: new bool[numInternalNodes]);
        }

        /// <summary>
        /// The GAM model visualization command. Because the data access commands must access private members of
        /// <see cref="GamModelParametersBase"/>, it is convenient to have the command itself nested within the base
        /// predictor class.
        /// </summary>
        internal sealed class VisualizationCommand : DataCommand.ImplBase<VisualizationCommand.Arguments>
        {
            public const string Summary = "Loads a model trained with a GAM learner, and starts an interactive web session to visualize it.";
            public const string LoadName = "GamVisualization";

            public sealed class Arguments : DataCommand.ArgumentsBase
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to open the GAM visualization page URL", ShortName = "o", SortOrder = 3)]
                public bool Open = true;

                internal Arguments SetServerIfNeeded(IHostEnvironment env)
                {
                    // We assume that if someone invoked this, they really did mean to start the web server.
                    if (env != null && Server == null)
                        Server = ServerChannel.CreateDefaultServerFactoryOrNull(env);
                    return this;
                }
            }

            private readonly string _inputModelPath;
            private readonly bool _open;

            public VisualizationCommand(IHostEnvironment env, Arguments args)
                : base(env, args.SetServerIfNeeded(env), LoadName)
            {
                Host.CheckValue(args, nameof(args));
                Host.CheckValue(args.Server, nameof(args.Server));
                Host.CheckNonWhiteSpace(args.InputModelFile, nameof(args.InputModelFile));

                _inputModelPath = args.InputModelFile;
                _open = args.Open;
            }

            public override void Run()
            {
                using (var ch = Host.Start("Run"))
                {
                    Run(ch);
                }
            }

            private sealed class Context
            {
                private readonly GamModelParametersBase _pred;
                private readonly RoleMappedData _data;

                private readonly VBuffer<ReadOnlyMemory<char>> _featNames;
                // The scores.
                private readonly float[] _scores;
                // The labels.
                private readonly float[] _labels;
                // For every feature, and for every bin, there is a list of documents with that feature.
                private readonly List<int>[][] _binDocsList;
                // Whenever the predictor is "modified," we up this version. This value is returned for anything
                // that is subject to change, and can be used by client web code to detect whenever something
                // may have happened behind its back.
                private long _version;
                private long _saveVersion;

                // Non-null if this object was created with an evaluator *and* scores and labels is non-empty.
                private readonly RoleMappedData _dataForEvaluator;
                // Non-null in the same conditions that the above is non-null.
                private readonly IEvaluator _eval;

                //the map of categorical indices, as defined in MetadataUtils
                private readonly int[] _catsMap;

                /// <summary>
                /// These are the number of input features, as opposed to the number of features used within GAM
                /// which may be lower.
                /// </summary>
                public int NumFeatures => _pred._inputType.Size;

                public Context(IChannel ch, GamModelParametersBase pred, RoleMappedData data, IEvaluator eval)
                {
                    Contracts.AssertValue(ch);
                    ch.AssertValue(pred);
                    ch.AssertValue(data);
                    ch.AssertValueOrNull(eval);

                    _saveVersion = -1;
                    _pred = pred;
                    _data = data;
                    var schema = _data.Schema;
                    var featCol = schema.Feature.Value;
                    int len = featCol.Type.GetValueCount();
                    ch.Check(len == _pred._numInputFeatures);

                    if (featCol.HasSlotNames(len))
                        featCol.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref _featNames);
                    else
                        _featNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(len);

                    var numFeatures = _pred._binEffects.Length;
                    _binDocsList = new List<int>[numFeatures][];
                    for (int f = 0; f < numFeatures; f++)
                    {
                        var binDocList = new List<int>[_pred._binEffects[f].Length];
                        for (int e = 0; e < _pred._binEffects[f].Length; e++)
                            binDocList[e] = new List<int>();
                        _binDocsList[f] = binDocList;
                    }
                    var labels = new List<float>();
                    var scores = new List<float>();

                    int[] bins = new int[numFeatures];
                    using (var cursor = new FloatLabelCursor(_data, CursOpt.Label | CursOpt.Features))
                    {
                        int doc = 0;
                        while (cursor.MoveNext())
                        {
                            labels.Add(cursor.Label);
                            var score = _pred.GetFeatureBinsAndScore(in cursor.Features, bins);
                            scores.Add((float)score);
                            for (int f = 0; f < numFeatures; f++)
                                _binDocsList[f][bins[f]].Add(doc);
                            ++doc;
                        }

                        _labels = labels.ToArray();
                        labels = null;
                        _scores = scores.ToArray();
                        scores = null;
                    }

                    ch.Assert(_scores.Length == _labels.Length);
                    if (_labels.Length > 0 && eval != null)
                    {
                        _eval = eval;
                        var builder = new ArrayDataViewBuilder(pred.Host);
                        builder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single, _labels);
                        builder.AddColumn(DefaultColumnNames.Score, NumberDataViewType.Single, _scores);
                        _dataForEvaluator = new RoleMappedData(builder.GetDataView(), opt: false,
                            RoleMappedSchema.ColumnRole.Label.Bind(DefaultColumnNames.Label),
                            new RoleMappedSchema.ColumnRole(AnnotationUtils.Const.ScoreValueKind.Score).Bind(DefaultColumnNames.Score));
                    }

                    var featureCol = _data.Schema.Schema[DefaultColumnNames.Features];
                    AnnotationUtils.TryGetCategoricalFeatureIndices(_data.Schema.Schema, featureCol.Index, out _catsMap);
                }

                public FeatureInfo GetInfoForIndex(int index) => FeatureInfo.GetInfoForIndex(this, index);
                public IEnumerable<FeatureInfo> GetInfos() => FeatureInfo.GetInfos(this);

                public long SetEffect(int feat, int bin, double effect)
                {
                    // Another version with multiple effects, perhaps?
                    int internalIndex;
                    if (!_pred._inputFeatureToShapeFunctionMap.TryGetValue(feat, out internalIndex))
                        return -1;
                    var effects = _pred._binEffects[internalIndex];
                    if (bin < 0 || bin > effects.Length)
                        return -1;

                    lock (_pred)
                    {
                        var deltaEffect = effect - effects[bin];
                        effects[bin] = effect;
                        foreach (var docIndex in _binDocsList[internalIndex][bin])
                            _scores[docIndex] += (float)deltaEffect;
                        return checked(++_version);
                    }
                }

                public MetricsInfo GetMetrics()
                {
                    if (_eval == null)
                        return null;

                    lock (_pred)
                    {
                        var metricDict = _eval.Evaluate(_dataForEvaluator);
                        IDataView metricsView;
                        if (!metricDict.TryGetValue(MetricKinds.OverallMetrics, out metricsView))
                            return null;
                        Contracts.AssertValue(metricsView);
                        return new MetricsInfo(_version, EvaluateUtils.GetMetrics(metricsView).ToArray());
                    }
                }

                /// <summary>
                /// This will write out a file, if needed. In all cases if something is written it will return
                /// a version number, with an indication based on sign of whether anything was actually written
                /// in this call.
                /// </summary>
                /// <param name="host">The host from the command</param>
                /// <param name="ch">The channel from the command</param>
                /// <param name="outFile">The (optionally empty) output file</param>
                /// <returns>Returns <c>null</c> if the model file could not be saved because <paramref name="outFile"/>
                /// was <c>null</c> or whitespace. Otherwise, if the current version if newer than the last version saved,
                /// it will save, and return that version. (In this case, the number is non-negative.) Otherwise, if the current
                /// version was the last version saved, then it will return the bitwise not of that version number (in this case,
                /// the number is negative).</returns>
                public long? SaveIfNeeded(IHost host, IChannel ch, string outFile)
                {
                    Contracts.AssertValue(ch);
                    ch.AssertValue(host);
                    ch.AssertValueOrNull(outFile);

                    if (string.IsNullOrWhiteSpace(outFile))
                        return null;

                    lock (_pred)
                    {
                        ch.Assert(_saveVersion <= _version);
                        if (_saveVersion == _version)
                            return ~_version;

                        // Note that this data pipe is the data pipe that was defined for the gam visualization
                        // command, which may not be quite the same thing as the data pipe in the original model,
                        // in the event that the user specified different loader settings, defined new transforms,
                        // etc.
                        using (var file = host.CreateOutputFile(outFile))
                            TrainUtils.SaveModel(host, ch, file, _pred, _data);
                        return _saveVersion = _version;
                    }
                }

                public sealed class MetricsInfo
                {
                    public long Version { get; }
                    public KeyValuePair<string, double>[] Metrics { get; }

                    public MetricsInfo(long version, KeyValuePair<string, double>[] metrics)
                    {
                        Version = version;
                        Metrics = metrics;
                    }
                }

                public sealed class FeatureInfo
                {
                    public int Index { get; }
                    public string Name { get; }

                    /// <summary>
                    /// The upper bounds of each bin.
                    /// </summary>
                    public IEnumerable<double> UpperBounds { get; }

                    /// <summary>
                    /// The amount added to the model for a document falling in a given bin.
                    /// </summary>
                    public IEnumerable<double> BinEffects { get; }

                    /// <summary>
                    /// The number of documents in each bin.
                    /// </summary>
                    public IEnumerable<int> DocCounts { get; }

                    /// <summary>
                    /// The version of the GAM context that has these values.
                    /// </summary>
                    public long Version { get; }

                    /// <summary>
                    /// For features belonging to the same categorical, this value will be the same,
                    /// Set to -1 for non-categoricals.
                    /// </summary>
                    public int CategoricalFeatureIndex { get; }

                    private FeatureInfo(Context context, int index, int internalIndex, int[] catsMap)
                    {
                        Contracts.AssertValue(context);
                        Contracts.Assert(context._pred._inputFeatureToShapeFunctionMap.ContainsKey(index)
                            && context._pred._inputFeatureToShapeFunctionMap[index] == internalIndex);
                        Index = index;
                        var name = context._featNames.GetItemOrDefault(index).ToString();
                        Name = string.IsNullOrEmpty(name) ? $"f{index}" : name;
                        var up = context._pred._binUpperBounds[internalIndex];
                        UpperBounds = up.Take(up.Length - 1);
                        BinEffects = context._pred._binEffects[internalIndex];
                        DocCounts = context._binDocsList[internalIndex].Select(Utils.Size);
                        Version = context._version;
                        CategoricalFeatureIndex = -1;

                        if (catsMap != null && index < catsMap[catsMap.Length - 1])
                        {
                            for (int i = 0; i < catsMap.Length; i += 2)
                            {
                                if (index >= catsMap[i] && index <= catsMap[i + 1])
                                {
                                    CategoricalFeatureIndex = i;
                                    break;
                                }
                            }
                        }
                    }

                    public static FeatureInfo GetInfoForIndex(Context context, int index)
                    {
                        Contracts.AssertValue(context);
                        Contracts.Assert(0 <= index && index < context._pred._inputType.Size);
                        lock (context._pred)
                        {
                            int internalIndex;
                            if (!context._pred._inputFeatureToShapeFunctionMap.TryGetValue(index, out internalIndex))
                                return null;
                            return new FeatureInfo(context, index, internalIndex, context._catsMap);
                        }
                    }

                    public static FeatureInfo[] GetInfos(Context context)
                    {
                        lock (context._pred)
                        {
                            return Utils.BuildArray(context._pred.NumberOfShapeFunctions,
                                i => new FeatureInfo(context, context._pred._shapeToInputMap[i], i, context._catsMap));
                        }
                    }
                }
            }

            /// <summary>
            /// Attempts to initialize required items, from the input model file. It could throw if something goes wrong.
            /// </summary>
            /// <param name="ch">The channel</param>
            /// <returns>A structure containing essential information about the GAM dataset that enables
            /// operations on top of that structure.</returns>
            private Context Init(IChannel ch)
            {
                ILegacyDataLoader loader;
                IPredictor rawPred;
                RoleMappedSchema schema;
                LoadModelObjects(ch, true, out rawPred, true, out schema, out loader);
                bool hadCalibrator = false;

                // The rawPred has two possible types:
                //  1. CalibratedPredictorBase<BinaryClassificationGamModelParameters, PlattCalibrator>
                //  2. RegressionGamModelParameters
                // For (1), the trained model, GamModelParametersBase, is a field we need to extract. For (2),
                // we don't need to do anything because RegressionGamModelParameters is derived from GamModelParametersBase.
                var calibrated = rawPred as CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>;
                while (calibrated != null)
                {
                    hadCalibrator = true;
                    rawPred = calibrated.SubModel;
                    calibrated = rawPred as CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>;
                }
                var pred = rawPred as GamModelParametersBase;
                ch.CheckUserArg(pred != null, nameof(ImplOptions.InputModelFile), "Predictor was not a " + nameof(GamModelParametersBase));
                var data = new RoleMappedData(loader, schema.GetColumnRoleNames(), opt: true);
                if (hadCalibrator && !string.IsNullOrWhiteSpace(ImplOptions.OutputModelFile))
                    ch.Warning("If you save the GAM model, only the GAM model, not the wrapping calibrator, will be saved.");

                return new Context(ch, pred, data, InitEvaluator(pred));
            }

            private IEvaluator InitEvaluator(GamModelParametersBase pred)
            {
                switch (pred.PredictionKind)
                {
                    case PredictionKind.BinaryClassification:
                        return new BinaryClassifierEvaluator(Host, new BinaryClassifierEvaluator.Arguments());
                    case PredictionKind.Regression:
                        return new RegressionEvaluator(Host, new RegressionEvaluator.Arguments());
                    default:
                        return null;
                }
            }

            private void Run(IChannel ch)
            {
                // First we're going to initialize a structure with lots of information about the predictor, trainer, etc.
                var context = Init(ch);

                // REVIEW: What to do with the data? Not sure. Take a sample? We could have
                // a very compressed one, since we can just "bin" everything based on pred._binUpperBounds. Anyway
                // whatever we choose to do, ultimately it will be exposed as some delegate on the server channel.
                // Maybe binning actually isn't wise, *if* we want people to be able to set their own split points
                // (which seems plausible). In the current version of the viz you can only set bin effects, but
                // "splitting" a bin might be desirable in some cases, maybe. Or not.

                // Now we have a gam predictor,
                AutoResetEvent ev = new AutoResetEvent(false);
                using (var server = InitServer(ch))
                using (var sch = Host.StartServerChannel("predictor/gam"))
                {
                    // The number of features.
                    sch?.Register("numFeatures", () => context.NumFeatures);
                    // Info for a particular feature.
                    sch?.Register<int, Context.FeatureInfo>("info", context.GetInfoForIndex);
                    // Info for all features.
                    sch?.Register("infos", context.GetInfos);
                    // Modification of the model.
                    sch?.Register<int, int, double, long>("setEffect", context.SetEffect);
                    // Getting the metrics.
                    sch?.Register("metrics", context.GetMetrics);
                    sch?.Register("canSave", () => !string.IsNullOrEmpty(ImplOptions.OutputModelFile));
                    sch?.Register("save", () => context.SaveIfNeeded(Host, ch, ImplOptions.OutputModelFile));
                    sch?.Register("quit", () =>
                    {
                        var retVal = context.SaveIfNeeded(Host, ch, ImplOptions.OutputModelFile);
                        ev.Set();
                        return retVal;
                    });

                    // Targets and scores for data.
                    sch?.Publish();

                    if (sch != null)
                    {
                        ch.Info("GAM viz server is ready and waiting.");
                        Uri uri = server.BaseAddress;
                        // Believe it or not, this is actually the recommended procedure according to MSDN.
                        if (_open)
                            System.Diagnostics.Process.Start(uri.AbsoluteUri + "content/GamViz/");
                        ev.WaitOne();
                        ch.Info("Quit signal received. Quitter.");
                    }
                    else
                        ch.Info("No server, exiting immediately.");
                }
            }
        }
    }
}
