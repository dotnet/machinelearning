// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.LightGBM;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Training;

[assembly: LoadableClass(LightGbmMulticlassTrainer.Summary, typeof(LightGbmMulticlassTrainer), typeof(Options),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    "LightGBM Multi-class Classifier", LightGbmMulticlassTrainer.LoadNameValue, LightGbmMulticlassTrainer.ShortName, DocName = "trainer/LightGBM.md")]

namespace Microsoft.ML.LightGBM
{

    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM"]/*' />
    public sealed class LightGbmMulticlassTrainer : LightGbmTrainerBase<VBuffer<float>, MulticlassPredictionTransformer<OvaModelParameters>, OvaModelParameters>
    {
        internal const string Summary = "LightGBM Multi Class Classifier";
        internal const string LoadNameValue = "LightGBMMulticlass";
        internal const string ShortName = "LightGBMMC";
        private const int _minDataToUseSoftmax = 50000;

        private const double _maxNumClass = 1e6;
        private int _numClass;
        private int _tlcNumClass;
        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        internal LightGbmMulticlassTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumn))
        {
            _numClass = -1;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmMulticlassTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of The label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The name for the column containing the initial weight.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        internal LightGbmMulticlassTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGBM.Options.Defaults.NumBoostRound)
            : base(env, LoadNameValue, TrainerUtils.MakeU4ScalarColumn(labelColumn), featureColumn, weights, null, numLeaves, minDataPerLeaf, learningRate, numBoostRound)
        {
            _numClass = -1;
        }

        private InternalTreeEnsemble GetBinaryEnsemble(int classID)
        {
            var res = new InternalTreeEnsemble();
            for (int i = classID; i < TrainedEnsemble.NumTrees; i += _numClass)
            {
                // Ignore dummy trees.
                if (TrainedEnsemble.GetTreeAt(i).NumLeaves > 1)
                    res.AddTree(TrainedEnsemble.GetTreeAt(i));
            }
            return res;
        }

        private LightGbmBinaryModelParameters CreateBinaryPredictor(int classID, string innerArgs)
        {
            return new LightGbmBinaryModelParameters(Host, GetBinaryEnsemble(classID), FeatureCount, innerArgs);
        }

        private protected override OvaModelParameters CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete.");

            Host.Assert(_numClass > 1, "Must know the number of classes before creating a predictor.");
            Host.Assert(TrainedEnsemble.NumTrees % _numClass == 0, "Number of trees should be a multiple of number of classes.");

            var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            IPredictorProducing<float>[] predictors = new IPredictorProducing<float>[_tlcNumClass];
            for (int i = 0; i < _tlcNumClass; ++i)
            {
                var pred = CreateBinaryPredictor(i, innerArgs);
                var cali = new PlattCalibrator(Host, -0.5, 0);
                predictors[i] = new FeatureWeightsCalibratedModelParameters<LightGbmBinaryModelParameters, PlattCalibrator>(Host, pred, cali);
            }
            string obj = (string)GetGbmParameters()["objective"];
            if (obj == "multiclass")
                return OvaModelParameters.Create(Host, OvaModelParameters.OutputFormula.Softmax, predictors);
            else
                return OvaModelParameters.Create(Host, predictors);
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BoolType || labelType is KeyType || labelType == NumberType.R4))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType}', but must be key, boolean or R4.");
            }
        }

        private protected override void ConvertNaNLabels(IChannel ch, RoleMappedData data, float[] labels)
        {
            // Only initialize one time.
            if (_numClass < 0)
            {
                float minLabel = float.MaxValue;
                float maxLabel = float.MinValue;
                bool hasNaNLabel = false;
                foreach (var labelColumn in labels)
                {
                    if (float.IsNaN(labelColumn))
                        hasNaNLabel = true;
                    else
                    {
                        minLabel = Math.Min(minLabel, labelColumn);
                        maxLabel = Math.Max(maxLabel, labelColumn);
                    }
                }
                ch.CheckParam(minLabel >= 0, nameof(data), "min labelColumn cannot be negative");
                if (maxLabel >= _maxNumClass)
                    throw ch.ExceptParam(nameof(data), $"max labelColumn cannot exceed {_maxNumClass}");

                if (data.Schema.Label.Value.Type is KeyType keyType)
                {
                    if (hasNaNLabel)
                        _numClass = keyType.GetCountAsInt32(Host) + 1;
                    else
                        _numClass = keyType.GetCountAsInt32(Host);
                    _tlcNumClass = keyType.GetCountAsInt32(Host);
                }
                else
                {
                    if (hasNaNLabel)
                        _numClass = (int)maxLabel + 2;
                    else
                        _numClass = (int)maxLabel + 1;
                    _tlcNumClass = (int)maxLabel + 1;
                }
            }
            float defaultLabel = _numClass - 1;
            for (int i = 0; i < labels.Length; ++i)
                if (float.IsNaN(labels[i]))
                    labels[i] = defaultLabel;
        }

        protected override void GetDefaultParameters(IChannel ch, int numRow, bool hasCategorical, int totalCats, bool hiddenMsg = false)
        {
            base.GetDefaultParameters(ch, numRow, hasCategorical, totalCats, true);
            int numLeaves = (int)Options["num_leaves"];
            int minDataPerLeaf = Args.MinDataPerLeaf ?? DefaultMinDataPerLeaf(numRow, numLeaves, _numClass);
            Options["min_data_per_leaf"] = minDataPerLeaf;
            if (!hiddenMsg)
            {
                if (!Args.LearningRate.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(Args.LearningRate) + " = " + Options["learning_rate"]);
                if (!Args.NumLeaves.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(Args.NumLeaves) + " = " + numLeaves);
                if (!Args.MinDataPerLeaf.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(Args.MinDataPerLeaf) + " = " + minDataPerLeaf);
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        {
            Host.AssertValue(ch);
            ch.Assert(PredictionKind == PredictionKind.MultiClassClassification);
            ch.Assert(_numClass > 1);
            Options["num_class"] = _numClass;
            bool useSoftmax = false;

            if (Args.UseSoftmax.HasValue)
                useSoftmax = Args.UseSoftmax.Value;
            else
            {
                if (labels.Length >= _minDataToUseSoftmax)
                    useSoftmax = true;

                ch.Info("Auto-tuning parameters: " + nameof(Args.UseSoftmax) + " = " + useSoftmax);
            }

            if (useSoftmax)
                Options["objective"] = "multiclass";
            else
                Options["objective"] = "multiclassova";

            // Add default metric.
            if (!Options.ContainsKey("metric"))
                Options["metric"] = "multi_error";
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var metadata = new SchemaShape(labelCol.Metadata.Where(x => x.Name == MetadataUtils.Kinds.KeyValues)
                .Concat(MetadataUtils.GetTrainerOutputMetadata()));
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(MetadataUtils.MetadataForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, metadata)
            };
        }

        protected override MulticlassPredictionTransformer<OvaModelParameters> MakeTransformer(OvaModelParameters model, Schema trainSchema)
            => new MulticlassPredictionTransformer<OvaModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);

        public MulticlassPredictionTransformer<OvaModelParameters> Train(IDataView trainData, IDataView validationData = null)
            => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    internal static partial class LightGbm
    {
        [TlcModule.EntryPoint(
            Name = "Trainers.LightGbmClassifier",
            Desc = "Train a LightGBM multi class model.",
            UserName = LightGbmMulticlassTrainer.Summary,
            ShortName = LightGbmMulticlassTrainer.ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClass(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Options, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new LightGbmMulticlassTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
