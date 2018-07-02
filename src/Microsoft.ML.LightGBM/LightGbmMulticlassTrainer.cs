// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;

[assembly: LoadableClass(LightGbmMulticlassTrainer.Summary, typeof(LightGbmMulticlassTrainer), typeof(LightGbmArguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    "LightGBM Multi-class Classifier", LightGbmMulticlassTrainer.LoadNameValue, LightGbmMulticlassTrainer.ShortName, DocName = "trainer/LightGBM.md")]

namespace Microsoft.ML.Runtime.LightGBM
{

    public sealed class LightGbmMulticlassTrainer : LightGbmTrainerBase<VBuffer<float>, OvaPredictor>
    {
        public const string Summary = "LightGBM Multi Class Classifier";
        public const string LoadNameValue = "LightGBMMulticlass";
        public const string ShortName = "LightGBMMC";
        private const int _minDataToUseSoftmax = 50000;

        private const double _maxNumClass = 1e6;
        private int _numClass;
        private int _tlcNumClass;

        public LightGbmMulticlassTrainer(IHostEnvironment env, LightGbmArguments args)
            : base(env, args, PredictionKind.MultiClassClassification, "LightGBMMulticlass")
        {
            _numClass = -1;
        }

        private FastTree.Internal.Ensemble GetBinaryEnsemble(int classID)
        {
            FastTree.Internal.Ensemble res = new FastTree.Internal.Ensemble();
            for (int i = classID; i < TrainedEnsemble.NumTrees; i += _numClass)
            {
                // Ignore dummy trees.
                if (TrainedEnsemble.GetTreeAt(i).NumLeaves > 1)
                    res.AddTree(TrainedEnsemble.GetTreeAt(i));
            }
            return res;
        }

        private LightGbmBinaryPredictor CreateBinaryPredictor(int classID, string innerArgs)
        {
            return new LightGbmBinaryPredictor(Host, GetBinaryEnsemble(classID), FeatureCount, innerArgs);
        }

        public override OvaPredictor CreatePredictor()
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
                predictors[i] = new FeatureWeightsCalibratedPredictor(Host, pred, cali);
            }
            return OvaPredictor.Create(Host, predictors);
        }

        protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Type;
            if (!(labelType.IsBool || labelType.IsKey || labelType == NumberType.R4))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Name}' is of type '{labelType}', but must be key, boolean or R4.");
            }
        }

        protected override void ConvertNaNLabels(IChannel ch, RoleMappedData data, float[] labels)
        {
            // Only initialize one time.
            if (_numClass < 0)
            {
                float minLabel = float.MaxValue;
                float maxLabel = float.MinValue;
                bool hasNaNLabel = false;
                foreach (var label in labels)
                {
                    if (float.IsNaN(label))
                        hasNaNLabel = true;
                    else
                    {
                        minLabel = Math.Min(minLabel, label);
                        maxLabel = Math.Max(maxLabel, label);
                    }
                }
                ch.CheckParam(minLabel >= 0, nameof(data), "min label cannot be negative");
                if (maxLabel >= _maxNumClass)
                    throw ch.ExceptParam(nameof(data), $"max label cannot exceed {_maxNumClass}");

                if (data.Schema.Label.Type.IsKey)
                {
                    ch.Check(data.Schema.Label.Type.AsKey.Contiguous, "label value should be contiguous");
                    if (hasNaNLabel)
                        _numClass = data.Schema.Label.Type.AsKey.Count + 1;
                    else
                        _numClass = data.Schema.Label.Type.AsKey.Count;
                    _tlcNumClass = data.Schema.Label.Type.AsKey.Count;
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

        protected override void GetDefaultParameters(IChannel ch, int numRow, bool hasCategorical, int totalCats, bool hiddenMsg=false)
        {
            base.GetDefaultParameters(ch, numRow, hasCategorical, totalCats, true);
            int numLeaves = int.Parse(Options["num_leaves"]);
            int minDataPerLeaf = Args.MinDataPerLeaf ?? DefaultMinDataPerLeaf(numRow, numLeaves, _numClass);
            Options["min_data_per_leaf"] = minDataPerLeaf.ToString();
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

        protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        {
            Host.AssertValue(ch);
            ch.Assert(PredictionKind == PredictionKind.MultiClassClassification);
            ch.Assert(_numClass > 1);
            Options["num_class"] = _numClass.ToString();
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
    }

    /// <summary>
    /// A component to train an LightGBM model.
    /// </summary>
    public static partial class LightGbm
    {
        [TlcModule.EntryPoint(
            Name = "Trainers.LightGbmClassifier", 
            Desc = "Train a LightGBM multi class model.", 
            UserName = LightGbmMulticlassTrainer.Summary, 
            ShortName = LightGbmMulticlassTrainer.ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClass(IHostEnvironment env, LightGbmArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<LightGbmArguments, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new LightGbmMulticlassTrainer(host, input),
                getLabel: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                getWeight: () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
