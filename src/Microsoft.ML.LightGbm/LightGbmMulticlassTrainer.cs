// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

[assembly: LoadableClass(LightGbmMulticlassTrainer.Summary, typeof(LightGbmMulticlassTrainer), typeof(LightGbmMulticlassTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    "LightGBM Multi-class Classifier", LightGbmMulticlassTrainer.LoadNameValue, LightGbmMulticlassTrainer.ShortName, DocName = "trainer/LightGBM.md")]

namespace Microsoft.ML.Trainers.LightGbm
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a boosted decision tree multi-class classification model using LightGBM.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [LightGbm](xref:Microsoft.ML.LightGbmExtensions.LightGbm(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,System.Nullable{System.Int32},System.Nullable{System.Int32},System.Nullable{System.Double},System.Int32))
    /// or [LightGbm(Options)](xref:Microsoft.ML.LightGbmExtensions.LightGbm(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-multiclass-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.LightGbm |
    /// | Exportable to ONNX | Yes |
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/algo-details-lightgbm.md)]
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="LightGbmExtensions.LightGbm(MulticlassClassificationCatalog.MulticlassClassificationTrainers, string, string, string, int?, int?, double?, int)"/>
    /// <seealso cref="LightGbmExtensions.LightGbm(MulticlassClassificationCatalog.MulticlassClassificationTrainers, LightGbmMulticlassTrainer.Options)"/>
    /// <seealso cref="Options"/>
    public sealed class LightGbmMulticlassTrainer : LightGbmTrainerBase<LightGbmMulticlassTrainer.Options,
                                                                        VBuffer<float>,
                                                                        MulticlassPredictionTransformer<OneVersusAllModelParameters>,
                                                                        OneVersusAllModelParameters>
    {
        internal const string Summary = "LightGBM Multi Class Classifier";
        internal const string LoadNameValue = "LightGBMMulticlass";
        internal const string ShortName = "LightGBMMC";
        private const int _minDataToUseSoftmax = 50000;

        private const double _maxNumClass = 1e6;

        // If there are NaN labels, they are converted to be equal to _numberOfClassesIncludingNan - 1.
        // This is done because NaN labels are going to be seen as an extra different class, when training the model in the WrappedLightGbmTraining class
        // But, when creating the Predictors, only _numberOfClasses is considered, ignoring the "extra class" of NaN labels.
        private int _numberOfClassesIncludingNan;
        private int _numberOfClasses;

        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        /// <summary>
        /// Options for the <see cref="LightGbmMulticlassTrainer"/> as used in
        ///  [LightGbm(Options)](xref:Microsoft.ML.LightGbmExtensions.LightGbm(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer.Options)).
        /// </summary>
        public sealed class Options : OptionsBase
        {
            public enum EvaluateMetricType
            {
                None,
                Default,
                Error,
                LogLoss,
            }

            /// <summary>
            /// Whether training data is unbalanced.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use for multi-class classification when training data is not balanced", ShortName = "us")]
            public bool UnbalancedSets = false;

            /// <summary>
            /// Whether to use softmax loss.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use softmax loss for the multi classification.")]
            [TlcModule.SweepableDiscreteParam("UseSoftmax", new object[] { true, false })]
            public bool? UseSoftmax;

            /// <summary>
            /// Parameter for the sigmoid function.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter for the sigmoid function.", ShortName = "sigmoid")]
            [TGUI(Label = "Sigmoid", SuggestedSweeps = "0.5,1")]
            public double Sigmoid = 0.5;

            /// <summary>
            /// Determines what evaluation metric to use.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Evaluation metrics.",
                ShortName = "em")]
            public EvaluateMetricType EvaluationMetric = EvaluateMetricType.Error;

            static Options()
            {
                NameMapping.Add(nameof(EvaluateMetricType), "metric");
                NameMapping.Add(nameof(EvaluateMetricType.None), "None");
                NameMapping.Add(nameof(EvaluateMetricType.Default), "");
                NameMapping.Add(nameof(EvaluateMetricType.Error), "multi_error");
                NameMapping.Add(nameof(EvaluateMetricType.LogLoss), "multi_logloss");
            }

            internal override Dictionary<string, object> ToDictionary(IHost host)
            {
                var res = base.ToDictionary(host);

                res[GetOptionName(nameof(UnbalancedSets))] = UnbalancedSets;
                res[GetOptionName(nameof(Sigmoid))] = Sigmoid;
                res[GetOptionName(nameof(EvaluateMetricType))] = GetOptionName(EvaluationMetric.ToString());

                return res;
            }
        }

        internal LightGbmMulticlassTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeU4ScalarColumn(options.LabelColumnName))
        {
            Contracts.CheckUserArg(options.Sigmoid > 0, nameof(Options.Sigmoid), "must be > 0.");
            _numberOfClassesIncludingNan = -1;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmMulticlassTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of The label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of iterations to use.</param>
        internal LightGbmMulticlassTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
            : this(env,
                  new Options()
                  {
                      LabelColumnName = labelColumnName,
                      FeatureColumnName = featureColumnName,
                      ExampleWeightColumnName = exampleWeightColumnName,
                      NumberOfLeaves = numberOfLeaves,
                      MinimumExampleCountPerLeaf = minimumExampleCountPerLeaf,
                      LearningRate = learningRate,
                      NumberOfIterations = numberOfIterations
                  })
        {
        }

        private InternalTreeEnsemble GetBinaryEnsemble(int classID)
        {
            var res = new InternalTreeEnsemble();
            for (int i = classID; i < TrainedEnsemble.NumTrees; i += _numberOfClassesIncludingNan)
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

        private protected override OneVersusAllModelParameters CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete.");

            Host.Assert(_numberOfClassesIncludingNan > 1, "Must know the number of classes before creating a predictor.");
            Host.Assert(TrainedEnsemble.NumTrees % _numberOfClassesIncludingNan == 0, "Number of trees should be a multiple of number of classes.");

            var innerArgs = LightGbmInterfaceUtils.JoinParameters(GbmOptions);
            IPredictorProducing<float>[] predictors = new IPredictorProducing<float>[_numberOfClasses];
            for (int i = 0; i < _numberOfClasses; ++i)
            {
                var pred = CreateBinaryPredictor(i, innerArgs);
                var cali = new PlattCalibrator(Host, -LightGbmTrainerOptions.Sigmoid, 0);
                predictors[i] = new FeatureWeightsCalibratedModelParameters<LightGbmBinaryModelParameters, PlattCalibrator>(Host, pred, cali);
            }
            string obj = (string)GetGbmParameters()["objective"];
            if (obj == "multiclass")
                return OneVersusAllModelParameters.Create(Host, OneVersusAllModelParameters.OutputFormula.Softmax, predictors);
            else
                return OneVersusAllModelParameters.Create(Host, predictors);
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BooleanDataViewType || labelType is KeyDataViewType || labelType == NumberDataViewType.Single))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType.RawType}', but must be of unsigned int, boolean or float.");
            }
        }

        private protected override void InitializeBeforeTraining()
        {
            _numberOfClassesIncludingNan = -1;
            _numberOfClasses = 0;
        }

        private protected override void ConvertNaNLabels(IChannel ch, RoleMappedData data, float[] labels)
        {
            // Only initialize one time.

            if (_numberOfClassesIncludingNan < 0)
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
                ch.CheckParam(minLabel >= 0, nameof(data), "Minimum value in label column cannot be negative");
                if (maxLabel >= _maxNumClass)
                    throw ch.ExceptParam(nameof(data), $"Maximum value {maxLabel} in label column exceeds {_maxNumClass}");

                if (data.Schema.Label.Value.Type is KeyDataViewType keyType)
                {
                    if (hasNaNLabel)
                        _numberOfClassesIncludingNan = keyType.GetCountAsInt32(Host) + 1;
                    else
                        _numberOfClassesIncludingNan = keyType.GetCountAsInt32(Host);
                    _numberOfClasses = keyType.GetCountAsInt32(Host);
                }
                else
                {
                    if (hasNaNLabel)
                        _numberOfClassesIncludingNan = (int)maxLabel + 2;
                    else
                        _numberOfClassesIncludingNan = (int)maxLabel + 1;
                    _numberOfClasses = (int)maxLabel + 1;
                }
            }

            float defaultLabel = _numberOfClassesIncludingNan - 1;
            for (int i = 0; i < labels.Length; ++i)
                if (float.IsNaN(labels[i]))
                    labels[i] = defaultLabel;
        }

        private protected override void GetDefaultParameters(IChannel ch, int numRow, bool hasCategorical, int totalCats, bool hiddenMsg = false)
        {
            base.GetDefaultParameters(ch, numRow, hasCategorical, totalCats, true);
            int numberOfLeaves = (int)GbmOptions["num_leaves"];
            int minimumExampleCountPerLeaf = LightGbmTrainerOptions.MinimumExampleCountPerLeaf ?? DefaultMinDataPerLeaf(numRow, numberOfLeaves, _numberOfClassesIncludingNan);
            GbmOptions["min_data_per_leaf"] = minimumExampleCountPerLeaf;
            if (!hiddenMsg)
            {
                if (!LightGbmTrainerOptions.LearningRate.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.LearningRate) + " = " + GbmOptions["learning_rate"]);
                if (!LightGbmTrainerOptions.NumberOfLeaves.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.NumberOfLeaves) + " = " + numberOfLeaves);
                if (!LightGbmTrainerOptions.MinimumExampleCountPerLeaf.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.MinimumExampleCountPerLeaf) + " = " + minimumExampleCountPerLeaf);
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        {
            Host.AssertValue(ch);
            ch.Assert(PredictionKind == PredictionKind.MulticlassClassification);
            ch.Assert(_numberOfClassesIncludingNan > 1);
            GbmOptions["num_class"] = _numberOfClassesIncludingNan;
            bool useSoftmax = false;

            if (LightGbmTrainerOptions.UseSoftmax.HasValue)
                useSoftmax = LightGbmTrainerOptions.UseSoftmax.Value;
            else
            {
                if (labels.Length >= _minDataToUseSoftmax)
                    useSoftmax = true;

                ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.UseSoftmax) + " = " + useSoftmax);
            }

            if (useSoftmax)
                GbmOptions["objective"] = "multiclass";
            else
                GbmOptions["objective"] = "multiclassova";
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var metadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, metadata)
            };
        }

        private protected override MulticlassPredictionTransformer<OneVersusAllModelParameters> MakeTransformer(OneVersusAllModelParameters model, DataViewSchema trainSchema)
            => new MulticlassPredictionTransformer<OneVersusAllModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);

        /// <summary>
        /// Trains a <see cref="LightGbmMulticlassTrainer"/> using both training and validation data, returns
        /// a <see cref="MulticlassPredictionTransformer{OneVsAllModelParameters}"/>.
        /// </summary>
        public MulticlassPredictionTransformer<OneVersusAllModelParameters> Fit(IDataView trainData, IDataView validationData)
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
        public static CommonOutputs.MulticlassClassificationOutput TrainMulticlass(IHostEnvironment env, LightGbmMulticlassTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<LightGbmMulticlassTrainer.Options, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new LightGbmMulticlassTrainer(host, input),
                getLabel: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                getWeight: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }
}
