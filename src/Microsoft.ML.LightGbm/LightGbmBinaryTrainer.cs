// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

[assembly: LoadableClass(LightGbmBinaryTrainer.Summary, typeof(LightGbmBinaryTrainer), typeof(LightGbmBinaryTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmBinaryTrainer.UserName, LightGbmBinaryTrainer.LoadNameValue, LightGbmBinaryTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(LightGbmBinaryModelParameters), null, typeof(SignatureLoadModel),
    "LightGBM Binary Executor",
    LightGbmBinaryModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(LightGbm), null, typeof(SignatureEntryPointModule), "LightGBM")]

namespace Microsoft.ML.Trainers.LightGbm
{
    public sealed class LightGbmBinaryModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "LightGBMBinaryExec";
        internal const string RegistrationName = "LightGBMBinaryPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBBINCL",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                //verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LightGbmBinaryModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010002;
        private protected override uint VerDefaultValueSerialized => 0x00010004;
        private protected override uint VerCategoricalSplitSerialized => 0x00010005;
        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        internal LightGbmBinaryModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmBinaryModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new LightGbmBinaryModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new ValueMapperCalibratedModelParameters<LightGbmBinaryModelParameters, ICalibrator>(env, predictor, calibrator);
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a boosted decision tree binary classification model using LightGBM.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM_remarks"]/*' />
    public sealed class LightGbmBinaryTrainer : LightGbmTrainerBase<LightGbmBinaryTrainer.Options, float,
        BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>,
        CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
    {
        internal const string UserName = "LightGBM Binary Classifier";
        internal const string LoadNameValue = "LightGBMBinary";
        internal const string ShortName = "LightGBM";
        internal const string Summary = "Train a LightGBM binary classification model.";

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public sealed class Options : OptionsBase
        {

            public enum EvaluateMetricType
            {
                None,
                Default,
                Logloss,
                Error,
                AreaUnderCurve,
            };

            /// <summary>
            /// Whether training data is unbalanced.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use for binary classification when training data is not balanced.", ShortName = "us")]
            public bool UnbalancedSets = false;

            /// <summary>
            /// Controls the balance of positive and negative weights in <see cref="LightGbmBinaryTrainer"/>.
            /// </summary>
            /// <value>
            /// This is useful for training on unbalanced data. A typical value to consider is sum(negative cases) / sum(positive cases).
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Control the balance of positive and negative weights, useful for unbalanced classes." +
                " A typical value to consider: sum(negative cases) / sum(positive cases).",
                ShortName = "ScalePosWeight")]
            public double WeightOfPositiveExamples = 1;

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
            public EvaluateMetricType EvaluationMetric = EvaluateMetricType.Logloss;

            static Options()
            {
                NameMapping.Add(nameof(EvaluateMetricType), "metric");
                NameMapping.Add(nameof(EvaluateMetricType.None), "");
                NameMapping.Add(nameof(EvaluateMetricType.Logloss), "binary_logloss");
                NameMapping.Add(nameof(EvaluateMetricType.Error), "binary_error");
                NameMapping.Add(nameof(EvaluateMetricType.AreaUnderCurve), "auc");
                NameMapping.Add(nameof(WeightOfPositiveExamples), "scale_pos_weight");
            }

            internal override Dictionary<string, object> ToDictionary(IHost host)
            {
                var res = base.ToDictionary(host);
                res[GetOptionName(nameof(UnbalancedSets))] = UnbalancedSets;
                res[GetOptionName(nameof(WeightOfPositiveExamples))] = WeightOfPositiveExamples;
                res[GetOptionName(nameof(Sigmoid))] = Sigmoid;
                if (EvaluationMetric != EvaluateMetricType.Default)
                    res[GetOptionName(nameof(EvaluateMetricType))] = GetOptionName(EvaluationMetric.ToString());

                return res;
            }
        }

        internal LightGbmBinaryTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            Contracts.CheckUserArg(options.Sigmoid > 0, nameof(Options.Sigmoid), "must be > 0.");
            Contracts.CheckUserArg(options.WeightOfPositiveExamples > 0, nameof(Options.WeightOfPositiveExamples), "must be > 0.");
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of The label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        internal LightGbmBinaryTrainer(IHostEnvironment env,
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

        private protected override CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator> CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(base.GbmOptions);
            var pred = new LightGbmBinaryModelParameters(Host, TrainedEnsemble, FeatureCount, innerArgs);
            var cali = new PlattCalibrator(Host, -0.5, 0);
            return new FeatureWeightsCalibratedModelParameters<LightGbmBinaryModelParameters, PlattCalibrator>(Host, pred, cali);
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BooleanDataViewType || labelType is KeyType || labelType == NumberDataViewType.Single))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType.RawType}', but must be unsigned int, boolean or float.");
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
            => GbmOptions["objective"] = "binary";

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
            MakeTransformer(CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator> model, DataViewSchema trainSchema)
         => new BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="LightGbmBinaryTrainer"/> using both training and validation data, returns
        /// a <see cref="BinaryPredictionTransformer{CalibratedModelParametersBase}"/>.
        /// </summary>
        public BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    internal static partial class LightGbm
    {
        [TlcModule.EntryPoint(
            Name = "Trainers.LightGbmBinaryClassifier",
            Desc = LightGbmBinaryTrainer.Summary,
            UserName = LightGbmBinaryTrainer.UserName,
            ShortName = LightGbmBinaryTrainer.ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, LightGbmBinaryTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<LightGbmBinaryTrainer.Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LightGbmBinaryTrainer(host, input),
                getLabel: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                getWeight: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }
}
