// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

[assembly: LoadableClass(LightGbmRegressionTrainer.Summary, typeof(LightGbmRegressionTrainer), typeof(LightGbmRegressionTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmRegressionTrainer.UserNameValue, LightGbmRegressionTrainer.LoadNameValue, LightGbmRegressionTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(LightGbmRegressionModelParameters), null, typeof(SignatureLoadModel),
    "LightGBM Regression Executor",
    LightGbmRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.LightGbm
{
    public sealed class LightGbmRegressionModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "LightGBMRegressionExec";
        internal const string RegistrationName = "LightGBMRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBSIREG",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LightGbmRegressionModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010002;
        private protected override uint VerDefaultValueSerialized => 0x00010004;
        private protected override uint VerCategoricalSplitSerialized => 0x00010005;
        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        internal LightGbmRegressionModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static LightGbmRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LightGbmRegressionModelParameters(env, ctx);
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a boosted decision tree regression model using LightGBM.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="LightGBM_remarks"]/*' />
    public sealed class LightGbmRegressionTrainer : LightGbmTrainerBase<LightGbmRegressionTrainer.Options,
                                                                            float,
                                                                            RegressionPredictionTransformer<LightGbmRegressionModelParameters>,
                                                                            LightGbmRegressionModelParameters>
    {
        internal const string Summary = "LightGBM Regression";
        internal const string LoadNameValue = "LightGBMRegression";
        internal const string ShortName = "LightGBMR";
        internal const string UserNameValue = "LightGBM Regressor";

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        public sealed class Options : OptionsBase
        {
            public enum EvaluateMetricType
            {
                None,
                Default,
                MeanAbsoluteError,
                RootMeanSquaredError,
                MeanSquaredError
            };

            /// <summary>
            /// Determines what evaluation metric to use.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Evaluation metrics.",
                ShortName = "em")]
            public EvaluateMetricType EvaluationMetric = EvaluateMetricType.RootMeanSquaredError;

            static Options()
            {
                NameMapping.Add(nameof(EvaluateMetricType), "metric");
                NameMapping.Add(nameof(EvaluateMetricType.None), "");
                NameMapping.Add(nameof(EvaluateMetricType.MeanAbsoluteError), "mae");
                NameMapping.Add(nameof(EvaluateMetricType.RootMeanSquaredError), "rmse");
                NameMapping.Add(nameof(EvaluateMetricType.MeanSquaredError), "mse");
            }

            internal override Dictionary<string, object> ToDictionary(IHost host)
            {
                var res = base.ToDictionary(host);
                if (EvaluationMetric != EvaluateMetricType.Default)
                    res[GetOptionName(nameof(EvaluateMetricType))] = GetOptionName(EvaluationMetric.ToString());

                return res;
            }
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        internal LightGbmRegressionTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
            : this(env, new Options()
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

        internal LightGbmRegressionTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
        }

        private protected override LightGbmRegressionModelParameters CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null,
                "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(GbmOptions);
            return new LightGbmRegressionModelParameters(Host, TrainedEnsemble, FeatureCount, innerArgs);
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BooleanDataViewType || labelType is KeyType || labelType == NumberDataViewType.Single))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType.RawType}', but must be an unsigned int, boolean or float.");
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        {
            GbmOptions["objective"] = "regression";
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override RegressionPredictionTransformer<LightGbmRegressionModelParameters> MakeTransformer(LightGbmRegressionModelParameters model, DataViewSchema trainSchema)
            => new RegressionPredictionTransformer<LightGbmRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="LightGbmRegressionTrainer"/> using both training and validation data, returns
        /// a <see cref="RegressionPredictionTransformer{LightGbmRegressionModelParameters}"/>.
        /// </summary>
        public RegressionPredictionTransformer<LightGbmRegressionModelParameters> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    internal static partial class LightGbm
    {
        [TlcModule.EntryPoint(Name = "Trainers.LightGbmRegressor",
            Desc = LightGbmRegressionTrainer.Summary,
            UserName = LightGbmRegressionTrainer.UserNameValue,
            ShortName = LightGbmRegressionTrainer.ShortName)]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, LightGbmRegressionTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<LightGbmRegressionTrainer.Options, CommonOutputs.RegressionOutput>(host, input,
                () => new LightGbmRegressionTrainer(host, input),
                getLabel: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                getWeight: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }
}
