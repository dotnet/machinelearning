// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.XGBoost;

[assembly: LoadableClass(XGBoostRegressionTrainer.Summary, typeof(XGBoostRegressionTrainer), typeof(XGBoostRegressionTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    XGBoostRegressionTrainer.UserNameValue, XGBoostRegressionTrainer.LoadNameValue, XGBoostRegressionTrainer.ShortName)]

[assembly: LoadableClass(typeof(XGBoostRegressionModelParameters), null, typeof(SignatureLoadModel),
    "XGBoost Regression Executor",
    XGBoostRegressionModelParameters.LoaderSignature)]
namespace Microsoft.ML.Trainers.XGBoost
{
    /// <summary>
    /// Model parameters for <see cref="XGBoostRegressionTrainer"/>.
    /// </summary>
    public sealed class XGBoostRegressionModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "XGBoostRegressionExec";
        internal const string RegistrationName = "XGBoostRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "XGBSIREG",
                verWrittenCur: 0x00010001, // Initial (not sure if "Categorical splits") are supported
                                           // verWrittenCur: 0x00010002, // _numFeatures serialized
                                           // verWrittenCur: 0x00010003, // Ini content out of predictor
                                           // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(XGBoostRegressionModelParameters).Assembly.FullName);
        }
        private protected override uint VerNumFeaturesSerialized => 0x00010002;
        private protected override uint VerDefaultValueSerialized => 0x00010004;
        private protected override uint VerCategoricalSplitSerialized => 0x00010005;
        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        internal XGBoostRegressionModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private XGBoostRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        internal static XGBoostRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new XGBoostRegressionModelParameters(env, ctx);
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a boosted decision tree regression model using XGBoost.
    /// </summary>
    public sealed class XGBoostRegressionTrainer :
        XGBoostTrainerBase<XGBoostRegressionTrainer.Options, float, RegressionPredictionTransformer<XGBoostRegressionModelParameters>, XGBoostRegressionModelParameters>
    {
        internal new const string Summary = "XGBoost Regression";
        internal new const string LoadNameValue = "XGBoostRegression";
        internal const string ShortName = "XGBoostR";
        internal new const string UserNameValue = "XGBoost Regressor";

#if false
        public XGBoostRegressionTrainer(IHost host, SchemaShape.Column feature, SchemaShape.Column label, SchemaShape.Column weight = default, SchemaShape.Column groupId = default)
            : base(host, feature, label, weight, groupId)
        {
            Console.WriteLine("In the constructor of the trainer 1");
        }
#endif

        public override TrainerInfo Info => throw new System.NotImplementedException();

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override RegressionPredictionTransformer<XGBoostRegressionModelParameters> MakeTransformer(XGBoostRegressionModelParameters model, DataViewSchema trainSchema)
#if false
=> new RegressionPredictionTransformer<XGBoostRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);
#else
        {
            return null;
        }
#endif

        /// <summary>
        /// Options for the <see cref="XGBoostRegressionTrainer"/> as used in
        /// [XGBoost(Options)](xref:Microsoft.ML.XGBoostExtensions.XGBoost(Microsoft.ML.RegressionCatalog.RegressionTrainers,Microsoft.ML.Trainers.XGBoost.XGBoostRegressionTrainer.Options)).
        /// </summary>
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
                NameMapping.Add(nameof(EvaluateMetricType.None), "None");
                NameMapping.Add(nameof(EvaluateMetricType.Default), "");
                NameMapping.Add(nameof(EvaluateMetricType.MeanAbsoluteError), "mae");
                NameMapping.Add(nameof(EvaluateMetricType.RootMeanSquaredError), "rmse");
                NameMapping.Add(nameof(EvaluateMetricType.MeanSquaredError), "mse");
            }

            internal override Dictionary<string, object> ToDictionary(IHost host)
            {
                var res = base.ToDictionary(host);
                res[GetOptionName(nameof(EvaluateMetricType))] = GetOptionName(EvaluationMetric.ToString());

                return res;
            }
        }

        /// <summary>
        /// Initializes a new instance of <see cref="XGBoostRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        internal XGBoostRegressionTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations
            )
            : this(env, new Options()
            {
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
                ExampleWeightColumnName = exampleWeightColumnName,
                NumberOfLeaves = numberOfLeaves,
#if false
                MinimumExampleCountPerLeaf = minimumExampleCountPerLeaf,
                LearningRate = learningRate,
                NumberOfIterations = numberOfIterations
#endif
            })
        {

            Console.WriteLine("In the constructor of the trainer 2");
        }

        internal XGBoostRegressionTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
            Console.WriteLine("In the constructor of the trainer 3");
        }

        private protected override XGBoostRegressionModelParameters CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null,
                "The predictor cannot be created before training is complete");
#if true
            return null;
#else
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(GbmOptions);
            return new LightGbmRegressionModelParameters(Host, TrainedEnsemble, FeatureCount, innerArgs);
#endif
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BooleanDataViewType || labelType is KeyDataViewType || labelType == NumberDataViewType.Single))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType.RawType}', but must be an unsigned int, boolean or float.");
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data
#if false
	, float[] labels, int[] groups
#endif
    )
        {
            System.Console.WriteLine("****** Checking and updating parameters");
            GbmOptions["objective"] = "reg:squarederror";
        }

#if false
        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }
#endif
    }
}
