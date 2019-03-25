// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(GamRegressionTrainer.Summary,
    typeof(GamRegressionTrainer), typeof(GamRegressionTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    GamRegressionTrainer.UserNameValue,
    GamRegressionTrainer.LoadNameValue,
    GamRegressionTrainer.ShortName, DocName = "trainer/GAM.md")]

[assembly: LoadableClass(typeof(GamRegressionModelParameters), null, typeof(SignatureLoadModel),
    "GAM Regression Predictor",
    GamRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a regression model with generalized additive models (GAM).
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="GAM_remarks"]/*' />
    public sealed class GamRegressionTrainer : GamTrainerBase<GamRegressionTrainer.Options, RegressionPredictionTransformer<GamRegressionModelParameters>, GamRegressionModelParameters>
    {
        /// <summary>
        /// Options for the <see cref="GamRegressionTrainer"/>.
        /// </summary>
        public partial class Options : OptionsBase
        {
            /// <summary>
            /// Determines what metric to use for pruning.
            /// </summary>
            /// <value>
            /// 1 means use least absolute deviation; 2 means use least squares. Default is 2.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Metric for pruning. (For regression, 1: L1, 2:L2; default L2)", ShortName = "pmetric")]
            [TGUI(Description = "Metric for pruning. (For regression, 1: L1, 2:L2; default L2")]
            public int PruningMetrics = 2;
        }

        internal const string LoadNameValue = "RegressionGamTrainer";
        internal const string UserNameValue = "Generalized Additive Model for Regression";
        internal const string ShortName = "gamr";

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        internal GamRegressionTrainer(IHostEnvironment env, Options options)
             : base(env, options, LoadNameValue, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName)) { }

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupColumnName">The name for the column containing the example weight.</param>
        /// <param name="numberOfIterations">The number of iterations to use in learning the features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        /// <param name="maximumBinCountPerFeature">The maximum number of bins to use to approximate features</param>
        internal GamRegressionTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupColumnName = null,
            int numberOfIterations = GamDefaults.NumberOfIterations,
            double learningRate = GamDefaults.LearningRate,
            int maximumBinCountPerFeature = GamDefaults.MaximumBinCountPerFeature)
            : base(env, LoadNameValue, TrainerUtils.MakeR4ScalarColumn(labelColumnName), featureColumnName, rowGroupColumnName, numberOfIterations, learningRate, maximumBinCountPerFeature)
        {
        }

        private protected override void CheckLabel(RoleMappedData data)
        {
            data.CheckRegressionLabel();
        }

        private protected override GamRegressionModelParameters TrainModelCore(TrainContext context)
        {
            TrainBase(context);
            return new GamRegressionModelParameters(Host, BinUpperBounds, BinEffects, MeanEffect, InputLength, FeatureMap);
        }

        private protected override ObjectiveFunctionBase CreateObjectiveFunction()
        {
            return new FastTreeRegressionTrainer.ObjectiveImpl(TrainSet, GamTrainerOptions);
        }

        private protected override void DefinePruningTest()
        {
            var validTest = new RegressionTest(ValidSetScore, GamTrainerOptions.PruningMetrics);
            // Because we specify pruning metrics as L2 by default, the results array will have 1 value
            PruningLossIndex = 0;
            PruningTest = new TestHistory(validTest, PruningLossIndex);
        }

        private protected override RegressionPredictionTransformer<GamRegressionModelParameters> MakeTransformer(GamRegressionModelParameters model, DataViewSchema trainSchema)
            => new RegressionPredictionTransformer<GamRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="GamRegressionTrainer"/> using both training and validation data, returns
        /// a <see cref="RegressionPredictionTransformer{RegressionGamModelParameters}"/>.
        /// </summary>
        public RegressionPredictionTransformer<GamRegressionModelParameters> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }
    }

    /// <summary>
    /// The model parameters class for Binary Classification GAMs
    /// </summary>
    public sealed class GamRegressionModelParameters : GamModelParametersBase
    {
        internal const string LoaderSignature = "RegressionGamPredictor";
        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        /// <summary>
        /// Construct a new Regression GAM with the defined properties.
        /// </summary>
        /// <param name="env">The Host Environment</param>
        /// <param name="binUpperBounds">An array of arrays of bin-upper-bounds for each feature.</param>
        /// <param name="binEffects">Anay array of arrays of effect sizes for each bin for each feature.</param>
        /// <param name="intercept">The intercept term for the model. Also referred to as the bias or the mean effect.</param>
        /// <param name="inputLength">The number of features passed from the dataset. Used when the number of input features is
        /// different than the number of shape functions. Use default if all features have a shape function.</param>
        /// <param name="featureToInputMap">A map from the feature shape functions (as described by the binUpperBounds and BinEffects)
        /// to the input feature. Used when the number of input features is different than the number of shape functions. Use default if all features have
        /// a shape function.</param>
        internal GamRegressionModelParameters(IHostEnvironment env,
            double[][] binUpperBounds, double[][] binEffects, double intercept, int inputLength = -1, int[] featureToInputMap = null)
            : base(env, LoaderSignature, binUpperBounds, binEffects, intercept, inputLength, featureToInputMap) { }

        private GamRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx) { }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GAM REGP",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010001, // Added Intercept but collided from release 0.6-0.9
                verWrittenCur: 0x00010002,    // Added Intercept (version revved to address collisions)
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(GamRegressionModelParameters).Assembly.FullName);
        }

        private static GamRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new GamRegressionModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            base.SaveCore(ctx);
        }
    }
}
