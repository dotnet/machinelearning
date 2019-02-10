// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Training;

[assembly: LoadableClass(RegressionGamTrainer.Summary,
    typeof(RegressionGamTrainer), typeof(RegressionGamTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    RegressionGamTrainer.UserNameValue,
    RegressionGamTrainer.LoadNameValue,
    RegressionGamTrainer.ShortName, DocName = "trainer/GAM.md")]

[assembly: LoadableClass(typeof(RegressionGamModelParameters), null, typeof(SignatureLoadModel),
    "GAM Regression Predictor",
    RegressionGamModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    public sealed class RegressionGamTrainer : GamTrainerBase<RegressionGamTrainer.Options, RegressionPredictionTransformer<RegressionGamModelParameters>, RegressionGamModelParameters>
    {
        public partial class Options : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Metric for pruning. (For regression, 1: L1, 2:L2; default L2)", ShortName = "pmetric")]
            [TGUI(Description = "Metric for pruning. (For regression, 1: L1, 2:L2; default L2")]
            public int PruningMetrics = 2;
        }

        internal const string LoadNameValue = "RegressionGamTrainer";
        internal const string UserNameValue = "Generalized Additive Model for Regression";
        internal const string ShortName = "gamr";

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        internal RegressionGamTrainer(IHostEnvironment env, Options options)
             : base(env, options, LoadNameValue, TrainerUtils.MakeR4ScalarColumn(options.LabelColumn)) { }

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeBinaryClassificationTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="numIterations">The number of iterations to use in learning the features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        /// <param name="maxBins">The maximum number of bins to use to approximate features</param>
        internal RegressionGamTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null,
            int numIterations = GamDefaults.NumIterations,
            double learningRate = GamDefaults.LearningRates,
            int maxBins = GamDefaults.MaxBins)
            : base(env, LoadNameValue, TrainerUtils.MakeR4ScalarColumn(labelColumn), featureColumn, weightColumn, numIterations, learningRate, maxBins)
        {
        }

        private protected override void CheckLabel(RoleMappedData data)
        {
            data.CheckRegressionLabel();
        }

        private protected override RegressionGamModelParameters TrainModelCore(TrainContext context)
        {
            TrainBase(context);
            return new RegressionGamModelParameters(Host, BinUpperBounds, BinEffects, MeanEffect, InputLength, FeatureMap);
        }

        protected override ObjectiveFunctionBase CreateObjectiveFunction()
        {
            return new FastTreeRegressionTrainer.ObjectiveImpl(TrainSet, Args);
        }

        protected override void DefinePruningTest()
        {
            var validTest = new RegressionTest(ValidSetScore, Args.PruningMetrics);
            // Because we specify pruning metrics as L2 by default, the results array will have 1 value
            PruningLossIndex = 0;
            PruningTest = new TestHistory(validTest, PruningLossIndex);
        }

        protected override RegressionPredictionTransformer<RegressionGamModelParameters> MakeTransformer(RegressionGamModelParameters model, Schema trainSchema)
            => new RegressionPredictionTransformer<RegressionGamModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        public RegressionPredictionTransformer<RegressionGamModelParameters> Train(IDataView trainData, IDataView validationData = null)
            => TrainTransformer(trainData, validationData);

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }
    }

    /// <summary>
    /// The model parameters class for Binary Classification GAMs
    /// </summary>
    public sealed class RegressionGamModelParameters : GamModelParametersBase
    {
        internal const string LoaderSignature = "RegressionGamPredictor";
        public override PredictionKind PredictionKind => PredictionKind.Regression;

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
        public RegressionGamModelParameters(IHostEnvironment env,
            double[][] binUpperBounds, double[][] binEffects, double intercept, int inputLength = -1, int[] featureToInputMap = null)
            : base(env, LoaderSignature, binUpperBounds, binEffects, intercept, inputLength, featureToInputMap) { }

        private RegressionGamModelParameters(IHostEnvironment env, ModelLoadContext ctx)
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
                loaderAssemblyName: typeof(RegressionGamModelParameters).Assembly.FullName);
        }

        private static RegressionGamModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new RegressionGamModelParameters(env, ctx);
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
