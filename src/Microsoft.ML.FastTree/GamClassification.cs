// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Calibrator;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Training;

[assembly: LoadableClass(BinaryClassificationGamTrainer.Summary,
    typeof(BinaryClassificationGamTrainer), typeof(BinaryClassificationGamTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    BinaryClassificationGamTrainer.UserNameValue,
    BinaryClassificationGamTrainer.LoadNameValue,
    BinaryClassificationGamTrainer.ShortName, DocName = "trainer/GAM.md")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(BinaryClassificationGamModelParameters), null, typeof(SignatureLoadModel),
    "GAM Binary Class Predictor",
    BinaryClassificationGamModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    public sealed class BinaryClassificationGamTrainer :
        GamTrainerBase<BinaryClassificationGamTrainer.Options,
        BinaryPredictionTransformer<CalibratedModelParametersBase<BinaryClassificationGamModelParameters, PlattCalibrator>>,
        CalibratedModelParametersBase<BinaryClassificationGamModelParameters, PlattCalibrator>>
    {
        public sealed class Options : OptionsBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Should we use derivatives optimized for unbalanced sets", ShortName = "us")]
            [TGUI(Label = "Optimize for unbalanced")]
            public bool UnbalancedSets = false;
        }

        internal const string LoadNameValue = "BinaryClassificationGamTrainer";
        internal const string UserNameValue = "Generalized Additive Model for Binary Classification";
        internal const string ShortName = "gam";
        private readonly double _sigmoidParameter;

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        private protected override bool NeedCalibration => true;

        /// <summary>
        /// Initializes a new instance of <see cref="BinaryClassificationGamTrainer"/>
        /// </summary>
        internal BinaryClassificationGamTrainer(IHostEnvironment env, Options options)
             : base(env, options, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(options.LabelColumn))
        {
            _sigmoidParameter = 1;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="BinaryClassificationGamTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="numIterations">The number of iterations to use in learning the features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        /// <param name="maxBins">The maximum number of bins to use to approximate features</param>
        internal BinaryClassificationGamTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null,
            int numIterations = GamDefaults.NumIterations,
            double learningRate = GamDefaults.LearningRates,
            int maxBins = GamDefaults.MaxBins)
            : base(env, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(labelColumn), featureColumn, weightColumn, numIterations, learningRate, maxBins)
        {
            _sigmoidParameter = 1;
        }

        private protected override void CheckLabel(RoleMappedData data)
        {
            data.CheckBinaryLabel();
        }

        private static bool[] ConvertTargetsToBool(double[] targets)
        {
            bool[] boolArray = new bool[targets.Length];
            int innerLoopSize = 1 + targets.Length / BlockingThreadPool.NumThreads;
            var actions = new Action[(int)Math.Ceiling(1.0 * targets.Length / innerLoopSize)];
            var actionIndex = 0;
            for (int d = 0; d < targets.Length; d += innerLoopSize)
            {
                var fromDoc = d;
                var toDoc = Math.Min(d + innerLoopSize, targets.Length);
                actions[actionIndex++] = () =>
                {
                    for (int doc = fromDoc; doc < toDoc; doc++)
                        boolArray[doc] = targets[doc] > 0;
                };
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
            return boolArray;
        }
        private protected override CalibratedModelParametersBase<BinaryClassificationGamModelParameters, PlattCalibrator> TrainModelCore(TrainContext context)
        {
            TrainBase(context);
            var predictor = new BinaryClassificationGamModelParameters(Host,
                BinUpperBounds, BinEffects, MeanEffect, InputLength, FeatureMap);
            var calibrator = new PlattCalibrator(Host, -1.0 * _sigmoidParameter, 0);
            return new ValueMapperCalibratedModelParameters<BinaryClassificationGamModelParameters, PlattCalibrator>(Host, predictor, calibrator);
        }

        private protected override ObjectiveFunctionBase CreateObjectiveFunction()
        {
            return new FastTreeBinaryClassificationTrainer.ObjectiveImpl(
                TrainSet,
                ConvertTargetsToBool(TrainSet.Targets),
                GamTrainerOptions.LearningRates,
                0,
                _sigmoidParameter,
                GamTrainerOptions.UnbalancedSets,
                GamTrainerOptions.MaxOutput,
                GamTrainerOptions.GetDerivativesSampleRate,
                false,
                GamTrainerOptions.RngSeed,
                ParallelTraining
            );
        }

        private protected override void DefinePruningTest()
        {
            var validTest = new BinaryClassificationTest(ValidSetScore,
                ConvertTargetsToBool(ValidSet.Targets), _sigmoidParameter);
            // As per FastTreeClassification.ConstructOptimizationAlgorithm()
            PruningLossIndex = GamTrainerOptions.UnbalancedSets ? 3 /*Unbalanced  sets  loss*/ : 1 /*normal loss*/;
            PruningTest = new TestHistory(validTest, PruningLossIndex);
        }

        private protected override BinaryPredictionTransformer<CalibratedModelParametersBase<BinaryClassificationGamModelParameters, PlattCalibrator>>
            MakeTransformer(CalibratedModelParametersBase<BinaryClassificationGamModelParameters, PlattCalibrator> model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<CalibratedModelParametersBase<BinaryClassificationGamModelParameters, PlattCalibrator>>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="BinaryClassificationGamTrainer"/> using both training and validation data, returns
        /// a <see cref="BinaryPredictionTransformer{CalibratedModelParametersBase}"/>.
        /// </summary>
        public BinaryPredictionTransformer<CalibratedModelParametersBase<BinaryClassificationGamModelParameters, PlattCalibrator>> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }
    }

    /// <summary>
    /// The model parameters class for Binary Classification GAMs
    /// </summary>
    public sealed class BinaryClassificationGamModelParameters : GamModelParametersBase, IPredictorProducing<float>
    {
        internal const string LoaderSignature = "BinaryClassGamPredictor";
        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        /// <summary>
        /// Construct a new Binary Classification GAM with the defined properties.
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
        internal BinaryClassificationGamModelParameters(IHostEnvironment env,
            double[][] binUpperBounds, double[][] binEffects, double intercept, int inputLength, int[] featureToInputMap)
            : base(env, LoaderSignature, binUpperBounds, binEffects, intercept, inputLength, featureToInputMap) { }

        private BinaryClassificationGamModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx) { }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GAM BINP",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010001, // Added Intercept but collided from release 0.6-0.9
                verWrittenCur: 0x00010002,    // Added Intercept (version revved to address collisions)
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(BinaryClassificationGamModelParameters).Assembly.FullName);
        }

        private static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var predictor = new BinaryClassificationGamModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new SchemaBindableCalibratedModelParameters<BinaryClassificationGamModelParameters, ICalibrator>(env, predictor, calibrator);
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
