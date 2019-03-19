// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(GamBinaryTrainer.Summary,
    typeof(GamBinaryTrainer), typeof(GamBinaryTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    GamBinaryTrainer.UserNameValue,
    GamBinaryTrainer.LoadNameValue,
    GamBinaryTrainer.ShortName, DocName = "trainer/GAM.md")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(GamBinaryModelParameters), null, typeof(SignatureLoadModel),
    "GAM Binary Class Predictor",
    GamBinaryModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a binary classification model with generalized additive models (GAM).
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="GAM_remarks"]/*' />
    public sealed class GamBinaryTrainer :
        GamTrainerBase<GamBinaryTrainer.Options,
        BinaryPredictionTransformer<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>,
        CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>
    {
        /// <summary>
        /// Options for the <see cref="GamBinaryTrainer"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// Whether to use derivatives optimized for unbalanced training data.
            /// </summary>
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
        /// Initializes a new instance of <see cref="GamBinaryTrainer"/>
        /// </summary>
        internal GamBinaryTrainer(IHostEnvironment env, Options options)
             : base(env, options, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            _sigmoidParameter = 1;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="GamBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupColumnName">The name for the column containing the example weight.</param>
        /// <param name="numberOfIterations">The number of iterations to use in learning the features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        /// <param name="maximumBinCountPerFeature">The maximum number of bins to use to approximate features</param>
        internal GamBinaryTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupColumnName = null,
            int numberOfIterations = GamDefaults.NumberOfIterations,
            double learningRate = GamDefaults.LearningRate,
            int maximumBinCountPerFeature = GamDefaults.MaximumBinCountPerFeature)
            : base(env, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(labelColumnName), featureColumnName, rowGroupColumnName, numberOfIterations, learningRate, maximumBinCountPerFeature)
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
        private protected override CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator> TrainModelCore(TrainContext context)
        {
            TrainBase(context);
            var predictor = new GamBinaryModelParameters(Host,
                BinUpperBounds, BinEffects, MeanEffect, InputLength, FeatureMap);
            var calibrator = new PlattCalibrator(Host, -1.0 * _sigmoidParameter, 0);
            return new ValueMapperCalibratedModelParameters<GamBinaryModelParameters, PlattCalibrator>(Host, predictor, calibrator);
        }

        private protected override ObjectiveFunctionBase CreateObjectiveFunction()
        {
            return new FastTreeBinaryTrainer.ObjectiveImpl(
                TrainSet,
                ConvertTargetsToBool(TrainSet.Targets),
                GamTrainerOptions.LearningRate,
                0,
                _sigmoidParameter,
                GamTrainerOptions.UnbalancedSets,
                GamTrainerOptions.MaximumTreeOutput,
                GamTrainerOptions.GetDerivativesSampleRate,
                false,
                GamTrainerOptions.Seed,
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

        private protected override BinaryPredictionTransformer<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>
            MakeTransformer(CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator> model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="GamBinaryTrainer"/> using both training and validation data, returns
        /// a <see cref="BinaryPredictionTransformer{CalibratedModelParametersBase}"/>.
        /// </summary>
        public BinaryPredictionTransformer<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }
    }

    /// <summary>
    /// The model parameters class for Binary Classification GAMs
    /// </summary>
    public sealed class GamBinaryModelParameters : GamModelParametersBase, IPredictorProducing<float>
    {
        internal const string LoaderSignature = "BinaryClassGamPredictor";
        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        /// <summary>
        /// Construct a new Binary Classification GAM with the defined properties.
        /// </summary>
        /// <param name="env">The Host Environment</param>
        /// <param name="binUpperBounds">An array of arrays of bin-upper-bounds for each feature.</param>
        /// <param name="binEffects">An array of arrays of effect sizes for each bin for each feature.</param>
        /// <param name="intercept">The intercept term for the model. Also referred to as the bias or the mean effect.</param>
        /// <param name="inputLength">The number of features passed from the dataset. Used when the number of input features is
        /// different than the number of shape functions. Use default if all features have a shape function.</param>
        /// <param name="featureToInputMap">A map from the feature shape functions, as described by <paramref name="binUpperBounds"/> and <paramref name="binEffects"/>.
        /// to the input feature. Used when the number of input features is different than the number of shape functions. Use default if all features have
        /// a shape function.</param>
        internal GamBinaryModelParameters(IHostEnvironment env,
            double[][] binUpperBounds, double[][] binEffects, double intercept, int inputLength, int[] featureToInputMap)
            : base(env, LoaderSignature, binUpperBounds, binEffects, intercept, inputLength, featureToInputMap) { }

        private GamBinaryModelParameters(IHostEnvironment env, ModelLoadContext ctx)
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
                loaderAssemblyName: typeof(GamBinaryModelParameters).Assembly.FullName);
        }

        private static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var predictor = new GamBinaryModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new SchemaBindableCalibratedModelParameters<GamBinaryModelParameters, ICalibrator>(env, predictor, calibrator);
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
