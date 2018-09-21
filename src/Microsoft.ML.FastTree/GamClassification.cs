// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;
using System.Threading.Tasks;

[assembly: LoadableClass(BinaryClassificationGamTrainer.Summary,
    typeof(BinaryClassificationGamTrainer), typeof(BinaryClassificationGamTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    BinaryClassificationGamTrainer.UserNameValue,
    BinaryClassificationGamTrainer.LoadNameValue,
    BinaryClassificationGamTrainer.ShortName, DocName = "trainer/GAM.md")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(BinaryClassGamPredictor), null, typeof(SignatureLoadModel),
    "GAM Binary Class Predictor",
    BinaryClassGamPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.FastTree
{
    public sealed class BinaryClassificationGamTrainer :
    GamTrainerBase<BinaryClassificationGamTrainer.Arguments, BinaryPredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Should we use derivatives optimized for unbalanced sets", ShortName = "us")]
            [TGUI(Label = "Optimize for unbalanced")]
            public bool UnbalancedSets = false;
        }

        internal const string LoadNameValue = "BinaryClassificationGamTrainer";
        internal const string UserNameValue = "Generalized Additive Model for Binary Classification";
        internal const string ShortName = "gam";
        private readonly double _sigmoidParameter;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        private protected override bool NeedCalibration => true;

        /// <summary>
        /// Initializes a new instance of <see cref="BinaryClassificationGamTrainer"/>
        /// </summary>
        internal BinaryClassificationGamTrainer(IHostEnvironment env, Arguments args)
             : base(env, args, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(args.LabelColumn))
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
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public BinaryClassificationGamTrainer(IHostEnvironment env, string labelColumn, string featureColumn, string weightColumn = null, Action<Arguments> advancedSettings = null)
            : base(env, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(labelColumn), featureColumn, weightColumn, advancedSettings)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));

            _sigmoidParameter = 1;
        }

        internal override void CheckLabel(RoleMappedData data)
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

        protected override IPredictorProducing<float> TrainModelCore(TrainContext context)
        {
            TrainBase(context);
            var predictor = new BinaryClassGamPredictor(Host, InputLength, TrainSet,
                MeanEffect, BinEffects, FeatureMap);
            var calibrator = new PlattCalibrator(Host, -1.0 * _sigmoidParameter, 0);
            return new CalibratedPredictor(Host, predictor, calibrator);
        }

        protected override ObjectiveFunctionBase CreateObjectiveFunction()
        {
            return new FastTreeBinaryClassificationTrainer.ObjectiveImpl(
                TrainSet,
                ConvertTargetsToBool(TrainSet.Targets),
                Args.LearningRates,
                0,
                _sigmoidParameter,
                Args.UnbalancedSets,
                Args.MaxOutput,
                Args.GetDerivativesSampleRate,
                false,
                Args.RngSeed,
                ParallelTraining
            );
        }

        protected override void DefinePruningTest()
        {
            var validTest = new BinaryClassificationTest(ValidSetScore,
                ConvertTargetsToBool(ValidSet.Targets), _sigmoidParameter);
            // As per FastTreeClassification.ConstructOptimizationAlgorithm()
            PruningLossIndex = Args.UnbalancedSets ? 3 /*Unbalanced  sets  loss*/ : 1 /*normal loss*/;
            PruningTest = new TestHistory(validTest, PruningLossIndex);
        }

        protected override BinaryPredictionTransformer<IPredictorProducing<float>> MakeTransformer(IPredictorProducing<float> model, ISchema trainSchema)
            => new BinaryPredictionTransformer<IPredictorProducing<float>>(Host, model, trainSchema, FeatureColumn.Name);

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }
    }

    public class BinaryClassGamPredictor : GamPredictorBase, IPredictorProducing<float>
    {
        public const string LoaderSignature = "BinaryClassGamPredictor";
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public BinaryClassGamPredictor(IHostEnvironment env, int inputLength, Dataset trainset,
            double meanEffect, double[][] binEffects, int[] featureMap)
            : base(env, LoaderSignature, inputLength, trainset, meanEffect, binEffects, featureMap) { }

        private BinaryClassGamPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx) { }

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GAM BINP",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var predictor = new BinaryClassGamPredictor(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new SchemaBindableCalibratedPredictor(env, predictor, calibrator);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            base.Save(ctx);
        }
    }
}
