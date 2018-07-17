// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(SdcaRegressionTrainer.Summary, typeof(SdcaRegressionTrainer), typeof(SdcaRegressionTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SdcaRegressionTrainer.UserNameValue,
    SdcaRegressionTrainer.LoadNameValue,
    SdcaRegressionTrainer.ShortName)]

namespace Microsoft.ML.Runtime.Learners
{
    using TScalarPredictor = IPredictorWithFeatureWeights<Float>;

    /// <include file='doc.xml' path='doc/members/member[@name="SDCA"]/*' />
    public sealed class SdcaRegressionTrainer : SdcaTrainerBase<IPredictor>, ITrainer<RoleMappedData, TScalarPredictor>, ITrainerEx
    {
        public const string LoadNameValue = "SDCAR";
        public const string UserNameValue = "Fast Linear Regression (SA-SDCA)";
        public const string ShortName = "sasdcar";
        internal const string Summary = "The SDCA linear regression trainer.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportSdcaRegressionLossFactory LossFunction = new SquaredLossFactory();

            public Arguments()
            {
                // Using a higher default tolerance for better RMS.
                ConvergenceTolerance = 0.01f;

                // Default to use unregularized bias in regression.
                BiasLearningRate = 1;
            }
        }

        private readonly ISupportSdcaRegressionLoss _loss;
        private readonly Arguments _args;

        public override PredictionKind PredictionKind { get { return PredictionKind.Regression; } }

        public override bool NeedCalibration { get { return false; } }

        protected override int WeightArraySize { get { return 1; } }

        public SdcaRegressionTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
            _loss = args.LossFunction.CreateComponent(env);
            base.Loss = _loss;
            NeedShuffle = args.Shuffle;
            _args = args;
        }

        public override IPredictor CreatePredictor()
        {
            Contracts.Assert(WeightArraySize == 1);
            Contracts.Assert(Utils.Size(Weights) == 1);
            Contracts.Assert(Utils.Size(Bias) == 1);
            Host.Check(Weights[0].Length > 0);
            VBuffer<Float> maybeSparseWeights = VBufferUtils.CreateEmpty<Float>(Weights[0].Length);
            VBufferUtils.CreateMaybeSparseCopy(ref Weights[0], ref maybeSparseWeights, Conversions.Instance.GetIsDefaultPredicate<Float>(NumberType.Float));
            return new LinearRegressionPredictor(Host, ref maybeSparseWeights, Bias[0]);
        }

        TScalarPredictor ITrainer<RoleMappedData, TScalarPredictor>.CreatePredictor()
        {
            var predictor = CreatePredictor() as TScalarPredictor;
            Contracts.AssertValue(predictor);
            return predictor;
        }

        protected override Float GetInstanceWeight(FloatLabelCursor cursor)
        {
            return cursor.Weight;
        }

        protected override void CheckLabel(RoleMappedData examples)
        {
            examples.CheckRegressionLabel();
        }

        // REVIEW: No extra benefits from using more threads in training.
        protected override int ComputeNumThreads(FloatLabelCursor.Factory cursorFactory)
        {
            int maxThreads;
            if (Host.ConcurrencyFactor < 1)
                maxThreads = Math.Min(2, Math.Max(1, Environment.ProcessorCount / 2));
            else
                maxThreads = Host.ConcurrencyFactor;

            return maxThreads;
        }

        // Using a different logic for default L2 parameter in regression.
        protected override Float TuneDefaultL2(IChannel ch, int maxIterations, long rowCount, int numThreads)
        {
            Contracts.AssertValue(ch);
            Contracts.Assert(maxIterations > 0);
            Contracts.Assert(rowCount > 0);
            Contracts.Assert(numThreads > 0);
            Float l2;

            if (rowCount > 10000)
                l2 = 1e-04f;
            else if (rowCount < 200)
                l2 = 1e-02f;
            else
                l2 = 1e-03f;

            ch.Info("Auto-tuning parameters: L2 = {0}.", l2);
            return l2;
        }
    }

    /// <summary>
    ///The Entry Point for the SDCA regressor.
    /// </summary>
    public static partial class Sdca
    {
        [TlcModule.EntryPoint(Name = "Trainers.StochasticDualCoordinateAscentRegressor",
            Desc = SdcaRegressionTrainer.Summary,
            UserName = SdcaRegressionTrainer.UserNameValue,
            ShortName = SdcaRegressionTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/doc.xml' path='doc/members/member[@name=""SDCA""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/doc.xml' path='doc/members/example[@name=""StochasticDualCoordinateAscentRegressor""]/*' />" })]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, SdcaRegressionTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSDCA");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<SdcaRegressionTrainer.Arguments, CommonOutputs.RegressionOutput>(host, input,
                () => new SdcaRegressionTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }
}