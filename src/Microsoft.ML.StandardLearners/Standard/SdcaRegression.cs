// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(SdcaRegressionTrainer.Summary, typeof(SdcaRegressionTrainer), typeof(SdcaRegressionTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SdcaRegressionTrainer.UserNameValue,
    SdcaRegressionTrainer.LoadNameValue,
    SdcaRegressionTrainer.ShortName)]

namespace Microsoft.ML.Trainers
{
    /// <include file='doc.xml' path='doc/members/member[@name="SDCA"]/*' />
    public sealed class SdcaRegressionTrainer : SdcaTrainerBase<SdcaRegressionTrainer.Arguments, RegressionPredictionTransformer<LinearRegressionPredictor>, LinearRegressionPredictor>
    {
        internal const string LoadNameValue = "SDCAR";
        internal const string UserNameValue = "Fast Linear Regression (SA-SDCA)";
        internal const string ShortName = "sasdcar";
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

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        /// <summary>
        /// Initializes a new instance of <see cref="SdcaRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="labelColumn">The label, or dependent variable.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="weightColumn">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public SdcaRegressionTrainer(IHostEnvironment env,
            string featureColumn,
            string labelColumn,
            string weightColumn = null,
            ISupportSdcaRegressionLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null,
            Action<Arguments> advancedSettings = null)
             : base(env, featureColumn, TrainerUtils.MakeR4ScalarLabel(labelColumn), TrainerUtils.MakeR4ScalarWeightColumn(weightColumn), advancedSettings,
                  l2Const, l1Threshold, maxIterations)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            _loss = loss ?? Args.LossFunction.CreateComponent(env);
            Loss = _loss;
        }

        internal SdcaRegressionTrainer(IHostEnvironment env, Arguments args, string featureColumn, string labelColumn, string weightColumn = null)
            : base(env, args, TrainerUtils.MakeR4ScalarLabel(labelColumn), TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
            Host.CheckValue(labelColumn, nameof(labelColumn));
            Host.CheckValue(featureColumn, nameof(featureColumn));

            _loss = args.LossFunction.CreateComponent(env);
            Loss = _loss;
        }

        internal SdcaRegressionTrainer(IHostEnvironment env, Arguments args)
            : this(env, args, args.FeatureColumn, args.LabelColumn)
        {
        }

        protected override LinearRegressionPredictor CreatePredictor(VBuffer<Float>[] weights, Float[] bias)
        {
            Host.CheckParam(Utils.Size(weights) == 1, nameof(weights));
            Host.CheckParam(Utils.Size(bias) == 1, nameof(bias));
            Host.CheckParam(weights[0].Length > 0, nameof(weights));

            VBuffer<Float> maybeSparseWeights = default;
            // below should be `in weights[0]`, but can't because of https://github.com/dotnet/roslyn/issues/29371
            VBufferUtils.CreateMaybeSparseCopy(weights[0], ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<Float>(NumberType.Float));
            return new LinearRegressionPredictor(Host, in maybeSparseWeights, bias[0]);
        }

        protected override Float GetInstanceWeight(FloatLabelCursor cursor)
        {
            return cursor.Weight;
        }

        protected override void CheckLabel(RoleMappedData examples, out int weightSetCount)
        {
            examples.CheckRegressionLabel();
            weightSetCount = 1;
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

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override RegressionPredictionTransformer<LinearRegressionPredictor> MakeTransformer(LinearRegressionPredictor model, Schema trainSchema)
            => new RegressionPredictionTransformer<LinearRegressionPredictor>(Host, model, trainSchema, FeatureColumn.Name);
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