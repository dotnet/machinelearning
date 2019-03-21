// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(PoissonRegressionTrainer.Summary, typeof(PoissonRegressionTrainer), typeof(PoissonRegressionTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    PoissonRegressionTrainer.UserNameValue,
    PoissonRegressionTrainer.LoadNameValue,
    "PoissonRegressionNew",
    "Poisson",
    PoissonRegressionTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(PoissonRegressionTrainer), null, typeof(SignatureEntryPointModule), PoissonRegressionTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers
{
    /// <include file='doc.xml' path='doc/members/member[@name="PoissonRegression"]/*' />
    public sealed class PoissonRegressionTrainer : LbfgsTrainerBase<PoissonRegressionTrainer.Options, RegressionPredictionTransformer<PoissonRegressionModelParameters>, PoissonRegressionModelParameters>
    {
        internal const string LoadNameValue = "PoissonRegression";
        internal const string UserNameValue = "Poisson Regression";
        internal const string ShortName = "PR";
        internal const string Summary = "Poisson Regression assumes the unknown function, denoted Y has a Poisson distribution.";

        public sealed class Options : OptionsBase
        {
        }

        private Double _lossNormalizer;

        /// <summary>
        /// Initializes a new instance of <see cref="PoissonRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="l1Weight">Weight of L1 regularizer term.</param>
        /// <param name="l2Weight">Weight of L2 regularizer term.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="memorySize">Memory size for <see cref="LogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        internal PoissonRegressionTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = Options.Defaults.L1Regularization,
            float l2Weight = Options.Defaults.L2Regularization,
            float optimizationTolerance = Options.Defaults.OptimizationTolerance,
            int memorySize = Options.Defaults.HistorySize,
            bool enforceNoNegativity = Options.Defaults.EnforceNonNegativity)
            : base(env, featureColumn, TrainerUtils.MakeR4ScalarColumn(labelColumn), weights,
                  l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="PoissonRegressionTrainer"/>
        /// </summary>
        internal PoissonRegressionTrainer(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
        }

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        private protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckRegressionLabel();
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override RegressionPredictionTransformer<PoissonRegressionModelParameters> MakeTransformer(PoissonRegressionModelParameters model, DataViewSchema trainSchema)
            => new RegressionPredictionTransformer<PoissonRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Continues the training of a <see cref="PoissonRegressionTrainer"/> using an already trained <paramref name="linearModel"/> and returns
        /// a <see cref="RegressionPredictionTransformer{PoissonRegressionModelParameters}"/>.
        /// </summary>
        public RegressionPredictionTransformer<PoissonRegressionModelParameters> Fit(IDataView trainData, LinearModelParameters linearModel)
            => TrainTransformer(trainData, initPredictor: linearModel);

        private protected override VBuffer<float> InitializeWeightsFromPredictor(IPredictor srcPredictor)
        {
            var modelParameters = (LinearModelParameters)srcPredictor;
            return InitializeWeights(modelParameters.Weights, new[] { modelParameters.Bias });
        }

        private protected override void PreTrainingProcessInstance(float label, in VBuffer<float> feat, float weight)
        {
            if (!(label >= 0))
                throw Contracts.Except("Poisson regression must regress to a non-negative label, but label {0} encountered", label);
            _lossNormalizer += MathUtils.LogGamma(label + 1);
        }

        // Make sure _lossnormalizer is added only once
        private protected override float DifferentiableFunction(in VBuffer<float> x, ref VBuffer<float> gradient, IProgressChannelProvider progress)
        {
            return base.DifferentiableFunction(in x, ref gradient, progress) + (float)(_lossNormalizer / NumGoodRows);
        }

        // Poisson: p(y;lambda) = lambda^y * exp(-lambda) / y!
        //  lambda is the parameter to the Poisson. It is the mean/expected number of occurrences
        //      p(y;lambda) is the probability that there are y occurences given the expected was lambda
        // Our goal is to maximize log-liklihood. Log(p(y;lambda)) = ylog(lambda) - lambda - log(y!)
        //   lambda = exp(w.x+b)
        //   then dlog(p(y))/dw_i = x_i*y - x_i*lambda = y*x_i - x_i * lambda
        //                  dp/db = y - lambda
        // Goal is to find w that maximizes
        // Note: We negate the above in ordrer to minimize

        private protected override float AccumulateOneGradient(in VBuffer<float> feat, float label, float weight,
            in VBuffer<float> x, ref VBuffer<float> grad, ref float[] scratch)
        {
            float bias = 0;
            x.GetItemOrDefault(0, ref bias);
            float dot = VectorUtils.DotProductWithOffset(in x, 1, in feat) + bias;
            float lambda = MathUtils.ExpSlow(dot);

            float y = label;
            float mult = -(y - lambda) * weight;
            VectorUtils.AddMultWithOffset(in feat, mult, ref grad, 1);
            // Due to the call to EnsureBiases, we know this region is dense.
            var editor = VBufferEditor.CreateFromBuffer(ref grad);
            Contracts.Assert(editor.Values.Length >= BiasCount && (grad.IsDense || editor.Indices[BiasCount - 1] == BiasCount - 1));
            editor.Values[0] += mult;
            // From the computer's perspective exp(infinity)==infinity
            // so inf-inf=nan, but in reality, infinity is just a large
            // number we can't represent, and exp(X)-X for X=inf is just inf.
            if (float.IsPositiveInfinity(lambda))
                return float.PositiveInfinity;
            return -(y * dot - lambda) * weight;
        }

        private protected override PoissonRegressionModelParameters CreatePredictor()
        {
            VBuffer<float> weights = default(VBuffer<float>);
            CurrentWeights.CopyTo(ref weights, 1, CurrentWeights.Length - 1);
            float bias = 0;
            CurrentWeights.GetItemOrDefault(0, ref bias);
            return new PoissonRegressionModelParameters(Host, in weights, bias);
        }

        private protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory factory, float loss, int numParams)
        {
            // No-op by design.
        }

        private protected override void ProcessPriorDistribution(float label, float weight)
        {
            // No-op by design.
        }

        [TlcModule.EntryPoint(Name = "Trainers.PoissonRegressor",
            Desc = "Train an Poisson regression model.",
            UserName = UserNameValue,
            ShortName = ShortName)]
        internal static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainPoisson");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.RegressionOutput>(host, input,
                () => new PoissonRegressionTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }
}
