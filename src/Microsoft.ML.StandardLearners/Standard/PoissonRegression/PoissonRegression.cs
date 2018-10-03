// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(PoissonRegression.Summary, typeof(PoissonRegression), typeof(PoissonRegression.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    PoissonRegression.UserNameValue,
    PoissonRegression.LoadNameValue,
    "PoissonRegressionNew",
    "Poisson",
    PoissonRegression.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(PoissonRegression), null, typeof(SignatureEntryPointModule), PoissonRegression.LoadNameValue)]

namespace Microsoft.ML.Runtime.Learners
{
    /// <include file='doc.xml' path='doc/members/member[@name="PoissonRegression"]/*' />
    public sealed class PoissonRegression : LbfgsTrainerBase<PoissonRegression.Arguments, RegressionPredictionTransformer<PoissonRegressionPredictor>, PoissonRegressionPredictor>
    {
        internal const string LoadNameValue = "PoissonRegression";
        internal const string UserNameValue = "Poisson Regression";
        internal const string ShortName = "PR";
        internal const string Summary = "Poisson Regression assumes the unknown function, denoted Y has a Poisson distribution.";

        public sealed class Arguments : ArgumentsBase
        {
        }

        private Double _lossNormalizer;

        /// <summary>
        /// Initializes a new instance of <see cref="PoissonRegression"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name for the example weight column.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularizer term.</param>
        /// <param name="l2Weight">Weight of L2 regularizer term.</param>
        /// <param name="memorySize">Memory size for <see cref="LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public PoissonRegression(IHostEnvironment env, string featureColumn, string labelColumn,
            string weightColumn = null,
            float l1Weight = Arguments.Defaults.L1Weight,
            float l2Weight = Arguments.Defaults.L2Weight,
            float optimizationTolerance = Arguments.Defaults.OptTol,
            int memorySize = Arguments.Defaults.MemorySize,
            bool enforceNoNegativity = Arguments.Defaults.EnforceNonNegativity,
            Action<Arguments> advancedSettings = null)
            : base(env, featureColumn, TrainerUtils.MakeR4ScalarLabel(labelColumn), weightColumn, advancedSettings,
                   l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="PoissonRegression"/>
        /// </summary>
        internal PoissonRegression(IHostEnvironment env, Arguments args)
            : base(env, args, TrainerUtils.MakeR4ScalarLabel(args.LabelColumn))
        {
        }

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckRegressionLabel();
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override RegressionPredictionTransformer<PoissonRegressionPredictor> MakeTransformer(PoissonRegressionPredictor model, ISchema trainSchema)
            => new RegressionPredictionTransformer<PoissonRegressionPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        protected override VBuffer<float> InitializeWeightsFromPredictor(PoissonRegressionPredictor srcPredictor)
        {
            Contracts.AssertValue(srcPredictor);
            return InitializeWeights(srcPredictor.Weights2, new[] { srcPredictor.Bias });
        }

        protected override void PreTrainingProcessInstance(float label, ref VBuffer<float> feat, float weight)
        {
            if (!(label >= 0))
                throw Contracts.Except("Poisson regression must regress to a non-negative label, but label {0} encountered", label);
            _lossNormalizer += MathUtils.LogGamma(label + 1);
        }

        // Make sure _lossnormalizer is added only once
        protected override float DifferentiableFunction(ref VBuffer<float> x, ref VBuffer<float> gradient, IProgressChannelProvider progress)
        {
            return base.DifferentiableFunction(ref x, ref gradient, progress) + (float)(_lossNormalizer / NumGoodRows);
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

        protected override float AccumulateOneGradient(ref VBuffer<float> feat, float label, float weight,
            ref VBuffer<float> x, ref VBuffer<float> grad, ref float[] scratch)
        {
            float bias = 0;
            x.GetItemOrDefault(0, ref bias);
            float dot = VectorUtils.DotProductWithOffset(ref x, 1, ref feat) + bias;
            float lambda = MathUtils.ExpSlow(dot);

            float y = label;
            float mult = -(y - lambda) * weight;
            VectorUtils.AddMultWithOffset(ref feat, mult, ref grad, 1);
            // Due to the call to EnsureBiases, we know this region is dense.
            Contracts.Assert(grad.Count >= BiasCount && (grad.IsDense || grad.Indices[BiasCount - 1] == BiasCount - 1));
            grad.Values[0] += mult;
            // From the computer's perspective exp(infinity)==infinity
            // so inf-inf=nan, but in reality, infinity is just a large
            // number we can't represent, and exp(X)-X for X=inf is just inf.
            if (float.IsPositiveInfinity(lambda))
                return float.PositiveInfinity;
            return -(y * dot - lambda) * weight;
        }

        protected override PoissonRegressionPredictor CreatePredictor()
        {
            VBuffer<float> weights = default(VBuffer<float>);
            CurrentWeights.CopyTo(ref weights, 1, CurrentWeights.Length - 1);
            float bias = 0;
            CurrentWeights.GetItemOrDefault(0, ref bias);
            return new PoissonRegressionPredictor(Host, ref weights, bias);
        }

        protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory factory, float loss, int numParams)
        {
            // No-op by design.
        }

        protected override void ProcessPriorDistribution(float label, float weight)
        {
            // No-op by design.
        }

        [TlcModule.EntryPoint(Name = "Trainers.PoissonRegressor",
            Desc = "Train an Poisson regression model.",
            UserName = UserNameValue,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/PoissonRegression/doc.xml' path='doc/members/member[@name=""PoissonRegression""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/PoissonRegression/doc.xml' path='doc/members/example[@name=""PoissonRegression""]/*' />"})]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainPoisson");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.RegressionOutput>(host, input,
                () => new PoissonRegression(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
