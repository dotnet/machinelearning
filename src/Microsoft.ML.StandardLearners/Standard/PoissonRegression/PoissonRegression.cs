// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
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
    public sealed class PoissonRegression : LbfgsTrainerBase<Float, PoissonRegressionPredictor>
    {
        internal const string LoadNameValue = "PoissonRegression";
        internal const string UserNameValue = "Poisson Regression";
        internal const string ShortName = "PR";
        internal const string Summary = "Poisson Regression assumes the unknown function, denoted Y has a Poisson distribution.";

        public sealed class Arguments : ArgumentsBase
        {
        }

        private Double _lossNormalizer;

        public PoissonRegression(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue)
        {
        }

        public override bool NeedCalibration { get { return false; } }

        public override PredictionKind PredictionKind { get { return PredictionKind.Regression; } }

        protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckRegressionLabel();
        }

        protected override VBuffer<Float> InitializeWeightsFromPredictor(PoissonRegressionPredictor srcPredictor)
        {
            Contracts.AssertValue(srcPredictor);
            return InitializeWeights(srcPredictor.Weights2, new[] { srcPredictor.Bias });
        }

        protected override void PreTrainingProcessInstance(Float label, ref VBuffer<Float> feat, Float weight)
        {
            if (!(label >= 0))
                throw Contracts.Except("Poisson regression must regress to a non-negative label, but label {0} encountered", label);
            _lossNormalizer += MathUtils.LogGamma(label + 1);
        }

        //Make sure _lossnormalizer is added only once
        protected override Float DifferentiableFunction(ref VBuffer<Float> x, ref VBuffer<Float> gradient, IProgressChannelProvider progress)
        {
            return base.DifferentiableFunction(ref x, ref gradient, progress) + (Float)(_lossNormalizer / NumGoodRows);
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

        protected override Float AccumulateOneGradient(ref VBuffer<Float> feat, Float label, Float weight,
            ref VBuffer<Float> x, ref VBuffer<Float> grad, ref Float[] scratch)
        {
            Float bias = 0;
            x.GetItemOrDefault(0, ref bias);
            Float dot = VectorUtils.DotProductWithOffset(ref x, 1, ref feat) + bias;
            Float lambda = MathUtils.ExpSlow(dot);

            Float y = label;
            Float mult = -(y - lambda) * weight;
            VectorUtils.AddMultWithOffset(ref feat, mult, ref grad, 1);
            // Due to the call to EnsureBiases, we know this region is dense.
            Contracts.Assert(grad.Count >= BiasCount && (grad.IsDense || grad.Indices[BiasCount - 1] == BiasCount - 1));
            grad.Values[0] += mult;
            // From the computer's perspective exp(infinity)==infinity
            // so inf-inf=nan, but in reality, infinity is just a large
            // number we can't represent, and exp(X)-X for X=inf is just inf.
            if (Float.IsPositiveInfinity(lambda))
                return Float.PositiveInfinity;
            return -(y * dot - lambda) * weight;
        }

        public override PoissonRegressionPredictor CreatePredictor()
        {
            VBuffer<Float> weights = default(VBuffer<Float>);
            CurrentWeights.CopyTo(ref weights, 1, CurrentWeights.Length - 1);
            Float bias = 0;
            CurrentWeights.GetItemOrDefault(0, ref bias);
            return new PoissonRegressionPredictor(Host, ref weights, bias);
        }

        protected override void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory factory, Float loss, int numParams)
        {
            // No-op by design.
        }

        protected override void ProcessPriorDistribution(Float label, Float weight)
        {
            // No-op by design.
        }

        [TlcModule.EntryPoint(Name = "Trainers.PoissonRegressor", Desc = "Train an Poisson regression model.", UserName = UserNameValue, ShortName = ShortName)]
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
