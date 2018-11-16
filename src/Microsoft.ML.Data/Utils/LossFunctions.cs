// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(LogLoss.Summary, typeof(LogLoss), null, typeof(SignatureClassificationLoss),
    "Log Loss", "LogLoss", "Logistic", "CrossEntropy")]

[assembly: LoadableClass(HingeLoss.Summary, typeof(HingeLoss), typeof(HingeLoss.Arguments), typeof(SignatureClassificationLoss),
    "Hinge Loss", "HingeLoss", "Hinge")]

[assembly: LoadableClass(SmoothedHingeLoss.Summary, typeof(SmoothedHingeLoss), typeof(SmoothedHingeLoss.Arguments), typeof(SignatureClassificationLoss),
    "Smoothed Hinge Loss", "SmoothedHingeLoss", "SmoothedHinge")]

[assembly: LoadableClass(ExpLoss.Summary, typeof(ExpLoss), typeof(ExpLoss.Arguments), typeof(SignatureClassificationLoss),
    "Exponential Loss", "ExpLoss", "Exp")]

[assembly: LoadableClass(SquaredLoss.Summary, typeof(SquaredLoss), null, typeof(SignatureRegressionLoss),
    "Squared Loss", "SquaredLoss", "L2")]

[assembly: LoadableClass(PoissonLoss.Summary, typeof(PoissonLoss), null, typeof(SignatureRegressionLoss),
    "Poisson Loss", "PoissonLoss", "Poisson")]

[assembly: LoadableClass(TweedieLoss.Summary, typeof(TweedieLoss), typeof(TweedieLoss.Arguments), typeof(SignatureRegressionLoss),
    "Tweedie Loss", "TweedieLoss", "Tweedie", "Tw")]

[assembly: EntryPointModule(typeof(ExpLoss.Arguments))]
[assembly: EntryPointModule(typeof(LogLossFactory))]
[assembly: EntryPointModule(typeof(HingeLoss.Arguments))]
[assembly: EntryPointModule(typeof(PoissonLossFactory))]
[assembly: EntryPointModule(typeof(SmoothedHingeLoss.Arguments))]
[assembly: EntryPointModule(typeof(SquaredLossFactory))]
[assembly: EntryPointModule(typeof(TweedieLoss.Arguments))]

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// The loss function may know the close-form solution to the optimal dual update
    /// Ref: Sec(6.2) of http://jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf
    /// </summary>
    public interface ISupportSdcaLoss : IScalarOutputLoss
    {
        //This method helps the optimizer pre-compute the invariants that will be used later in DualUpdate.
        //scaledFeaturesNormSquared = instanceWeight * (|x|^2 + 1) / (lambda * n), where
        //  - x is the features vector
        //  - lambda is the L2 const
        //  - n is the number of instances
        //    Note that if we are going to implement Online-DCA then n = t and varies.
        Float ComputeDualUpdateInvariant(Float scaledFeaturesNormSquared);

        /// <summary>
        /// Compute the dual update (\Delta\alpha_i) in SDCA
        /// - alpha: dual variable at the specified instance
        /// - lambdaN: L2 const x number of instances
        /// - cached invariant, hinted by the method above
        /// </summary>
        Float DualUpdate(Float output, Float label, Float dual, Float invariant, int maxNumThreads);

        /// <summary>
        /// The dual loss function for a training example.
        /// If f(x) denotes the loss function on an individual training example,
        /// then this function returns -f*(-x*), where f*(x*) is the Fenchel conjugate
        /// of f(x).
        /// </summary>
        /// <param name="label">The label of the example.</param>
        /// <param name="dual">The dual variable of the example.</param>
        Double DualLoss(Float label, Double dual);
    }

    public interface ISupportSdcaClassificationLoss : ISupportSdcaLoss, IClassificationLoss
    {
    }

    public interface ISupportSdcaRegressionLoss : ISupportSdcaLoss, IRegressionLoss
    {
    }

    [TlcModule.ComponentKind("SDCAClassificationLossFunction")]
    public interface ISupportSdcaClassificationLossFactory : IComponentFactory<ISupportSdcaClassificationLoss>
    {
    }

    [TlcModule.ComponentKind("SDCARegressionLossFunction")]
    public interface ISupportSdcaRegressionLossFactory : IComponentFactory<ISupportSdcaRegressionLoss>
    {
        new ISupportSdcaRegressionLoss CreateComponent(IHostEnvironment env);
    }

    [TlcModule.Component(Name = "LogLoss", FriendlyName = "Log loss", Aliases = new[] { "Logistic", "CrossEntropy" },
        Desc = "Log loss.")]
    public sealed class LogLossFactory : ISupportSdcaClassificationLossFactory, ISupportClassificationLossFactory
    {
        public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env) => new LogLoss();

        IClassificationLoss IComponentFactory<IClassificationLoss>.CreateComponent(IHostEnvironment env) => new LogLoss();
    }

    public sealed class LogLoss : ISupportSdcaClassificationLoss
    {
        internal const string Summary = "The log loss function for classification. Supported by SDCA.";
        private const Float Threshold = 0.5f;

        public Double Loss(Float output, Float label)
        {
            Float prediction = MathUtils.Sigmoid(output);
            return label > 0 ? -Log(prediction) : -Log(1 - prediction);
        }

        public Float Derivative(Float output, Float label)
        {
            Float prediction = MathUtils.Sigmoid(output);
            return label > 0 ? prediction - 1 : prediction;
        }

        public Float ComputeDualUpdateInvariant(Float scaledFeaturesNormSquared)
        {
            return 1 / Math.Max(1, (Float)0.25 + scaledFeaturesNormSquared);
        }

        // REVIEW: this dual update uses a different log loss formulation,
        //although the two are equivalents if the labels are restricted to 0 and 1
        //Need to update so that it can handle probability label and true to the
        //definition, which is a smooth loss function
        public Float DualUpdate(Float output, Float label, Float dual, Float invariant, int maxNumThreads)
        {
            label = label > 0 ? 1 : -1;

            // This is an approximate solution
            // REVIEW: is it necessary to do a few extra Newton steps?
            var fullUpdate = (MathUtils.Sigmoid(-label * output) * label - dual) * invariant;
            return maxNumThreads >= 2 && Math.Abs(fullUpdate) > Threshold ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(Float label, Double dual)
        {
            // Normalize the dual with label.
            if (label <= 0)
                dual = -dual;

            // The dual variable is out of the feasible region [0, 1].
            if (dual < 0 || dual > 1)
                return Double.NegativeInfinity;

            return MathUtils.Entropy(dual, useLnNotLog2: true);
        }

        private static Double Log(Double x)
        {
            return Math.Log(Math.Max(x, 1e-8));
        }
    }

    /// <summary>
    /// Hinge Loss
    /// </summary>
    public sealed class HingeLoss : ISupportSdcaClassificationLoss
    {
        [TlcModule.Component(Name = "HingeLoss", FriendlyName = "Hinge loss", Alias = "Hinge", Desc = "Hinge loss.")]
        public sealed class Arguments : ISupportSdcaClassificationLossFactory, ISupportClassificationLossFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Margin value", ShortName = "marg")]
            public Float Margin = Defaults.Margin;

            public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env) => new HingeLoss(this);

            IClassificationLoss IComponentFactory<IClassificationLoss>.CreateComponent(IHostEnvironment env) => new HingeLoss(this);
        }

        internal const string Summary = "The Hinge loss function for classification. Supported by SDCA.";
        private const Float Threshold = 0.5f;
        private readonly Float _margin;

        internal HingeLoss(Arguments args)
        {
            _margin = args.Margin;
        }

        private static class Defaults
        {
            public const float Margin = 1;
        }

        public HingeLoss(float margin = Defaults.Margin)
            : this(new Arguments() { Margin = margin })
        {
        }

        public Double Loss(Float output, Float label)
        {
            Float truth = label > 0 ? 1 : -1;
            Float loss = _margin - truth * output;
            return loss > 0 ? loss : 0;
        }

        public Float Derivative(Float output, Float label)
        {
            Float truth = label > 0 ? 1 : -1;
            return _margin > truth * output ? -truth : 0;
        }

        public Float ComputeDualUpdateInvariant(Float scaledFeaturesNormSquared)
        {
            return 1 / scaledFeaturesNormSquared;
        }

        public Float DualUpdate(Float output, Float label, Float alpha, Float invariant, int maxNumThreads)
        {
            Float truth = label > 0 ? 1 : -1;
            var tmp = (_margin - output * truth) * invariant + alpha * truth;
            var fullUpdate = truth * Math.Max(0, Math.Min(1, tmp)) - alpha;
            return maxNumThreads >= 2 && Math.Abs(fullUpdate) > Threshold ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(Float label, Double dual)
        {
            if (label <= 0)
                dual = -dual;

            // The dual variable is out of the feasible region [0, 1].
            if (dual < 0 || dual > 1)
                return Double.NegativeInfinity;

            return _margin * dual;
        }
    }

    public sealed class SmoothedHingeLoss : ISupportSdcaClassificationLoss
    {
        [TlcModule.Component(Name = "SmoothedHingeLoss", FriendlyName = "Smoothed Hinge Loss", Alias = "SmoothedHinge",
                             Desc = "Smoothed Hinge loss.")]
        public sealed class Arguments : ISupportSdcaClassificationLossFactory, ISupportClassificationLossFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Smoothing constant", ShortName = "smooth")]
            public Float SmoothingConst = Defaults.SmoothingConst;

            public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env) => new SmoothedHingeLoss(env, this);

            IClassificationLoss IComponentFactory<IClassificationLoss>.CreateComponent(IHostEnvironment env) => new SmoothedHingeLoss(env, this);
        }

        internal const string Summary = "The smooth Hinge loss function for classification. Supported by SDCA.";
        private const Float Threshold = 0.5f;
        // The smoothed Hinge loss is 1/(_SmoothParam) smooth (its definition can be found in http://jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf (page 568 Definition 1)
        private readonly Float _smoothConst;
        private readonly Double _halfSmoothConst;
        private readonly Double _doubleSmoothConst;

        private static class Defaults
        {
            public const float SmoothingConst = 1;
        }

        /// <summary>
        /// Constructor for smoothed hinge losee.
        /// </summary>
        /// <param name="smoothingConstant">The smoothing constant.</param>
        public SmoothedHingeLoss(float smoothingConstant = Defaults.SmoothingConst)
        {
            Contracts.CheckParam(smoothingConstant >= 0, nameof(smoothingConstant), "Must be non-negative.");
            _smoothConst = smoothingConstant;
            _halfSmoothConst = _smoothConst / 2;
            _doubleSmoothConst = _smoothConst * 2;
        }

        private SmoothedHingeLoss(IHostEnvironment env, Arguments args)
            : this(args.SmoothingConst)
        {
        }

        public Double Loss(Float output, Float label)
        {
            Float truth = label > 0 ? 1 : -1;
            Float u = 1 - truth * output;

            if (u < 0)
                return 0;

            if (u < _smoothConst) // u > 1 - _smoothConst
                return u * u / _doubleSmoothConst;

            return u - _halfSmoothConst;
        }

        public Float Derivative(Float output, Float label)
        {
            Float truth = label > 0 ? 1 : -1;
            Float u = 1 - truth * output;

            if (u < 0)
                return 0;

            if (u < _smoothConst)
                return -truth * u / _smoothConst;

            return -truth;
        }

        public Float ComputeDualUpdateInvariant(Float scaledFeaturesNormSquared)
        {
            return 1 / (scaledFeaturesNormSquared + _smoothConst);
        }

        public Float DualUpdate(Float output, Float label, Float alpha, Float invariant, int maxNumThreads)
        {
            Float truth = label > 0 ? 1 : -1;
            var tmp = (1 - output * truth - _smoothConst * alpha * truth) * invariant + alpha * truth;
            var fullUpdate = truth * Math.Max(0, Math.Min(1, tmp)) - alpha;
            return maxNumThreads >= 2 && Math.Abs(fullUpdate) > Threshold ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(Float label, Double dual)
        {
            if (label <= 0)
                dual = -dual;

            // The dual variable is out of the feasible region [0, 1].
            if (dual < 0 || dual > 1)
                return Double.NegativeInfinity;

            return dual * (1 - dual * _halfSmoothConst);
        }
    }

    /// <summary>
    /// Exponential Loss
    /// </summary>
    public sealed class ExpLoss : IClassificationLoss
    {
        [TlcModule.Component(Name = "ExpLoss", FriendlyName = "Exponential Loss", Desc = "Exponential loss.")]
        public sealed class Arguments : ISupportClassificationLossFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Beta (dilation)", ShortName = "beta")]
            public Float Beta = 1;

            public IClassificationLoss CreateComponent(IHostEnvironment env) => new ExpLoss(this);
        }

        internal const string Summary = "The exponential loss function for classification.";

        private readonly Float _beta;

        public ExpLoss(Arguments args)
        {
            _beta = args.Beta;
        }

        public Double Loss(Float output, Float label)
        {
            Float truth = label > 0 ? 1 : -1;
            return MathUtils.ExpSlow(-_beta * truth * output);
        }

        public Float Derivative(Float output, Float label)
        {
            Float truth = label > 0 ? 1 : -1;
            Float factor = -_beta * truth;
            return factor * MathUtils.ExpSlow(factor * output);
        }
    }

    [TlcModule.Component(Name = "SquaredLoss", FriendlyName = "Squared Loss", Alias = "L2", Desc = "Squared loss.")]
    public sealed class SquaredLossFactory : ISupportSdcaRegressionLossFactory, ISupportRegressionLossFactory
    {
        public ISupportSdcaRegressionLoss CreateComponent(IHostEnvironment env) => new SquaredLoss();

        IRegressionLoss IComponentFactory<IRegressionLoss>.CreateComponent(IHostEnvironment env) => new SquaredLoss();
    }

    public sealed class SquaredLoss : ISupportSdcaRegressionLoss
    {
        internal const string Summary = "The squared loss function for regression.";

        public Double Loss(Float output, Float label)
        {
            Float diff = output - label;
            return diff * diff;
        }

        public Float Derivative(Float output, Float label)
        {
            Float diff = output - label;
            return 2 * diff;
        }

        public Float ComputeDualUpdateInvariant(Float scaledFeaturesNormSquared)
        {
            return 1 / ((Float)0.5 + scaledFeaturesNormSquared);
        }

        public Float DualUpdate(Float output, Float label, Float dual, Float invariant, int maxNumThreads)
        {
            var fullUpdate = (label - output - (Float)0.5 * dual) * invariant;
            return maxNumThreads >= 2 ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(Float label, Double dual)
        {
            return -dual * (dual / 4 - label);
        }
    }

    [TlcModule.Component(Name = "PoissonLoss", FriendlyName = "Poisson Loss", Desc = "Poisson loss.")]
    public sealed class PoissonLossFactory : ISupportRegressionLossFactory
    {
        public IRegressionLoss CreateComponent(IHostEnvironment env) => new PoissonLoss();
    }

    /// <summary>
    /// Poisson Loss.
    /// </summary>
    public sealed class PoissonLoss : IRegressionLoss
    {
        internal const string Summary = "The Poisson loss function for regression.";

        public Double Loss(Float output, Float label)
        {
            // REVIEW: This is stupid and leads to error whenever this loss is used in an evaluator.
            // The output is in the log-space, while the label is in the original space, while the evaluator
            // has absolutely no way of knowing about this requirement.
            return Math.Exp(output) - label * output;
        }

        public Float Derivative(Float output, Float label)
        {
            return (Float)Math.Exp(output) - label;
        }
    }

    /// <summary>
    /// Tweedie loss, based on the log-likelihood of the Tweedie distribution.
    /// </summary>
    public sealed class TweedieLoss : IRegressionLoss
    {
        [TlcModule.Component(Name = "TweedieLoss", FriendlyName = "Tweedie Loss", Alias = "tweedie", Desc = "Tweedie loss.")]
        public sealed class Arguments : ISupportRegressionLossFactory
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText =
                "Index parameter for the Tweedie distribution, in the range [1, 2]. 1 is Poisson loss, 2 is gamma loss, " +
                "and intermediate values are compound Poisson loss.")]
            public Double Index = 1.5;

            public IRegressionLoss CreateComponent(IHostEnvironment env) => new TweedieLoss(this);
        }

        internal const string Summary = "The Tweedie loss function for regression.";

        private readonly Double _index;  // The index parameter specified by the user.
        private readonly Double _index1; // 1 minus the index parameter.
        private readonly Double _index2; // 2 minus the index parameter.

        public TweedieLoss(Arguments args)
        {
            Contracts.CheckUserArg(1 <= args.Index && args.Index <= 2, nameof(args.Index), "Must be in the range [1, 2]");
            _index = args.Index;
            _index1 = 1 - _index;
            _index2 = 2 - _index;
        }

        /// <summary>
        /// Constructor for Tweedie loss.
        /// </summary>
        /// <param name="index">Index parameter for the Tweedie distribution, in the range [1, 2].
        /// 1 is Poisson loss, 2 is gamma loss, and intermediate values are compound Poisson loss.</param>
        public TweedieLoss(double index = 1.5)
        {
            Contracts.CheckParam(1 <= index && index <= 2, nameof(index), "Must be in the range [1, 2]");
            _index = index;
            _index1 = 1 - _index;
            _index2 = 2 - _index;
        }

        private static void Clamp(ref Float val)
        {
            const Float eps = (Float)1e-7;
            if (val < eps) // I tawt I taw a negwawive wowue.
                val = eps; // I did! I did taw a negwawive wowue!!
        }

        public Double Loss(Float output, Float label)
        {
            Clamp(ref output);
            Clamp(ref label);

            // This is the log likelihood.
            // If output were in the original log-space, then this would be exp((1-rho) * output).
            // However output is not in the log space.
            if (_index1 == 0)
                return output - label * Math.Log(output) + MathUtils.LogGamma(label);
            // The last log-gamma is not necessary from the point of view of the mathematical models,
            // but to the extent that this loss is used as a human comprehensible value, it's nice to
            // have its minimum at 0.
            if (_index2 == 0)
                return output + label / output - Math.Sqrt(label);

            return Math.Pow(output, _index2) / _index2 - label * Math.Pow(output, _index1) / _index1
                - (Math.Pow(label, _index2) / _index2 - label * Math.Pow(label, _index1) / _index1);
        }

        public Float Derivative(Float output, Float label)
        {
            Clamp(ref output);
            Clamp(ref label);

            if (_index1 == 0)
                return output - label;
            return (Float)(Math.Pow(output, _index2) - label * Math.Pow(output, _index1));
        }
    }
}