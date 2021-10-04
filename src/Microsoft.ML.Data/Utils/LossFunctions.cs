// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(LogLoss.Summary, typeof(LogLoss), null, typeof(SignatureClassificationLoss),
    "Log Loss", "LogLoss", "Logistic", "CrossEntropy")]

[assembly: LoadableClass(HingeLoss.Summary, typeof(HingeLoss), typeof(HingeLoss.Options), typeof(SignatureClassificationLoss),
    "Hinge Loss", "HingeLoss", "Hinge")]

[assembly: LoadableClass(SmoothedHingeLoss.Summary, typeof(SmoothedHingeLoss), typeof(SmoothedHingeLoss.Options), typeof(SignatureClassificationLoss),
    "Smoothed Hinge Loss", "SmoothedHingeLoss", "SmoothedHinge")]

[assembly: LoadableClass(ExpLoss.Summary, typeof(ExpLoss), typeof(ExpLoss.Options), typeof(SignatureClassificationLoss),
    "Exponential Loss", "ExpLoss", "Exp")]

[assembly: LoadableClass(SquaredLoss.Summary, typeof(SquaredLoss), null, typeof(SignatureRegressionLoss),
    "Squared Loss", "SquaredLoss", "L2")]

[assembly: LoadableClass(PoissonLoss.Summary, typeof(PoissonLoss), null, typeof(SignatureRegressionLoss),
    "Poisson Loss", "PoissonLoss", "Poisson")]

[assembly: LoadableClass(TweedieLoss.Summary, typeof(TweedieLoss), typeof(TweedieLoss.Options), typeof(SignatureRegressionLoss),
    "Tweedie Loss", "TweedieLoss", "Tweedie", "Tw")]

[assembly: EntryPointModule(typeof(ExpLoss.Options))]
[assembly: EntryPointModule(typeof(LogLossFactory))]
[assembly: EntryPointModule(typeof(HingeLoss.Options))]
[assembly: EntryPointModule(typeof(PoissonLossFactory))]
[assembly: EntryPointModule(typeof(SmoothedHingeLoss.Options))]
[assembly: EntryPointModule(typeof(SquaredLossFactory))]
[assembly: EntryPointModule(typeof(TweedieLoss.Options))]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// The loss function may know the close-form solution to the optimal dual update
    /// Ref: Sec(6.2) of http://jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf
    /// </summary>
    public interface ISupportSdcaLoss : IScalarLoss
    {
        //This method helps the optimizer pre-compute the invariants that will be used later in DualUpdate.
        //scaledFeaturesNormSquared = instanceWeight * (|x|^2 + 1) / (lambda * n), where
        //  - x is the features vector
        //  - lambda is the L2 const
        //  - n is the number of instances
        //    Note that if we are going to implement Online-DCA then n = t and varies.
        float ComputeDualUpdateInvariant(float scaledFeaturesNormSquared);

        /// <summary>
        /// Compute the dual update (\Delta\alpha_i) in SDCA
        /// - alpha: dual variable at the specified instance
        /// - lambdaN: L2 const x number of instances
        /// - cached invariant, hinted by the method above
        /// </summary>
        float DualUpdate(float output, float label, float dual, float invariant, int maxNumThreads);

        /// <summary>
        /// The dual loss function for a training example.
        /// If f(x) denotes the loss function on an individual training example,
        /// then this function returns -f*(-x*), where f*(x*) is the Fenchel conjugate
        /// of f(x).
        /// </summary>
        /// <param name="label">The label of the example.</param>
        /// <param name="dual">The dual variable of the example.</param>
        Double DualLoss(float label, float dual);
    }

    public interface ISupportSdcaClassificationLoss : ISupportSdcaLoss, IClassificationLoss
    {
    }

    public interface ISupportSdcaRegressionLoss : ISupportSdcaLoss, IRegressionLoss
    {
    }

    [TlcModule.ComponentKind("SDCAClassificationLossFunction")]
    [BestFriend]
    internal interface ISupportSdcaClassificationLossFactory : IComponentFactory<ISupportSdcaClassificationLoss>
    {
    }

    [TlcModule.ComponentKind("SDCARegressionLossFunction")]
    [BestFriend]
    internal interface ISupportSdcaRegressionLossFactory : IComponentFactory<ISupportSdcaRegressionLoss>
    {
        new ISupportSdcaRegressionLoss CreateComponent(IHostEnvironment env);
    }

    [TlcModule.Component(Name = "LogLoss", FriendlyName = "Log loss", Aliases = new[] { "Logistic", "CrossEntropy" },
        Desc = "Log loss.")]
    [BestFriend]
    internal sealed class LogLossFactory : ISupportSdcaClassificationLossFactory, ISupportClassificationLossFactory
    {
        public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env) => new LogLoss();

        IClassificationLoss IComponentFactory<IClassificationLoss>.CreateComponent(IHostEnvironment env) => new LogLoss();
    }

    /// <summary>
    /// The Log Loss, also known as the Cross Entropy Loss. It is commonly used in classification tasks.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// The Log Loss function is defined as:
    ///
    /// $L(p(\hat{y}), y) = -y ln(\hat{y}) - (1 - y) ln(1 - \hat{y})$
    ///
    /// where $\hat{y}$ is the predicted score, $p(\hat{y})$ is the probability of belonging to the positive class by applying a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to the score, and $y \in \\{0, 1\\}$ is the true label.
    ///
    /// Note that the labels used in this calculation are 0 and 1, unlike [Hinge Loss](xref:Microsoft.ML.Trainers.HingeLoss) and [Exponential Loss](xref:Microsoft.ML.Trainers.ExpLoss), where the labels used are -1 and 1.
    ///
    /// The Log Loss function provides a measure of how *certain* a classifier's predictions are, instead of just measuring how *correct* they are.
    /// For example, a predicted probability of 0.80 for a true label of 1 gets penalized more than a predicted probability of 0.99.
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    public sealed class LogLoss : ISupportSdcaClassificationLoss
    {
        internal const string Summary = "The log loss function for classification. Supported by SDCA.";
        private const float Threshold = 0.5f;

        public Double Loss(float output, float label)
        {
            float prediction = MathUtils.Sigmoid(output);
            return label > 0 ? -Log(prediction) : -Log(1 - prediction);
        }

        public float Derivative(float output, float label)
        {
            float prediction = MathUtils.Sigmoid(output);
            return label > 0 ? prediction - 1 : prediction;
        }

        public float ComputeDualUpdateInvariant(float scaledFeaturesNormSquared)
        {
            return 1 / Math.Max(1, (float)0.25 + scaledFeaturesNormSquared);
        }

        // REVIEW: this dual update uses a different log loss formulation,
        //although the two are equivalents if the labels are restricted to 0 and 1
        //Need to update so that it can handle probability label and true to the
        //definition, which is a smooth loss function
        public float DualUpdate(float output, float label, float dual, float invariant, int maxNumThreads)
        {
            label = label > 0 ? 1 : -1;

            // This is an approximate solution
            // REVIEW: is it necessary to do a few extra Newton steps?
            var fullUpdate = (MathUtils.Sigmoid(-label * output) * label - dual) * invariant;
            return maxNumThreads >= 2 && Math.Abs(fullUpdate) > Threshold ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(float label, float dual)
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
    /// Hinge Loss, commonly used in classification tasks.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// The Hinge Loss function is defined as:
    ///
    /// $L(\hat{y}, y) = max(0, m - y\hat{y})$
    ///
    /// where $\hat{y}$ is the predicted score, $y \in \\{-1, 1\\}$ is the true label, and $m$ is the margin parameter set to 1 by default.
    ///
    /// Note that the labels used in this calculation are -1 and 1, unlike [Log Loss](xref:Microsoft.ML.Trainers.LogLoss), where the labels used are 0 and 1.
    /// Also unlike [Log Loss](xref:Microsoft.ML.Trainers.LogLoss), $\hat{y}$ is the raw predicted score, not the predicted probability (which is calculated by applying a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to the predicted score).
    ///
    /// While the hinge loss function is both convex and continuous, it is not smooth (that is not differentiable) at $y\hat{y} = m$.
    /// Consequently, it cannot be used with gradient descent methods or stochastic gradient descent methods, which rely on differentiability over the entire domain.
    ///
    /// For more, see [Hinge Loss for classification](https://en.wikipedia.org/wiki/Loss_functions_for_classification#Hinge_loss).
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    public sealed class HingeLoss : ISupportSdcaClassificationLoss
    {
        [TlcModule.Component(Name = "HingeLoss", FriendlyName = "Hinge loss", Alias = "Hinge", Desc = "Hinge loss.")]
        [BestFriend]
        internal sealed class Options : ISupportSdcaClassificationLossFactory, ISupportClassificationLossFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Margin value", ShortName = "marg")]
            public float Margin = Defaults.Margin;

            public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env) => new HingeLoss(this);

            IClassificationLoss IComponentFactory<IClassificationLoss>.CreateComponent(IHostEnvironment env) => new HingeLoss(this);
        }

        internal const string Summary = "The Hinge loss function for classification. Supported by SDCA.";
        private const float Threshold = 0.5f;
        private readonly float _margin;

        private HingeLoss(Options options)
        {
            _margin = options.Margin;
        }

        private static class Defaults
        {
            public const float Margin = 1;
        }

        public HingeLoss(float margin = Defaults.Margin)
            : this(new Options() { Margin = margin })
        {
        }

        public Double Loss(float output, float label)
        {
            float truth = label > 0 ? 1 : -1;
            float loss = _margin - truth * output;
            return loss > 0 ? loss : 0;
        }

        public float Derivative(float output, float label)
        {
            float truth = label > 0 ? 1 : -1;
            return _margin > truth * output ? -truth : 0;
        }

        public float ComputeDualUpdateInvariant(float scaledFeaturesNormSquared)
        {
            return 1 / scaledFeaturesNormSquared;
        }

        public float DualUpdate(float output, float label, float alpha, float invariant, int maxNumThreads)
        {
            float truth = label > 0 ? 1 : -1;
            var tmp = (_margin - output * truth) * invariant + alpha * truth;
            var fullUpdate = truth * Math.Max(0, Math.Min(1, tmp)) - alpha;
            return maxNumThreads >= 2 && Math.Abs(fullUpdate) > Threshold ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(float label, float dual)
        {
            if (label <= 0)
                dual = -dual;

            // The dual variable is out of the feasible region [0, 1].
            if (dual < 0 || dual > 1)
                return Double.NegativeInfinity;

            return _margin * dual;
        }
    }

    /// <summary>
    /// A smooth version of the <see cref="HingeLoss"/> function, commonly used in classification tasks.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// Let $f(\hat{y}, y) = 1 - y\hat{y}$, where $\hat{y}$ is the predicted score and $y \in \\{-1, 1\\}$ is the true label. $f(\hat{y}, y)$ here is the non-zero portion of the [Hinge Loss](xref:Microsoft.ML.Trainers.HingeLoss).
    ///
    /// Note that the labels used in this calculation are -1 and 1, unlike [Log Loss](xref:Microsoft.ML.Trainers.LogLoss), where the labels used are 0 and 1.
    /// Also unlike [Log Loss](xref:Microsoft.ML.Trainers.LogLoss), $\hat{y}$ is the raw predicted score, not the predicted probability (which is calculated by applying a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to the predicted score).
    ///
    /// The Smoothed Hinge Loss function is then defined as:
    ///
    /// $
    /// L(f(\hat{y}, y)) =
    /// \begin{cases}
    /// 0                                  & \text{if } f(\hat{y}, y) < 0 \\\\
    /// \frac{(f(\hat{y}, y))^2}{2\alpha}  & \text{if } f(\hat{y}, y) < \alpha \\\\
    /// f(\hat{y}, y) - \frac{\alpha}{2}   & \text{otherwise}
    /// \end{cases}
    /// $
    ///
    /// where $\alpha$ is a smoothing parameter set to 1 by default.
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    public sealed class SmoothedHingeLoss : ISupportSdcaClassificationLoss
    {
        [TlcModule.Component(Name = "SmoothedHingeLoss", FriendlyName = "Smoothed Hinge Loss", Alias = "SmoothedHinge",
                             Desc = "Smoothed Hinge loss.")]
        internal sealed class Options : ISupportSdcaClassificationLossFactory, ISupportClassificationLossFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Smoothing constant", ShortName = "smooth")]
            public float SmoothingConst = Defaults.SmoothingConst;

            public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env) => new SmoothedHingeLoss(env, this);

            IClassificationLoss IComponentFactory<IClassificationLoss>.CreateComponent(IHostEnvironment env) => new SmoothedHingeLoss(env, this);
        }

        internal const string Summary = "The smooth Hinge loss function for classification. Supported by SDCA.";
        private const float Threshold = 0.5f;
        // The smoothed Hinge loss is 1/(_SmoothParam) smooth (its definition can be found in http://jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf (page 568 Definition 1)
        private readonly float _smoothConst;
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

        private SmoothedHingeLoss(IHostEnvironment env, Options options)
            : this(options.SmoothingConst)
        {
        }

        public Double Loss(float output, float label)
        {
            float truth = label > 0 ? 1 : -1;
            float u = 1 - truth * output;

            if (u < 0)
                return 0;

            if (u < _smoothConst) // u > 1 - _smoothConst
                return u * u / _doubleSmoothConst;

            return u - _halfSmoothConst;
        }

        public float Derivative(float output, float label)
        {
            float truth = label > 0 ? 1 : -1;
            float u = 1 - truth * output;

            if (u < 0)
                return 0;

            if (u < _smoothConst)
                return -truth * u / _smoothConst;

            return -truth;
        }

        public float ComputeDualUpdateInvariant(float scaledFeaturesNormSquared)
        {
            return 1 / (scaledFeaturesNormSquared + _smoothConst);
        }

        public float DualUpdate(float output, float label, float alpha, float invariant, int maxNumThreads)
        {
            float truth = label > 0 ? 1 : -1;
            var tmp = (1 - output * truth - _smoothConst * alpha * truth) * invariant + alpha * truth;
            var fullUpdate = truth * Math.Max(0, Math.Min(1, tmp)) - alpha;
            return maxNumThreads >= 2 && Math.Abs(fullUpdate) > Threshold ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(float label, float dual)
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
    /// Exponential Loss, commonly used in classification tasks.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// The Exponential Loss function is defined as:
    ///
    /// $L(\hat{y}, y) = e^{-\beta y \hat{y}}$
    ///
    /// where $\hat{y}$ is the predicted score, $y \in \\{-1, 1\\}$ is the true label, and $\beta$ is a scale factor set to 1 by default.
    ///
    /// Note that the labels used in this calculation are -1 and 1, unlike [Log Loss](xref:Microsoft.ML.Trainers.LogLoss), where the labels used are 0 and 1.
    /// Also unlike [Log Loss](xref:Microsoft.ML.Trainers.LogLoss), $\hat{y}$ is the raw predicted score, not the predicted probability (which is calculated by applying a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to the predicted score).
    ///
    /// The Exponential Loss function penalizes incorrect predictions more than the [Hinge Loss](xref:Microsoft.ML.Trainers.HingeLoss) and has a larger gradient.
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    public sealed class ExpLoss : IClassificationLoss
    {
        [TlcModule.Component(Name = "ExpLoss", FriendlyName = "Exponential Loss", Desc = "Exponential loss.")]
        internal sealed class Options : ISupportClassificationLossFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Beta (dilation)", ShortName = "beta")]
            public float Beta = 1;

            public IClassificationLoss CreateComponent(IHostEnvironment env) => new ExpLoss(this);
        }

        internal const string Summary = "The exponential loss function for classification.";

        private readonly float _beta;

        internal ExpLoss(Options options)
        {
            _beta = options.Beta;
        }

        public ExpLoss(float beta = 1)
        {
            _beta = beta;
        }

        public Double Loss(float output, float label)
        {
            float truth = label > 0 ? 1 : -1;
            return MathUtils.ExpSlow(-_beta * truth * output);
        }

        public float Derivative(float output, float label)
        {
            float truth = label > 0 ? 1 : -1;
            float factor = -_beta * truth;
            return factor * MathUtils.ExpSlow(factor * output);
        }
    }

    [TlcModule.Component(Name = "SquaredLoss", FriendlyName = "Squared Loss", Alias = "L2", Desc = "Squared loss.")]
    [BestFriend]
    internal sealed class SquaredLossFactory : ISupportSdcaRegressionLossFactory, ISupportRegressionLossFactory
    {
        public ISupportSdcaRegressionLoss CreateComponent(IHostEnvironment env) => new SquaredLoss();

        IRegressionLoss IComponentFactory<IRegressionLoss>.CreateComponent(IHostEnvironment env) => new SquaredLoss();
    }

    /// <summary>
    /// The Squared Loss, commonly used in regression tasks.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// The Squared Loss function is defined as:
    ///
    /// $L(\hat{y}, y) = (\hat{y} - y)^2$
    ///
    /// where $\hat{y}$ is the predicted value and $y$ is the true value.
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    public sealed class SquaredLoss : ISupportSdcaRegressionLoss
    {
        internal const string Summary = "The squared loss function for regression.";

        public Double Loss(float output, float label)
        {
            float diff = output - label;
            return diff * diff;
        }

        public float Derivative(float output, float label)
        {
            float diff = output - label;
            return 2 * diff;
        }

        public float ComputeDualUpdateInvariant(float scaledFeaturesNormSquared)
        {
            return 1 / ((float)0.5 + scaledFeaturesNormSquared);
        }

        public float DualUpdate(float output, float label, float dual, float invariant, int maxNumThreads)
        {
            var fullUpdate = (label - output - (float)0.5 * dual) * invariant;
            return maxNumThreads >= 2 ? fullUpdate / maxNumThreads : fullUpdate;
        }

        public Double DualLoss(float label, float dual)
        {
            return -dual * (dual / 4 - label);
        }
    }

    [TlcModule.Component(Name = "PoissonLoss", FriendlyName = "Poisson Loss", Desc = "Poisson loss.")]
    [BestFriend]
    internal sealed class PoissonLossFactory : ISupportRegressionLossFactory
    {
        public IRegressionLoss CreateComponent(IHostEnvironment env) => new PoissonLoss();
    }

    /// <summary>
    /// Poisson Loss function for Poisson Regression.
    /// </summary>
    /// <remarks type="text/markdown"><![CDATA[
    ///
    /// The Poisson Loss function is defined as:
    ///
    /// $L(\hat{y}, y) = e^{\hat{y}} - y\hat{y}$
    ///
    /// where $\hat{y}$ is the predicted value, $y$ is the true label.
    ///
    /// ]]>
    /// </remarks>
    public sealed class PoissonLoss : IRegressionLoss
    {
        internal const string Summary = "The Poisson loss function for regression.";

        public Double Loss(float output, float label)
        {
            // REVIEW: This is stupid and leads to error whenever this loss is used in an evaluator.
            // The output is in the log-space, while the label is in the original space, while the evaluator
            // has absolutely no way of knowing about this requirement.
            return Math.Exp(output) - label * output;
        }

        public float Derivative(float output, float label)
        {
            return (float)Math.Exp(output) - label;
        }
    }

    /// <summary>
    /// Tweedie loss, based on the log-likelihood of the Tweedie distribution. This loss function is used in Tweedie regression.
    /// </summary>
    /// <remarks type="text/markdown"><![CDATA[
    ///
    /// The Tweedie Loss function is defined as:
    ///
    /// $
    /// L(\hat{y}, y, i) =
    /// \begin{cases}
    /// \hat{y} - y ln(\hat{y}) + ln(\Gamma(y))                                                                                     & \text{if } i = 1 \\\\
    /// \hat{y} + \frac{y}{\hat{y}} - \sqrt{y}                                                                                      & \text{if } i = 2 \\\\
    /// \frac{(\hat{y})^{2 - i}}{2 - i} - y \frac{(\hat{y})^{1 - i}}{1 - i} - (\frac{y^{2 - i}}{2 - i} - y\frac{y^{1 - i}}{1 - i})  & \text{otherwise}
    /// \end{cases}
    /// $
    ///
    /// where $\hat{y}$ is the predicted value, $y$ is the true label, $\Gamma$ is the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function), and $i$ is the index parameter for the [Tweedie distribution](https://en.wikipedia.org/wiki/Tweedie_distribution), in the range [1, 2].
    /// $i$ is set to 1.5 by default. $i = 1$ is Poisson loss, $i = 2$ is gamma loss, and intermediate values are compound Poisson-Gamma loss.
    ///
    /// ]]>
    /// </remarks>
    public sealed class TweedieLoss : IRegressionLoss
    {
        [TlcModule.Component(Name = "TweedieLoss", FriendlyName = "Tweedie Loss", Alias = "tweedie", Desc = "Tweedie loss.")]
        internal sealed class Options : ISupportRegressionLossFactory
        {
            [Argument(ArgumentType.LastOccurrenceWins, HelpText =
                "Index parameter for the Tweedie distribution, in the range [1, 2]. 1 is Poisson loss, 2 is gamma loss, " +
                "and intermediate values are compound Poisson loss.")]
            public Double Index = 1.5;

            public IRegressionLoss CreateComponent(IHostEnvironment env) => new TweedieLoss(this);
        }

        internal const string Summary = "The Tweedie loss function for regression.";

        private readonly Double _index;  // The index parameter specified by the user.
        private readonly Double _index1; // 1 minus the index parameter.
        private readonly Double _index2; // 2 minus the index parameter.

        private TweedieLoss(Options options)
        {
            Contracts.CheckUserArg(1 <= options.Index && options.Index <= 2, nameof(options.Index), "Must be in the range [1, 2]");
            _index = options.Index;
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

        private static void Clamp(ref float val)
        {
            const float eps = (float)1e-7;
            if (val < eps) // I tawt I taw a negwawive wowue.
                val = eps; // I did! I did taw a negwawive wowue!!
        }

        public Double Loss(float output, float label)
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

        public float Derivative(float output, float label)
        {
            Clamp(ref output);
            Clamp(ref label);

            if (_index1 == 0)
                return output - label;
            return (float)(Math.Pow(output, _index2) - label * Math.Pow(output, _index1));
        }
    }
}
