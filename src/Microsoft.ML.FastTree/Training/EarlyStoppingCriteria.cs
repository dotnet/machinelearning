// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(typeof(TolerantEarlyStoppingRule), typeof(TolerantEarlyStoppingRule.Options), typeof(SignatureEarlyStoppingCriterion), "Tolerant (TR)", "tr")]
[assembly: LoadableClass(typeof(GeneralityLossRule), typeof(GeneralityLossRule.Options), typeof(SignatureEarlyStoppingCriterion), "Loss of Generality (GL)", "gl")]
[assembly: LoadableClass(typeof(LowProgressRule), typeof(LowProgressRule.Options), typeof(SignatureEarlyStoppingCriterion), "Low Progress (LP)", "lp")]
[assembly: LoadableClass(typeof(GeneralityToProgressRatioRule), typeof(GeneralityToProgressRatioRule.Options), typeof(SignatureEarlyStoppingCriterion), "Generality to Progress Ratio (PQ)", "pq")]
[assembly: LoadableClass(typeof(ConsecutiveGeneralityLossRule), typeof(ConsecutiveGeneralityLossRule.Options), typeof(SignatureEarlyStoppingCriterion), "Consecutive Loss in Generality (UP)", "up")]

[assembly: EntryPointModule(typeof(TolerantEarlyStoppingRule))]
[assembly: EntryPointModule(typeof(GeneralityLossRule))]
[assembly: EntryPointModule(typeof(LowProgressRule))]
[assembly: EntryPointModule(typeof(GeneralityToProgressRatioRule))]
[assembly: EntryPointModule(typeof(ConsecutiveGeneralityLossRule))]

[assembly: EntryPointModule(typeof(TolerantEarlyStoppingRule.Options))]
[assembly: EntryPointModule(typeof(GeneralityLossRule.Options))]
[assembly: EntryPointModule(typeof(LowProgressRule.Options))]
[assembly: EntryPointModule(typeof(GeneralityToProgressRatioRule.Options))]
[assembly: EntryPointModule(typeof(ConsecutiveGeneralityLossRule.Options))]

namespace Microsoft.ML.Trainers.FastTree
{
    internal delegate void SignatureEarlyStoppingCriterion(bool lowerIsBetter);

    /// <summary>
    /// Early stopping rule used to terminate training process once meeting a specified criterion.
    /// Used for setting <see cref="EarlyStoppingRule"/> <see cref="BoostedTreeOptions.EarlyStoppingRule"/>.
    /// </summary>
    public abstract class EarlyStoppingRuleBase
    {
        /// <summary>
        /// Check if the learning should stop or not.
        /// </summary>
        /// <param name="validationScore">A non negative number. Higher score means better result unless "_lowerIsBetter" is true.</param>
        /// <param name="trainingScore">A non negative number. Higher score means better result unless "_lowerIsBetter" is true.</param>
        /// <param name="isBestCandidate">True if the current result is the best ever.</param>
        /// <returns>If true, the learning should stop.</returns>
        public abstract bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate);

        /// <summary>
        /// Having <see langword="private protected"/> constructor without parameter prevents derivations of <see cref="EarlyStoppingRuleBase"/> from being
        /// implemented by the public.
        /// </summary>
        private protected EarlyStoppingRuleBase() { }

        /// <summary>
        /// Create <see cref="IEarlyStoppingCriterionFactory"/> for supporting legacy infra built upon <see cref="IComponentFactory"/>.
        /// </summary>
        internal abstract IEarlyStoppingCriterionFactory BuildFactory();
    }

    [BestFriend]
    [TlcModule.ComponentKind("EarlyStoppingCriterion")]
    internal interface IEarlyStoppingCriterionFactory : IComponentFactory<bool, EarlyStoppingRuleBase>
    {
        new EarlyStoppingRuleBase CreateComponent(IHostEnvironment env, bool lowerIsBetter);
    }

    public abstract class EarlyStoppingRule : EarlyStoppingRuleBase
    {
        private float _bestScore;

        /// <summary>
        /// It's <see langword="true"/> if the selected stopping metric should be as low as possible, and <see langword="false"/> otherwise.
        /// </summary>
        private protected bool LowerIsBetter { get; }

        private protected float BestScore {
            get { return _bestScore; }
            set
            {
                Contracts.Assert((LowerIsBetter && value <= _bestScore) || value >= _bestScore);
                _bestScore = value;
            }
        }

        private protected EarlyStoppingRule(bool lowerIsBetter) : base()
        {
            LowerIsBetter = lowerIsBetter;
            _bestScore = LowerIsBetter ? float.PositiveInfinity : float.NegativeInfinity;
        }

        /// <summary>
        /// Lazy constructor. It doesn't initialize anything because in runtime, <see cref="EarlyStoppingRule.EarlyStoppingRule(bool)"/> will be
        /// called inside the training process to initialize needed fields.
        /// </summary>
        private protected EarlyStoppingRule() : base()
        {
        }

        /// <summary>
        /// Check if the given score is the best ever. The best score will be stored at this._bestScore.
        /// </summary>
        /// <param name="score">The latest score</param>
        /// <returns>True if the given score is the best ever.</returns>
        private protected bool CheckBestScore(float score)
        {
            bool isBestEver = ((score > BestScore) != LowerIsBetter);
            if (isBestEver)
                BestScore = score;

            return isBestEver;
        }
    }

    public sealed class TolerantEarlyStoppingRule : EarlyStoppingRule
    {
        [BestFriend]
        [TlcModule.Component(FriendlyName = "Tolerant (TR)", Name = "TR", Desc = "Stop if validation score exceeds threshold value.")]
        internal sealed class Options : IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance threshold. (Non negative value)", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f)]
            public float Threshold = 0.01f;

            public EarlyStoppingRuleBase CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new TolerantEarlyStoppingRule(this, lowerIsBetter);
            }
        }

        /// <summary>
        /// The upper bound of the indicated metric on validation set.
        /// </summary>
        public float Threshold { get; }

        /// <summary>
        /// Create a rule which may terminate the training process if validation score exceeds <paramref name="threshold"/> compared with
        /// the best historical validation score.
        /// </summary>
        /// <param name="threshold">The maximum difference allowed between the (current) validation score and its best historical value.</param>
        public TolerantEarlyStoppingRule(float threshold = 0.01f)
            : base()
        {
            Contracts.CheckUserArg(threshold >= 0, nameof(threshold), "Must be non-negative.");
            Threshold = threshold;
        }

        // Used in command line tool to construct lodable class.
        private TolerantEarlyStoppingRule(Options options, bool lowerIsBetter)
            : base(lowerIsBetter)
        {
            Contracts.CheckUserArg(options.Threshold >= 0, nameof(options.Threshold), "Must be non-negative.");
            Threshold = options.Threshold;
        }

        /// <summary>
        /// See <see cref="EarlyStoppingRuleBase.CheckScore(float, float, out bool)"/>.
        /// </summary>
        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            if (LowerIsBetter)
                return (validationScore - BestScore > Threshold);
            else
                return (BestScore - validationScore > Threshold);
        }

        internal override IEarlyStoppingCriterionFactory BuildFactory() => new Options() { Threshold = Threshold };
    }

    // For the detail of the following rules, see the following paper.
    // Lodwich, Aleksander, Yves Rangoni, and Thomas Breuel. "Evaluation of robustness and performance of early stopping rules with multi layer perceptrons."
    // Neural Networks, 2009. IJCNN 2009. International Joint Conference on. IEEE, 2009.

    public abstract class MovingWindowRule : EarlyStoppingRule
    {
        [BestFriend]
        internal class Options
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Threshold in range [0,1].", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f, Max = 1.0f)]
            public float Threshold = 0.01f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The window size.", ShortName = "w")]
            [TlcModule.Range(Inf = 0)]
            public int WindowSize = 5;
        }

        /// <summary>
        /// A threshold in range [0, 1].
        /// </summary>
        public float Threshold { get; }

        /// <summary>
        /// The number of historical validation scores considered when determining if the training process should stop.
        /// </summary>
        public int WindowSize { get; }

        // Hide this because it's a runtime value.
        private protected Queue<float> PastScores;

        private protected MovingWindowRule(float threshold, int windowSize)
            : base()
        {
            Contracts.CheckUserArg(0 <= threshold && threshold <= 1, nameof(threshold), "Must be in range [0,1].");
            Contracts.CheckUserArg(windowSize > 0, nameof(windowSize), "Must be positive.");

            Threshold = threshold;
            WindowSize = windowSize;
            PastScores = new Queue<float>(windowSize);
        }

        private protected MovingWindowRule(bool lowerIsBetter, float threshold, int windowSize)
            : base(lowerIsBetter)
        {
            Contracts.CheckUserArg(0 <= threshold && threshold <= 1, nameof(threshold), "Must be in range [0,1].");
            Contracts.CheckUserArg(windowSize > 0, nameof(windowSize), "Must be positive.");

            Threshold = threshold;
            WindowSize = windowSize;
            PastScores = new Queue<float>(windowSize);
        }

        /// <summary>
        /// Calculate the average score in the given list of scores.
        /// </summary>
        /// <returns>The moving average.</returns>
        private float GetRecentAvg(Queue<float> recentScores)
        {
            float avg = 0;

            foreach (float score in recentScores)
                avg += score;

            Contracts.Assert(recentScores.Count > 0);
            return avg / recentScores.Count;
        }

        /// <summary>
        /// Get the best score in the given list of scores.
        /// </summary>
        /// <param name="recentScores">The list of scores.</param>
        /// <returns>The best score.</returns>
        private float GetRecentBest(IEnumerable<float> recentScores)
        {
            float recentBestScore = LowerIsBetter ? float.PositiveInfinity : float.NegativeInfinity;
            foreach (float score in recentScores)
            {
                if ((score > recentBestScore) != LowerIsBetter)
                    recentBestScore = score;
            }

            return recentBestScore;
        }

        private protected bool CheckRecentScores(float score, int windowSize, out float recentBest, out float recentAverage)
        {
            if (PastScores.Count >= windowSize)
            {
                PastScores.Dequeue();
                PastScores.Enqueue(score);
                recentAverage = GetRecentAvg(PastScores);
                recentBest = GetRecentBest(PastScores);
                return true;
            }
            else
            {
                PastScores.Enqueue(score);
                recentBest = default(float);
                recentAverage = default(float);
                return false;
            }
        }
    }

    /// <summary>
    /// Loss of Generality (GL).
    /// </summary>
    public sealed class GeneralityLossRule : EarlyStoppingRule
    {
        [BestFriend]
        [TlcModule.Component(FriendlyName = "Loss of Generality (GL)", Name = "GL",
                            Desc = "Stop in case of loss of generality.")]
        internal sealed class Options : IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Threshold in range [0,1].", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f, Max = 1.0f)]
            public float Threshold = 0.01f;

            public EarlyStoppingRuleBase CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new GeneralityLossRule(this, lowerIsBetter);
            }
        }

        /// <summary>
        /// The maximum gap (in percentage such as 0.01 for 1% and 0.5 for 50%) between the (current) validation
        /// score and its best historical value.
        /// </summary>
        public float Threshold { get; }

        /// <summary>
        /// Create a rule which may terminate the training process in case of loss of generality. The loss of generality means
        /// the specified score on validation start increaseing.
        /// </summary>
        /// <param name="threshold">The maximum gap (in percentage such as 0.01 for 1% and 0.5 for 50%) between the (current) validation
        /// score and its best historical value.</param>
        public GeneralityLossRule(float threshold = 0.01f) :
            base()
        {
            Contracts.CheckUserArg(0 <= threshold && threshold <= 1, nameof(threshold), "Must be in range [0,1].");
            Threshold = threshold;
        }

        // Used in command line tool to construct lodable class.
        private GeneralityLossRule(Options options, bool lowerIsBetter)
            : base(lowerIsBetter)
        {
            Contracts.CheckUserArg(0 <= options.Threshold && options.Threshold <= 1, nameof(options.Threshold), "Must be in range [0,1].");
            Threshold = options.Threshold;
        }

        /// <summary>
        /// See <see cref="EarlyStoppingRuleBase.CheckScore(float, float, out bool)"/>.
        /// </summary>
        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            if (LowerIsBetter)
                return (validationScore > (1 + Threshold) * BestScore);
            else
                return (validationScore < (1 - Threshold) * BestScore);
        }

        internal override IEarlyStoppingCriterionFactory BuildFactory() => new Options() { Threshold = Threshold };
}

    /// <summary>
    /// Low Progress (LP).
    /// This rule fires when the improvements on the score stall.
    /// </summary>
    public sealed class LowProgressRule : MovingWindowRule
    {
        [BestFriend]
        [TlcModule.Component(FriendlyName = "Low Progress (LP)", Name = "LP", Desc = "Stops in case of low progress.")]
        internal new sealed class Options : MovingWindowRule.Options, IEarlyStoppingCriterionFactory
        {
            public EarlyStoppingRuleBase CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new LowProgressRule(this, lowerIsBetter);
            }
        }

        /// <summary>
        /// Create a rule which may terminate the training process when the improvements in terms of validation score is slow.
        /// It will terminate the training process if the average of the recent <see cref="MovingWindowRule.WindowSize"/> validation scores
        /// is worse than the best historical validation score.
        /// </summary>
        /// <param name="threshold">The maximum gap (in percentage such as 0.01 for 1% and 0.5 for 50%) between the (current) averaged validation
        /// score and its best historical value.</param>
        /// <param name="windowSize">See <see cref="MovingWindowRule.WindowSize"/>.</param>
        public LowProgressRule(float threshold = 0.01f, int windowSize = 5)
            : base(threshold, windowSize)
        {
        }

        // Used in command line tool to construct lodable class.
        private LowProgressRule(Options options, bool lowerIsBetter)
            : base(lowerIsBetter, options.Threshold, options.WindowSize)
        {
        }

        /// <summary>
        /// See <see cref="EarlyStoppingRuleBase.CheckScore(float, float, out bool)"/>.
        /// </summary>
        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);
            Contracts.Assert(trainingScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            float recentBest;
            float recentAverage;
            if (CheckRecentScores(trainingScore, WindowSize, out recentBest, out recentAverage))
            {
                if (LowerIsBetter)
                    return (recentAverage <= (1 + Threshold) * recentBest);
                else
                    return (recentAverage >= (1 - Threshold) * recentBest);
            }

            return false;
        }

        internal override IEarlyStoppingCriterionFactory BuildFactory() => new Options() { Threshold = Threshold, WindowSize = WindowSize };
}

    /// <summary>
    /// Generality to Progress Ratio (PQ).
    /// </summary>
    public sealed class GeneralityToProgressRatioRule : MovingWindowRule
    {
        [BestFriend]
        [TlcModule.Component(FriendlyName = "Generality to Progress Ratio (PQ)", Name = "PQ", Desc = "Stops in case of generality to progress ration exceeds threshold.")]
        internal new sealed class Options : MovingWindowRule.Options, IEarlyStoppingCriterionFactory
        {
            public EarlyStoppingRuleBase CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new GeneralityToProgressRatioRule(this, lowerIsBetter);
            }
        }

        /// <summary>
        /// Create a rule which may terminate the training process when generality-to-progress ratio exceeds <paramref name="threshold"/>.
        /// </summary>
        /// <param name="threshold">The maximum ratio gap (in percentage such as 0.01 for 1% and 0.5 for 50%).</param>
        /// <param name="windowSize">See <see cref="MovingWindowRule.WindowSize"/>.</param>
        public GeneralityToProgressRatioRule(float threshold = 0.01f, int windowSize = 5)
            : base(threshold, windowSize)
        {
        }

        // Used in command line tool to construct lodable class.
        private GeneralityToProgressRatioRule(Options options, bool lowerIsBetter)
            : base(lowerIsBetter, options.Threshold, options.WindowSize)
        {
        }

        /// <summary>
        /// See <see cref="EarlyStoppingRuleBase.CheckScore(float, float, out bool)"/>.
        /// </summary>
        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);
            Contracts.Assert(trainingScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            float recentBest;
            float recentAverage;
            if (CheckRecentScores(trainingScore, WindowSize, out recentBest, out recentAverage))
            {
                if (LowerIsBetter)
                    return (validationScore * recentBest >= (1 + Threshold) * BestScore * recentAverage);
                else
                    return (validationScore * recentBest <= (1 - Threshold) * BestScore * recentAverage);
            }

            return false;
        }

        [BestFriend]
        internal override IEarlyStoppingCriterionFactory BuildFactory() => new Options() { Threshold = Threshold, WindowSize = WindowSize };
    }

    /// <summary>
    /// Consecutive Loss in Generality (UP).
    /// </summary>
    public sealed class ConsecutiveGeneralityLossRule : EarlyStoppingRule
    {
        [BestFriend]
        [TlcModule.Component(FriendlyName = "Consecutive Loss in Generality (UP)", Name = "UP",
            Desc = "Stops in case of consecutive loss in generality.")]
        internal sealed class Options : IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The window size.", ShortName = "w")]
            [TlcModule.Range(Inf = 0)]
            public int WindowSize = 5;

            public EarlyStoppingRuleBase CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new ConsecutiveGeneralityLossRule(this, lowerIsBetter);
            }
        }

        /// <summary>
        /// The number of historical validation scores considered when determining if the training process should stop.
        /// </summary>
        public int WindowSize { get; }

        private int _count;
        private float _prevScore;

        /// <summary>
        /// Creates a rule which terminates the training process if the validation score is not improved in <see cref="WindowSize"/> consecutive iterations.
        /// </summary>
        /// <param name="windowSize">Number of training iterations allowed to have no improvement.</param>
        public ConsecutiveGeneralityLossRule(int windowSize = 5)
            : base()
        {
            Contracts.CheckUserArg(windowSize > 0, nameof(windowSize), "Must be positive");
            WindowSize = windowSize;
            _prevScore = LowerIsBetter ? float.PositiveInfinity : float.NegativeInfinity;
        }

        private ConsecutiveGeneralityLossRule(Options options, bool lowerIsBetter)
            : base(lowerIsBetter)
        {
            Contracts.CheckUserArg(options.WindowSize > 0, nameof(options.WindowSize), "Must be positive");
            WindowSize = options.WindowSize;
            _prevScore = LowerIsBetter ? float.PositiveInfinity : float.NegativeInfinity;
        }

        /// <summary>
        /// See <see cref="EarlyStoppingRuleBase.CheckScore(float, float, out bool)"/>.
        /// </summary>
        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            _count = ((validationScore < _prevScore) != LowerIsBetter) ? _count + 1 : 0;
            _prevScore = validationScore;

            return (_count >= WindowSize);
        }

        [BestFriend]
        internal override IEarlyStoppingCriterionFactory BuildFactory() => new Options() { WindowSize = WindowSize };
    }
}
