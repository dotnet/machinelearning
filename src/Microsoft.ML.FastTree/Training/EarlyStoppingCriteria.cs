// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(typeof(TolerantEarlyStoppingCriterion), typeof(TolerantEarlyStoppingCriterion.Options), typeof(SignatureEarlyStoppingCriterion), "Tolerant (TR)", "tr")]
[assembly: LoadableClass(typeof(GLEarlyStoppingCriterion), typeof(GLEarlyStoppingCriterion.Options), typeof(SignatureEarlyStoppingCriterion), "Loss of Generality (GL)", "gl")]
[assembly: LoadableClass(typeof(LPEarlyStoppingCriterion), typeof(LPEarlyStoppingCriterion.Options), typeof(SignatureEarlyStoppingCriterion), "Low Progress (LP)", "lp")]
[assembly: LoadableClass(typeof(PQEarlyStoppingCriterion), typeof(PQEarlyStoppingCriterion.Options), typeof(SignatureEarlyStoppingCriterion), "Generality to Progress Ratio (PQ)", "pq")]
[assembly: LoadableClass(typeof(UPEarlyStoppingCriterion), typeof(UPEarlyStoppingCriterion.Options), typeof(SignatureEarlyStoppingCriterion), "Consecutive Loss in Generality (UP)", "up")]

[assembly: EntryPointModule(typeof(TolerantEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(GLEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(LPEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(PQEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(UPEarlyStoppingCriterion))]

namespace Microsoft.ML.Trainers.FastTree
{
    internal delegate void SignatureEarlyStoppingCriterion(bool lowerIsBetter);

    // These criteria will be used in FastTree and NeuralNets.
    public abstract class IEarlyStoppingCriterion
    {
        /// <summary>
        /// Check if the learning should stop or not.
        /// </summary>
        /// <param name="validationScore">A non negative number. Higher score means better result unless "_lowerIsBetter" is true.</param>
        /// <param name="trainingScore">A non negative number. Higher score means better result unless "_lowerIsBetter" is true.</param>
        /// <param name="isBestCandidate">True if the current result is the best ever.</param>
        /// <returns>If true, the learning should stop.</returns>
        public abstract bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate);
    }

    [TlcModule.ComponentKind("EarlyStoppingCriterion")]
    public interface IEarlyStoppingCriterionFactory : IComponentFactory<bool, IEarlyStoppingCriterion>
    {
        new IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter);
    }

    public abstract class EarlyStoppingCriterion : IEarlyStoppingCriterion
    {
        private float _bestScore;

        protected readonly bool LowerIsBetter;
        protected float BestScore {
            get { return _bestScore; }
            set
            {
                Contracts.Assert((LowerIsBetter && value <= _bestScore) || value >= _bestScore);
                _bestScore = value;
            }
        }

        internal EarlyStoppingCriterion(bool lowerIsBetter)
        {
            LowerIsBetter = lowerIsBetter;
            _bestScore = LowerIsBetter ? float.PositiveInfinity : float.NegativeInfinity;
        }

        /// <summary>
        /// Check if the given score is the best ever. The best score will be stored at this._bestScore.
        /// </summary>
        /// <param name="score">The latest score</param>
        /// <returns>True if the given score is the best ever.</returns>
        protected bool CheckBestScore(float score)
        {
            bool isBestEver = ((score > BestScore) != LowerIsBetter);
            if (isBestEver)
                BestScore = score;

            return isBestEver;
        }
    }

    public sealed class TolerantEarlyStoppingCriterion : EarlyStoppingCriterion
    {
        [TlcModule.Component(FriendlyName = "Tolerant (TR)", Name = "TR", Desc = "Stop if validation score exceeds threshold value.")]
        public sealed class Options : IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance threshold. (Non negative value)", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f)]
            public float Threshold = 0.01f;

            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new TolerantEarlyStoppingCriterion(Threshold, lowerIsBetter);
            }
        }

        public float Threshold { get; }

        public TolerantEarlyStoppingCriterion(float threshold, bool lowerIsBetter = true)
            : base(lowerIsBetter)
        {
            Contracts.CheckUserArg(threshold >= 0, nameof(threshold), "Must be non-negative.");
            Threshold = threshold;
        }

        [BestFriend]
        internal TolerantEarlyStoppingCriterion(Options options, bool lowerIsBetter = true)
            : this(options.Threshold, lowerIsBetter)
        {
        }

        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            if (LowerIsBetter)
                return (validationScore - BestScore > Threshold);
            else
                return (BestScore - validationScore > Threshold);
        }
    }

    // For the detail of the following rules, see the following paper.
    // Lodwich, Aleksander, Yves Rangoni, and Thomas Breuel. "Evaluation of robustness and performance of early stopping rules with multi layer perceptrons."
    // Neural Networks, 2009. IJCNN 2009. International Joint Conference on. IEEE, 2009.

    public abstract class MovingWindowEarlyStoppingCriterion : EarlyStoppingCriterion
    {
        public class Options
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Threshold in range [0,1].", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f, Max = 1.0f)]
            public float Threshold = 0.01f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The window size.", ShortName = "w")]
            [TlcModule.Range(Inf = 0)]
            public int WindowSize = 5;
        }

        public float Threshold { get; }
        public int WindowSize { get; }

        protected Queue<float> PastScores;

        private protected MovingWindowEarlyStoppingCriterion(bool lowerIsBetter, float threshold = 0.01f, int windowSize = 5)
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

        protected bool CheckRecentScores(float score, int windowSize, out float recentBest, out float recentAverage)
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
    public sealed class GLEarlyStoppingCriterion : EarlyStoppingCriterion
    {
        [TlcModule.Component(FriendlyName = "Loss of Generality (GL)", Name = "GL",
                            Desc = "Stop in case of loss of generality.")]
        public sealed class Options : IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Threshold in range [0,1].", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f, Max = 1.0f)]
            public float Threshold = 0.01f;

            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new GLEarlyStoppingCriterion(lowerIsBetter, Threshold);
            }
        }

        public float Threshold { get; }

        public GLEarlyStoppingCriterion(bool lowerIsBetter = true, float threshold = 0.01f) :
            base(lowerIsBetter)
        {
            Contracts.CheckUserArg(0 <= threshold && threshold <= 1, nameof(threshold), "Must be in range [0,1].");
            Threshold = threshold;
        }

        [BestFriend]
        internal GLEarlyStoppingCriterion(Options options, bool lowerIsBetter = true)
            : this(lowerIsBetter, options.Threshold)
        {
        }

        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            if (LowerIsBetter)
                return (validationScore > (1 + Threshold) * BestScore);
            else
                return (validationScore < (1 - Threshold) * BestScore);
        }
    }

    /// <summary>
    /// Low Progress (LP).
    /// This rule fires when the improvements on the score stall.
    /// </summary>
    public sealed class LPEarlyStoppingCriterion : MovingWindowEarlyStoppingCriterion
    {
        [TlcModule.Component(FriendlyName = "Low Progress (LP)", Name = "LP", Desc = "Stops in case of low progress.")]
        public new sealed class Options : MovingWindowEarlyStoppingCriterion.Options, IEarlyStoppingCriterionFactory
        {
            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new LPEarlyStoppingCriterion(lowerIsBetter, Threshold, WindowSize);
            }
        }

        public LPEarlyStoppingCriterion(bool lowerIsBetter, float threshold = 0.01f, int windowSize = 5)
            : base(lowerIsBetter, threshold, windowSize)
        {
        }

        [BestFriend]
        internal LPEarlyStoppingCriterion(Options options, bool lowerIsBetter = true)
            : this(lowerIsBetter, options.Threshold, options.WindowSize)
        {
        }

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
    }

    /// <summary>
    /// Generality to Progress Ratio (PQ).
    /// </summary>
    public sealed class PQEarlyStoppingCriterion : MovingWindowEarlyStoppingCriterion
    {
        [TlcModule.Component(FriendlyName = "Generality to Progress Ratio (PQ)", Name = "PQ", Desc = "Stops in case of generality to progress ration exceeds threshold.")]
        public new sealed class Options : MovingWindowEarlyStoppingCriterion.Options, IEarlyStoppingCriterionFactory
        {
            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new PQEarlyStoppingCriterion(lowerIsBetter, Threshold, WindowSize);
            }
        }

        public PQEarlyStoppingCriterion(bool lowerIsBetter, float threshold = 0.01f, int windowSize = 5)
            : base(lowerIsBetter, threshold, windowSize)
        {
        }

        [BestFriend]
        internal PQEarlyStoppingCriterion(Options options, bool lowerIsBetter = true)
            : this(lowerIsBetter, options.Threshold, options.WindowSize)
        {
        }

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
    }

    /// <summary>
    /// Consecutive Loss in Generality (UP).
    /// </summary>
    public sealed class UPEarlyStoppingCriterion : EarlyStoppingCriterion
    {
        [TlcModule.Component(FriendlyName = "Consecutive Loss in Generality (UP)", Name = "UP",
            Desc = "Stops in case of consecutive loss in generality.")]
        public sealed class Options : IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The window size.", ShortName = "w")]
            [TlcModule.Range(Inf = 0)]
            public int WindowSize = 5;

            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new UPEarlyStoppingCriterion(lowerIsBetter, WindowSize);
            }
        }

        public int WindowSize { get; }
        private int _count;
        private float _prevScore;

        public UPEarlyStoppingCriterion(bool lowerIsBetter, int windowSize = 5)
            : base(lowerIsBetter)
        {
            Contracts.CheckUserArg(windowSize > 0, nameof(windowSize), "Must be positive");
            WindowSize = windowSize;
            _prevScore = LowerIsBetter ? float.PositiveInfinity : float.NegativeInfinity;
        }

        [BestFriend]
        internal UPEarlyStoppingCriterion(Options options, bool lowerIsBetter = true)
            : this(lowerIsBetter, options.WindowSize)
        {
        }

        public override bool CheckScore(float validationScore, float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            _count = ((validationScore < _prevScore) != LowerIsBetter) ? _count + 1 : 0;
            _prevScore = validationScore;

            return (_count >= WindowSize);
        }
    }
}
