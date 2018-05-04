// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(TolerantEarlyStoppingCriterion), typeof(TolerantEarlyStoppingCriterion.Arguments), typeof(SignatureEarlyStoppingCriterion), "Tolerant (TR)", "tr")]
[assembly: LoadableClass(typeof(GLEarlyStoppingCriterion), typeof(GLEarlyStoppingCriterion.Arguments), typeof(SignatureEarlyStoppingCriterion), "Loss of Generality (GL)", "gl")]
[assembly: LoadableClass(typeof(LPEarlyStoppingCriterion), typeof(LPEarlyStoppingCriterion.Arguments), typeof(SignatureEarlyStoppingCriterion), "Low Progress (LP)", "lp")]
[assembly: LoadableClass(typeof(PQEarlyStoppingCriterion), typeof(PQEarlyStoppingCriterion.Arguments), typeof(SignatureEarlyStoppingCriterion), "Generality to Progress Ratio (PQ)", "pq")]
[assembly: LoadableClass(typeof(UPEarlyStoppingCriterion), typeof(UPEarlyStoppingCriterion.Arguments), typeof(SignatureEarlyStoppingCriterion), "Consecutive Loss in Generality (UP)", "up")]

[assembly: EntryPointModule(typeof(TolerantEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(GLEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(LPEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(PQEarlyStoppingCriterion))]
[assembly: EntryPointModule(typeof(UPEarlyStoppingCriterion))]

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
    public delegate void SignatureEarlyStoppingCriterion(bool lowerIsBetter);

    // These criteria will be used in FastTree and NeuralNets.
    public interface IEarlyStoppingCriterion
    {
        /// <summary>
        /// Check if the learning should stop or not.
        /// </summary>
        /// <param name="validationScore">A non negative number. Higher score means better result unless "_lowerIsBetter" is true.</param>
        /// <param name="trainingScore">A non negative number. Higher score means better result unless "_lowerIsBetter" is true.</param>
        /// <param name="isBestCandidate">True if the current result is the best ever.</param>
        /// <returns>If true, the learning should stop.</returns>
        bool CheckScore(Float validationScore, Float trainingScore, out bool isBestCandidate);
    }

    [TlcModule.ComponentKind("EarlyStoppingCriterion")]
    public interface IEarlyStoppingCriterionFactory : IComponentFactory<bool, IEarlyStoppingCriterion>
    {
        new IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter);
    }

    public abstract class EarlyStoppingCriterion<TArguments> : IEarlyStoppingCriterion
        where TArguments : EarlyStoppingCriterion<TArguments>.ArgumentsBase
    {
        public abstract class ArgumentsBase { }

        private Float _bestScore;

        protected readonly TArguments Args;
        protected readonly bool LowerIsBetter;
        protected Float BestScore {
            get { return _bestScore; }
            set
            {
                Contracts.Assert((LowerIsBetter && value <= _bestScore) || value >= _bestScore);
                _bestScore = value;
            }
        }

        internal EarlyStoppingCriterion(TArguments args, bool lowerIsBetter)
        {
            Args = args;
            LowerIsBetter = lowerIsBetter;
            _bestScore = LowerIsBetter ? Float.PositiveInfinity : Float.NegativeInfinity;
        }

        public abstract bool CheckScore(Float validationScore, Float trainingScore, out bool isBestCandidate);

        /// <summary>
        /// Check if the given score is the best ever. The best score will be stored at this._bestScore.
        /// </summary>
        /// <param name="score">The latest score</param>
        /// <returns>True if the given score is the best ever.</returns>
        protected bool CheckBestScore(Float score)
        {
            bool isBestEver = ((score > BestScore) != LowerIsBetter);
            if (isBestEver)
                BestScore = score;

            return isBestEver;
        }
    }

    public sealed class TolerantEarlyStoppingCriterion : EarlyStoppingCriterion<TolerantEarlyStoppingCriterion.Arguments>
    {
        [TlcModule.Component(FriendlyName = "Tolerant (TR)", Name = "TR", Desc = "Stop if validation score exceeds threshold value.")]
        public class Arguments : ArgumentsBase, IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance threshold. (Non negative value)", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f)]
            public float Threshold = 0.01f;

            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new TolerantEarlyStoppingCriterion(this, lowerIsBetter);
            }
        }

        public TolerantEarlyStoppingCriterion(Arguments args, bool lowerIsBetter)
            : base(args, lowerIsBetter)
        {
            Contracts.CheckUserArg(Args.Threshold >= 0, nameof(args.Threshold), "Must be non-negative.");
        }

        public override bool CheckScore(Float validationScore, Float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            if (LowerIsBetter)
                return (validationScore - BestScore > Args.Threshold);
            else
                return (BestScore - validationScore > Args.Threshold);
        }
    }

    // For the detail of the following rules, see the following paper.
    // Lodwich, Aleksander, Yves Rangoni, and Thomas Breuel. "Evaluation of robustness and performance of early stopping rules with multi layer perceptrons." 
    // Neural Networks, 2009. IJCNN 2009. International Joint Conference on. IEEE, 2009.

    public abstract class MovingWindowEarlyStoppingCriterion : EarlyStoppingCriterion<MovingWindowEarlyStoppingCriterion.Arguments>
    {
        public class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Threshold in range [0,1].", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f, Max = 1.0f)]
            public Float Threshold = 0.01f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The window size.", ShortName = "w")]
            [TlcModule.Range(Inf = 0)]
            public int WindowSize = 5;
        }

        protected internal Queue<Float> PastScores;

        internal MovingWindowEarlyStoppingCriterion(Arguments args, bool lowerIsBetter)
            : base(args, lowerIsBetter)
        {
            Contracts.CheckUserArg(0 <= Args.Threshold && args.Threshold <= 1, nameof(args.Threshold), "Must be in range [0,1].");
            Contracts.CheckUserArg(Args.WindowSize > 0, nameof(args.WindowSize), "Must be positive.");

            PastScores = new Queue<Float>(Args.WindowSize);
        }

        /// <summary>
        /// Calculate the average score in the given list of scores.
        /// </summary>
        /// <returns>The moving average.</returns>
        private Float GetRecentAvg(Queue<Float> recentScores)
        {
            Float avg = 0;

            foreach (Float score in recentScores)
                avg += score;

            Contracts.Assert(recentScores.Count > 0);
            return avg / recentScores.Count;
        }

        /// <summary>
        /// Get the best score in the given list of scores.
        /// </summary>
        /// <param name="recentScores">The list of scores.</param>
        /// <returns>The best score.</returns>
        private Float GetRecentBest(IEnumerable<Float> recentScores)
        {
            Float recentBestScore = LowerIsBetter ? Float.PositiveInfinity : Float.NegativeInfinity;
            foreach (Float score in recentScores)
            {
                if ((score > recentBestScore) != LowerIsBetter)
                    recentBestScore = score;
            }

            return recentBestScore;
        }

        protected bool CheckRecentScores(Float score, int windowSize, out Float recentBest, out Float recentAverage)
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
                recentBest = default(Float);
                recentAverage = default(Float);
                return false;
            }
        }
    }

    /// <summary>
    /// Loss of Generality (GL).
    /// </summary>
    public sealed class GLEarlyStoppingCriterion : EarlyStoppingCriterion<GLEarlyStoppingCriterion.Arguments>
    {
        [TlcModule.Component(FriendlyName = "Loss of Generality (GL)", Name = "GL",
                            Desc = "Stop in case of loss of generality.")]
        public class Arguments : ArgumentsBase, IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Threshold in range [0,1].", ShortName = "th")]
            [TlcModule.Range(Min = 0.0f, Max = 1.0f)]
            public float Threshold = 0.01f;

            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new GLEarlyStoppingCriterion(this, lowerIsBetter);
            }
        }

        public GLEarlyStoppingCriterion(Arguments args, bool lowerIsBetter)
            : base(args, lowerIsBetter)
        {
            Contracts.CheckUserArg(0 <= Args.Threshold && args.Threshold <= 1, nameof(args.Threshold), "Must be in range [0,1].");
        }

        public override bool CheckScore(Float validationScore, Float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            if (LowerIsBetter)
                return (validationScore > (1 + Args.Threshold) * BestScore);
            else
                return (validationScore < (1 - Args.Threshold) * BestScore);
        }
    }

    /// <summary>
    /// Low Progress (LP).
    /// This rule fires when the improvements on the score stall.
    /// </summary>
    public sealed class LPEarlyStoppingCriterion : MovingWindowEarlyStoppingCriterion
    {
        [TlcModule.Component(FriendlyName = "Low Progress (LP)", Name = "LP", Desc = "Stops in case of low progress.")]
        public new sealed class Arguments : MovingWindowEarlyStoppingCriterion.Arguments, IEarlyStoppingCriterionFactory
        {
            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new LPEarlyStoppingCriterion(this, lowerIsBetter);
            }
        }

        public LPEarlyStoppingCriterion(Arguments args, bool lowerIsBetter)
            : base(args, lowerIsBetter) { }

        public override bool CheckScore(Float validationScore, Float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);
            Contracts.Assert(trainingScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            Float recentBest;
            Float recentAverage;
            if (CheckRecentScores(trainingScore, Args.WindowSize, out recentBest, out recentAverage))
            {
                if (LowerIsBetter)
                    return (recentAverage <= (1 + Args.Threshold) * recentBest);
                else
                    return (recentAverage >= (1 - Args.Threshold) * recentBest);
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
        public new sealed class Arguments : MovingWindowEarlyStoppingCriterion.Arguments, IEarlyStoppingCriterionFactory
        {
            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new PQEarlyStoppingCriterion(this, lowerIsBetter);
            }
        }

        public PQEarlyStoppingCriterion(Arguments args, bool lowerIsBetter)
            : base(args, lowerIsBetter) { }

        public override bool CheckScore(Float validationScore, Float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);
            Contracts.Assert(trainingScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            Float recentBest;
            Float recentAverage;
            if (CheckRecentScores(trainingScore, Args.WindowSize, out recentBest, out recentAverage))
            {
                if (LowerIsBetter)
                    return (validationScore * recentBest >= (1 + Args.Threshold) * BestScore * recentAverage);
                else
                    return (validationScore * recentBest <= (1 - Args.Threshold) * BestScore * recentAverage);
            }

            return false;
        }
    }

    /// <summary>
    /// Consecutive Loss in Generality (UP).
    /// </summary>
    public sealed class UPEarlyStoppingCriterion : EarlyStoppingCriterion<UPEarlyStoppingCriterion.Arguments>
    {
        [TlcModule.Component(FriendlyName = "Consecutive Loss in Generality (UP)", Name = "UP",
            Desc = "Stops in case of consecutive loss in generality.")]
        public sealed class Arguments : ArgumentsBase, IEarlyStoppingCriterionFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The window size.", ShortName = "w")]
            [TlcModule.Range(Inf = 0)]
            public int WindowSize = 5;

            public IEarlyStoppingCriterion CreateComponent(IHostEnvironment env, bool lowerIsBetter)
            {
                return new UPEarlyStoppingCriterion(this, lowerIsBetter);
            }
        }

        private int _count;
        private Float _prevScore;

        public UPEarlyStoppingCriterion(Arguments args, bool lowerIsBetter)
            : base(args, lowerIsBetter)
        {
            Contracts.CheckUserArg(Args.WindowSize > 0, nameof(args.WindowSize), "Must be positive");

            _prevScore = LowerIsBetter ? Float.PositiveInfinity : Float.NegativeInfinity;
        }

        public override bool CheckScore(Float validationScore, Float trainingScore, out bool isBestCandidate)
        {
            Contracts.Assert(validationScore >= 0);

            isBestCandidate = CheckBestScore(validationScore);

            _count = ((validationScore < _prevScore) != LowerIsBetter) ? _count + 1 : 0;
            _prevScore = validationScore;

            return (_count >= Args.WindowSize);
        }
    }
}
