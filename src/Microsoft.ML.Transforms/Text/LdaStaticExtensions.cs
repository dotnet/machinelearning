// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.TextAnalytics;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Information on the result of fitting a LDA transform.
    /// </summary>
    public sealed class LdaFitResult
    {
        /// <summary>
        /// For user defined delegates that accept instances of the containing type.
        /// </summary>
        /// <param name="result"></param>
        public delegate void OnFit(LdaFitResult result);

        public LdaTransformer.LdaState LdaState;
        public LdaFitResult(LdaTransformer.LdaState state)
        {
            LdaState = state;
        }
    }

    public static class LdaStaticExtensions
    {
        private struct Config
        {
            public readonly int NumTopic;
            public readonly Single AlphaSum;
            public readonly Single Beta;
            public readonly int MHStep;
            public readonly int NumIter;
            public readonly int LikelihoodInterval;
            public readonly int NumThread;
            public readonly int NumMaxDocToken;
            public readonly int NumSummaryTermPerTopic;
            public readonly int NumBurninIter;
            public readonly bool ResetRandomGenerator;

            public readonly Action<LdaTransformer.LdaState> OnFit;

            public Config(int numTopic, Single alphaSum, Single beta, int mhStep, int numIter, int likelihoodInterval,
                int numThread, int numMaxDocToken, int numSummaryTermPerTopic, int numBurninIter, bool resetRandomGenerator,
                Action<LdaTransformer.LdaState> onFit)
            {
                NumTopic = numTopic;
                AlphaSum = alphaSum;
                Beta = beta;
                MHStep = mhStep;
                NumIter = numIter;
                LikelihoodInterval = likelihoodInterval;
                NumThread = numThread;
                NumMaxDocToken = numMaxDocToken;
                NumSummaryTermPerTopic = numSummaryTermPerTopic;
                NumBurninIter = numBurninIter;
                ResetRandomGenerator = resetRandomGenerator;

                OnFit = onFit;
            }
        }

        private static Action<LdaTransformer.LdaState> Wrap(LdaFitResult.OnFit onFit)
        {
            if (onFit == null)
                return null;

            return state => onFit(new LdaFitResult(state));
        }

        private interface ILdaCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplVector : Vector<float>, ILdaCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new LdaTransformer.ColumnInfo[toOutput.Length];
                Action<LdaTransformer> onFit = null;
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (ILdaCol)toOutput[i];

                    infos[i] = new LdaTransformer.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]],
                        tcol.Config.NumTopic,
                        tcol.Config.AlphaSum,
                        tcol.Config.Beta,
                        tcol.Config.MHStep,
                        tcol.Config.NumIter,
                        tcol.Config.LikelihoodInterval,
                        tcol.Config.NumThread,
                        tcol.Config.NumMaxDocToken,
                        tcol.Config.NumSummaryTermPerTopic,
                        tcol.Config.NumBurninIter,
                        tcol.Config.ResetRandomGenerator);

                    if (tcol.Config.OnFit != null)
                    {
                        int ii = i; // Necessary because if we capture i that will change to toOutput.Length on call.
                        onFit += tt => tcol.Config.OnFit(tt.GetLdaState(ii));
                    }
                }

                var est = new LdaEstimator(env, infos);
                if (onFit == null)
                    return est;

                return est.WithOnFitDelegate(onFit);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="numTopic">The number of topics in the LDA.</param>
        /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
        /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
        /// <param name="mhstep">Number of Metropolis Hasting step.</param>
        /// <param name="numIterations">Number of iterations.</param>
        /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
        /// <param name="numThreads">The number of training threads. Default value depends on number of logical processors.</param>
        /// <param name="numMaxDocToken">The threshold of maximum count of tokens per doc.</param>
        /// <param name="numSummaryTermPerTopic">The number of words to summarize the topic.</param>
        /// <param name="numBurninIterations">The number of burn-in iterations.</param>
        /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
        /// <param name="onFit">Called upon fitting with the learnt enumeration on the dataset.</param>
        public static Vector<float> ToLdaTopicVector(this Vector<float> input,
            int numTopic = LdaEstimator.Defaults.NumTopic,
            Single alphaSum = LdaEstimator.Defaults.AlphaSum,
            Single beta = LdaEstimator.Defaults.Beta,
            int mhstep = LdaEstimator.Defaults.Mhstep,
            int numIterations = LdaEstimator.Defaults.NumIterations,
            int likelihoodInterval = LdaEstimator.Defaults.LikelihoodInterval,
            int numThreads = LdaEstimator.Defaults.NumThreads,
            int numMaxDocToken = LdaEstimator.Defaults.NumMaxDocToken,
            int numSummaryTermPerTopic = LdaEstimator.Defaults.NumSummaryTermPerTopic,
            int numBurninIterations = LdaEstimator.Defaults.NumBurninIterations,
            bool resetRandomGenerator = LdaEstimator.Defaults.ResetRandomGenerator,
            LdaFitResult.OnFit onFit = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector(input,
                new Config(numTopic, alphaSum, beta, mhstep, numIterations, likelihoodInterval, numThreads, numMaxDocToken, numSummaryTermPerTopic,
                numBurninIterations, resetRandomGenerator, Wrap(onFit)));
        }
    }
}