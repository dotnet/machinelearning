// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Information on the result of fitting a LDA transform.
    /// </summary>
    public sealed class LatentDirichletAllocationFitResult
    {
        /// <summary>
        /// For user defined delegates that accept instances of the containing type.
        /// </summary>
        /// <param name="result"></param>
        public delegate void OnFit(LatentDirichletAllocationFitResult result);

        public LatentDirichletAllocationTransformer.ModelParameters LdaTopicSummary;
        public LatentDirichletAllocationFitResult(LatentDirichletAllocationTransformer.ModelParameters ldaTopicSummary)
        {
            LdaTopicSummary = ldaTopicSummary;
        }
    }

    public static class LatentDirichletAllocationStaticExtensions
    {
        private struct Config
        {
            public readonly int NumberOfTopics;
            public readonly Single AlphaSum;
            public readonly Single Beta;
            public readonly int SamplingStepCount;
            public readonly int MaximumNumberOfIterations;
            public readonly int LikelihoodInterval;
            public readonly int NumberOfThreads;
            public readonly int MaximumTokenCountPerDocument;
            public readonly int NumberOfSummaryTermsPerTopic;
            public readonly int NumberOfBurninIterations;
            public readonly bool ResetRandomGenerator;

            public readonly Action<LatentDirichletAllocationTransformer.ModelParameters> OnFit;

            public Config(int numberOfTopics, Single alphaSum, Single beta, int samplingStepCount, int maximumNumberOfIterations, int likelihoodInterval,
                int numberOfThreads, int maximumTokenCountPerDocument, int numberOfSummaryTermsPerTopic, int numberOfBurninIterations, bool resetRandomGenerator,
                Action<LatentDirichletAllocationTransformer.ModelParameters> onFit)
            {
                NumberOfTopics = numberOfTopics;
                AlphaSum = alphaSum;
                Beta = beta;
                SamplingStepCount = samplingStepCount;
                MaximumNumberOfIterations = maximumNumberOfIterations;
                LikelihoodInterval = likelihoodInterval;
                NumberOfThreads = numberOfThreads;
                MaximumTokenCountPerDocument = maximumTokenCountPerDocument;
                NumberOfSummaryTermsPerTopic = numberOfSummaryTermsPerTopic;
                NumberOfBurninIterations = numberOfBurninIterations;
                ResetRandomGenerator = resetRandomGenerator;

                OnFit = onFit;
            }
        }

        private static Action<LatentDirichletAllocationTransformer.ModelParameters> Wrap(LatentDirichletAllocationFitResult.OnFit onFit)
        {
            if (onFit == null)
                return null;

            return ldaTopicSummary => onFit(new LatentDirichletAllocationFitResult(ldaTopicSummary));
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
                var infos = new LatentDirichletAllocationEstimator.ColumnOptions[toOutput.Length];
                Action<LatentDirichletAllocationTransformer> onFit = null;
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (ILdaCol)toOutput[i];

                    infos[i] = new LatentDirichletAllocationEstimator.ColumnOptions(outputNames[toOutput[i]],
                        inputNames[tcol.Input],
                        tcol.Config.NumberOfTopics,
                        tcol.Config.AlphaSum,
                        tcol.Config.Beta,
                        tcol.Config.SamplingStepCount,
                        tcol.Config.MaximumNumberOfIterations,
                        tcol.Config.LikelihoodInterval,
                        tcol.Config.NumberOfThreads,
                        tcol.Config.MaximumTokenCountPerDocument,
                        tcol.Config.NumberOfSummaryTermsPerTopic,
                        tcol.Config.NumberOfBurninIterations,
                        tcol.Config.ResetRandomGenerator);

                    if (tcol.Config.OnFit != null)
                    {
                        int ii = i; // Necessary because if we capture i that will change to toOutput.Length on call.
                        onFit += tt => tcol.Config.OnFit(tt.GetLdaDetails(ii));
                    }
                }

                var est = new LatentDirichletAllocationEstimator(env, infos);
                if (onFit == null)
                    return est;

                return est.WithOnFitDelegate(onFit);
            }
        }

        /// <include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="input">A vector of floats representing the document.</param>
        /// <param name="numberOfTopics">The number of topics.</param>
        /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
        /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
        /// <param name="samplingStepCount">Number of Metropolis Hasting step.</param>
        /// <param name="maximumNumberOfIterations">Number of iterations.</param>
        /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
        /// <param name="numberOfThreads">The number of training threads. Default value depends on number of logical processors.</param>
        /// <param name="maximumTokenCountPerDocument">The threshold of maximum count of tokens per doc.</param>
        /// <param name="numberOfSummaryTermsPerTopic">The number of words to summarize the topic.</param>
        /// <param name="numberOfBurninIterations">The number of burn-in iterations.</param>
        /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
        /// <param name="onFit">Called upon fitting with the learnt enumeration on the dataset.</param>
        public static Vector<float> LatentDirichletAllocation(this Vector<float> input,
            int numberOfTopics = LatentDirichletAllocationEstimator.Defaults.NumberOfTopics,
            Single alphaSum = LatentDirichletAllocationEstimator.Defaults.AlphaSum,
            Single beta = LatentDirichletAllocationEstimator.Defaults.Beta,
            int samplingStepCount = LatentDirichletAllocationEstimator.Defaults.SamplingStepCount,
            int maximumNumberOfIterations = LatentDirichletAllocationEstimator.Defaults.MaximumNumberOfIterations,
            int likelihoodInterval = LatentDirichletAllocationEstimator.Defaults.LikelihoodInterval,
            int numberOfThreads = LatentDirichletAllocationEstimator.Defaults.NumberOfThreads,
            int maximumTokenCountPerDocument = LatentDirichletAllocationEstimator.Defaults.MaximumTokenCountPerDocument,
            int numberOfSummaryTermsPerTopic = LatentDirichletAllocationEstimator.Defaults.NumberOfSummaryTermsPerTopic,
            int numberOfBurninIterations = LatentDirichletAllocationEstimator.Defaults.NumberOfBurninIterations,
            bool resetRandomGenerator = LatentDirichletAllocationEstimator.Defaults.ResetRandomGenerator,
            LatentDirichletAllocationFitResult.OnFit onFit = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector(input,
                new Config(numberOfTopics, alphaSum, beta, samplingStepCount, maximumNumberOfIterations, likelihoodInterval, numberOfThreads, maximumTokenCountPerDocument, numberOfSummaryTermsPerTopic,
                numberOfBurninIterations, resetRandomGenerator, Wrap(onFit)));
        }
    }
}