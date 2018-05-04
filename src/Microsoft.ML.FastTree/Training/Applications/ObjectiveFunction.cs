// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public abstract class ObjectiveFunctionBase
    {
        // buffer for gradient, weights and scores
        protected double[] Gradient;
        protected double[] Scores;

        // parameters
        protected double LearningRate;
        protected double Shrinkage;
        protected int GradSamplingRate;
        protected bool BestStepRankingRegressionTrees;

        protected double MaxTreeOutput;
        // random number generator
        private readonly Random _rnd;

        protected const int QueryThreadChunkSize = 100;

        public readonly Dataset Dataset;

        public double[] Weights { get; protected set; }

        public ObjectiveFunctionBase(
            Dataset dataset,
            double learningRate,
            double shrinkage,
            double maxTreeOutput,
            int gradSamplingRate,
            bool useBestStepRankingRegressionTree,
            int randomNumberGeneratorSeed)
        {
            Dataset = dataset;
            LearningRate = learningRate;
            Shrinkage = shrinkage;
            MaxTreeOutput = maxTreeOutput;
            GradSamplingRate = gradSamplingRate;
            BestStepRankingRegressionTrees = useBestStepRankingRegressionTree;
            _rnd = new Random(randomNumberGeneratorSeed);
            Gradient = new double[Dataset.NumDocs];
            Weights = new double[Dataset.NumDocs];
        }

        public virtual double[] GetGradient(IChannel ch, double[] scores)
        {
            Scores = scores;
            int sampleIndex = _rnd.Next(GradSamplingRate);
            using (Timer.Time(TimerEvent.ObjectiveFunctionGetDerivatives))
            {
                // REVIEW: This partitioning doesn't look optimal.
                // Probably make sence to investigate better ways of splitting data?
                var actions = new Action[(int)Math.Ceiling((double)Dataset.NumQueries / QueryThreadChunkSize)];
                var actionIndex = 0;
                var queue = new ConcurrentQueue<int>(Enumerable.Range(0, BlockingThreadPool.NumThreads));
                // fill the vectors with their correct values, query-by-query
                for (int q = 0; q < Dataset.NumQueries; q += QueryThreadChunkSize)
                {
                    int start = q;
                    actions[actionIndex++] = () =>
                      {
                          var threadIndex = 0;
                          Contracts.Check(queue.TryDequeue(out threadIndex));
                          GetGradientChunk(start, start + Math.Min(QueryThreadChunkSize, Dataset.NumQueries - start), GradSamplingRate, sampleIndex, threadIndex);
                          queue.Enqueue(threadIndex);
                      };
                }

                Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
            }
            return Gradient;
        }

        protected void GetGradientChunk(int startQuery, int endQuery, int sampleRate, int sampleIndex, int threadIndex)
        {
            for (int i = startQuery; i < endQuery; i++)
            {
                if (i % sampleRate == sampleIndex)
                {
                    GetGradientInOneQuery(i, threadIndex);
                }
            }
        }

        protected abstract void GetGradientInOneQuery(int query, int threadIndex);
    }
}
