// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public class ScoreTracker
    {
        public string DatasetName;
        public Dataset Dataset;
        public double[] Scores;
        protected double[] InitScores;
        public delegate void ScoresUpdatedDelegate();
        public ScoresUpdatedDelegate ScoresUpdated;
        public ScoreTracker(ScoreTracker s)
        {
            DatasetName = s.DatasetName;
            Dataset = s.Dataset;
            InitScores = s.InitScores;
            Scores = (double[])s.Scores.Clone();
        }

        public ScoreTracker(string datasetName, Dataset set, double[] initScores)
        {
            Initialize(datasetName, set, initScores);
        }
        public void Initialize(string datasetName, Dataset set, double[] initScores)
        {
            DatasetName = datasetName;
            Dataset = set;
            InitScores = initScores;
            InitializeScores(initScores);
        }

        //Creates linear combination of scores1 + tree * multiplier
        public void Initialize(ScoreTracker scores1, RegressionTree tree, DocumentPartitioning partitioning, double multiplier)
        {
            InitScores = null;
            if (Scores == null || Scores.Length != scores1.Scores.Length)
            {
                Scores = (double[])scores1.Scores.Clone();
            }
            else
            {
                Array.Copy(scores1.Scores, Scores, Scores.Length);
            }
            AddScores(tree, partitioning, multiplier);
            SendScoresUpdatedMessage();
        }

        //InitScores -initScores can be null in such case the scores are reinitialized to Zero
        private void InitializeScores(double[] initScores)
        {
            if (initScores == null)
            {
                if (Scores == null)
                    Scores = new double[Dataset.NumDocs];
                else
                    Array.Clear(Scores, 0, Scores.Length);
            }
            else
            {
                if (initScores.Length != Dataset.NumDocs)
                    throw Contracts.Except("The length of initScores do not match the length of training set");
                Scores = (double[])initScores.Clone();
            }
            SendScoresUpdatedMessage();
        }

        public virtual void SetScores(double[] scores)
        {
            Scores = scores;
            SendScoresUpdatedMessage();
        }

        public void SendScoresUpdatedMessage()
        {
            if (ScoresUpdated != null)
                ScoresUpdated();
        }

        public void RandomizeScores(int rngSeed, bool reverseRandomization)
        {
            Random rndStart = new Random(rngSeed);
            for (int i = 0; i < Scores.Length; ++i)
                Scores[i] += 10.0 * rndStart.NextDouble() * (reverseRandomization ? -1.0 : 1.0);
            SendScoresUpdatedMessage();
        }

        public virtual void AddScores(RegressionTree tree, double multiplier)
        {
            tree.AddOutputsToScores(Dataset, Scores, multiplier);
            SendScoresUpdatedMessage();
        }

        //Use faster method for score update with Partitioning
        // suitable for TrainSet
        public virtual void AddScores(RegressionTree tree, DocumentPartitioning partitioning, double multiplier)
        {
            Parallel.For(0, tree.NumLeaves, new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, (leaf) =>
            {
                int[] documents;
                int begin;
                int count;
                partitioning.ReferenceLeafDocuments(leaf, out documents, out begin, out count);
                double output = tree.LeafValue(leaf) * multiplier;
                for (int i = begin; i < begin + count; ++i)
                    Scores[documents[i]] += output;
            });

            SendScoresUpdatedMessage();
        }
    }

    //Accelerated gradient descent score tracker
    public class AgdScoreTracker : ScoreTracker
    {
        private int _k;
        public double[] YK;
        public double[] XK { get { return Scores; } set { Scores = value; } } //An Xk is an alias to scores
        public AgdScoreTracker(string datsetName, Dataset set, double[] initScores)
            : base(datsetName, set, initScores)
        {
            _k = 0;
            YK = (double[])XK.Clone();
        }

        public override void SetScores(double[] scores)
        {
            throw Contracts.ExceptNotSupp("This code should not be reachable");
        }

        //Computes AGD specific mutiplier. Given that we have tree number t in ensamble (we count trees starting from 0)
        //And we have total k trees in ensemble, what should be the multiplier on the tree when sum the ensemble together based on AGD formula being
        //X[k+1] = Y[k] + Tree[k]
        //Y[k+1] = X[k+1] + C[k] * (X[k+1] – X[k])
        //C[k] = (k-1) / (k+2)

        private static Dictionary<int, Dictionary<int, double>> _treeMultiplierMap = new Dictionary<int, Dictionary<int, double>>();
        public static double TreeMultiplier(int t, int k)
        {
            if (_treeMultiplierMap.ContainsKey(t))
            {
                if (_treeMultiplierMap[t].ContainsKey(k))
                    return _treeMultiplierMap[t][k];
            }
            else
            {
                _treeMultiplierMap[t] = new Dictionary<int, double>();
            }
            double result = double.NaN;
            if (k == t)
                result = 0.0;
            else if (k == t + 1)
                result = 1.0;
            else
                result = TreeMultiplier(t, k - 1) + (k - 1.0 - 1.0) / (k - 1.0 + 2.0) * (TreeMultiplier(t, k - 1) - TreeMultiplier(t, k - 2)); //This is last tree beeing added X[k] = Y[k-1] + 1.0 * T[k]
            _treeMultiplierMap[t][k] = result;
            return result;
        }

        public override void AddScores(RegressionTree tree, double multiplier)
        {
            _k++;
            double coeff = (_k - 1.0) / (_k + 2.0);

            int innerLoopSize = 1 + Dataset.NumDocs / BlockingThreadPool.NumThreads;   // +1 is to make sure we don't have a few left over at the end
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * Dataset.NumDocs / innerLoopSize)];
            var actionIndex = 0;
            for (int d = 0; d < Dataset.NumDocs; d += innerLoopSize)
            {
                var fromDoc = d;
                var toDoc = Math.Min(d + innerLoopSize, Dataset.NumDocs);
                actions[actionIndex++] = () =>
                {
                    var featureBins = Dataset.GetFeatureBinRowwiseIndexer();
                    for (int doc = fromDoc; doc < toDoc; doc++)
                    {
                        double output = multiplier * tree.GetOutput(featureBins[doc]);
                        double newXK = YK[doc] + output;
                        double newYK = newXK + coeff * (newXK - XK[doc]);
                        XK[doc] = newXK;
                        YK[doc] = newYK;
                    }
                };
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
            SendScoresUpdatedMessage();
        }

        public override void AddScores(RegressionTree tree, DocumentPartitioning partitioning, double multiplier)
        {
            _k++;
            double coeff = (_k - 1.0) / (_k + 2.0);
            var actions = new Action[tree.NumLeaves];
            Parallel.For(0, tree.NumLeaves, new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads },
                (int leaf) =>
                {
                    int[] documents;
                    int begin;
                    int count;
                    partitioning.ReferenceLeafDocuments(leaf, out documents, out begin, out count);
                    double output = tree.LeafValue(leaf) * multiplier;
                    for (int i = begin; i < begin + count; ++i)
                    {
                        int doc = documents[i];
                        double newXK = YK[doc] + output;
                        double newYK = newXK + coeff * (newXK - XK[doc]);
                        XK[doc] = newXK;
                        YK[doc] = newYK;
                    }
                });
            SendScoresUpdatedMessage();
        }
    }
}
