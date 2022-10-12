// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.Trainers.FastTree;
using static Microsoft.ML.Trainers.FastTree.Dataset.RowForwardIndexer;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// SMAC is based on SMBO (sequential model based optimization). This implementation uses random forest as regressor to and 
    /// Expected Improvemnt as acquisition function. In practice, smac works well on search space which dimension is no larger than 20.
    /// </summary>
    internal class SmacTuner : ITuner
    {
        private readonly ITuner _randomTuner;
        private readonly MLContext _context;
        private readonly SearchSpace.SearchSpace _searchSpace;
        private readonly List<TrialResult> _histories;
        private Queue<Parameter> _candidates;
        private readonly Random _rnd = new Random();
        private readonly IChannel _channel;

        #region Smac Tuner arguments
        // Number of points to use for random initialization
        private readonly int _numberInitialPopulation;

        private readonly int _fitModelEveryNTrials;

        // Number of regression trees in forest
        private readonly int _numOfTrees;

        // Minimum number of data points required to be in a node if it is to be split further
        private readonly int _nMinForSplit;

        // Fraction of eligible dimensions to split on (i.e., split ratio)
        private readonly float _splitRatio;

        // Number of search parents to use for local search in maximizing EI acquisition function
        private readonly int _localSearchParentCount;

        // Number of random configurations when maximizing EI acquisition function
        private readonly int _numRandomEISearchConfigurations;

        private readonly double _epsilon;

        // Number of neighbors to sample for locally searching each numerical parameter
        private readonly int _numNeighborsForNumericalParams;
        #endregion
        public SmacTuner(MLContext context,
            SearchSpace.SearchSpace searchSpace,
            int numberInitialPopulation = 20,
            int fitModelEveryNTrials = 10,
            int numberOfTrees = 10,
            int nMinForSpit = 2,
            float splitRatio = 0.8f,
            int localSearchParentCount = 5,
            int numRandomEISearchConfigurations = 5000,
            double epsilon = 1e-5,
            int numNeighboursForNumericalParams = 4,
            int? seed = null,
            IChannel channel = null)
        {
            _context = context;
            _searchSpace = searchSpace;
            _randomTuner = new RandomSearchTuner(_searchSpace, seed);
            _histories = new List<TrialResult>();
            _candidates = new Queue<Parameter>();
            _fitModelEveryNTrials = fitModelEveryNTrials;
            _numberInitialPopulation = numberInitialPopulation;

            if (seed is int)
            {
                _rnd = new Random(seed.Value);
            }

            _numOfTrees = numberOfTrees;
            _nMinForSplit = nMinForSpit;
            _splitRatio = splitRatio;
            _localSearchParentCount = localSearchParentCount;
            _numRandomEISearchConfigurations = numRandomEISearchConfigurations;
            _epsilon = epsilon;
            _numNeighborsForNumericalParams = numNeighboursForNumericalParams;
            _channel = channel;
        }

        public Parameter Propose(TrialSettings settings)
        {
            var trialCount = _histories.Count + _candidates.Count;
            if (trialCount <= _numberInitialPopulation)
            {
                return _randomTuner.Propose(settings);
            }
            else
            {
                if (_candidates.Count <= 0)
                {
                    var model = FitModel(_histories);
                    _candidates = new Queue<Parameter>(GenerateCandidateConfigurations(_fitModelEveryNTrials, _histories, model));
                }

                return _candidates.Dequeue();
            }
        }

        private FastForestRegressionModelParameters FitModel(IEnumerable<TrialResult> history)
        {
            Single[] losses = new Single[history.Count()];
            Single[][] features = new Single[history.Count()][];

            int i = 0;
            foreach (var r in history)
            {
                features[i] = _searchSpace.MappingToFeatureSpace(r.TrialSettings.Parameter).Select(f => Convert.ToSingle(f)).ToArray();
                losses[i] = Convert.ToSingle(r.Loss);
                i++;
            }

            ArrayDataViewBuilder dvBuilder = new ArrayDataViewBuilder(_context);
            dvBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single, losses);
            dvBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single, features);

            _channel?.Trace("SMAC - start fitting random forest regressor");
            IDataView data = dvBuilder.GetDataView();
            // Set relevant random forest arguments.
            // Train random forest.
            var trainer = _context.Regression.Trainers.FastForest(new FastForestRegressionTrainer.Options()
            {
                FeatureFraction = _splitRatio,
                NumberOfTrees = _numOfTrees,
                MinimumExampleCountPerLeaf = _nMinForSplit,
            });
            var trainTestSplit = _context.Data.TrainTestSplit(data);
            var model = trainer.Fit(trainTestSplit.TrainSet);
            var predictor = model.Model;
            var test = model.Transform(trainTestSplit.TestSet);
            var eval = _context.Regression.Evaluate(test);
            _channel?.Trace("SMAC - end fitting random forest regressor");
            _channel?.Trace($"SMAC - fitting metric: rsquare {eval.RSquared}");

            // Return random forest predictor.
            return predictor;
        }

        /// <summary>
        /// Generates a set of candidate configurations to sweep through, based on a combination of random and local
        /// search, as outlined in Hutter et al. - Sequential Model-Based Optimization for General Algorithm Conﬁguration.
        /// Makes use of class private members which determine how many candidates are returned. This number will include
        /// random configurations interleaved (per the paper), and thus will be double the specified value.
        /// </summary>
        /// <param name="numOfCandidates">Number of candidate solutions to return.</param>
        /// <param name="previousRuns">History of previously evaluated points, with their empirical performance values.</param>
        /// <param name="forest">Trained random forest ensemble. Used in evaluating the candidates.</param>
        /// <returns>An array of ParamaterSets which are the candidate configurations to sweep.</returns>
        private Parameter[] GenerateCandidateConfigurations(int numOfCandidates, IEnumerable<TrialResult> previousRuns, FastForestRegressionModelParameters forest)
        {
            // Get k best previous runs ParameterSets.
            var bestKParamSets = _histories.OrderBy(i => i.Loss).Take(_localSearchParentCount).Select(r => r.TrialSettings.Parameter);

            // Perform local searches using the k best previous run configurations.
            var eiChallengers = GreedyPlusRandomSearch(bestKParamSets, forest, numOfCandidates >> 1);

            // Generate another set of random configurations to interleave.
            var randomChallengers = Enumerable.Range(0, numOfCandidates - eiChallengers.Length).Select(i => _randomTuner.Propose(new TrialSettings()));

            // Return interleaved challenger candidates with random candidates. Since the number of candidates from either can be less than
            // the number asked for, since we only generate unique candidates, and the number from either method may vary considerably.
            return eiChallengers.Concat(randomChallengers).ToArray();
        }

        /// <summary>
        /// Does a mix of greedy local search around best performing parameter sets, while throwing random parameter sets into the mix.
        /// </summary>
        /// <param name="parents">Beginning locations for local greedy search.</param>
        /// <param name="forest">Trained random forest, used later for evaluating parameters.</param>
        /// <param name="numOfCandidates">Number of candidate configurations returned by the method (top K).</param>
        /// <returns>Array of parameter sets, which will then be evaluated.</returns>
        private Parameter[] GreedyPlusRandomSearch(IEnumerable<Parameter> parents, FastForestRegressionModelParameters forest, int numOfCandidates)
        {
            double bestLoss = _histories.Min(t => t.Metric);

            var configurations = new HashSet<Tuple<double, Parameter>>();

            // Perform local search.
            foreach (var c in parents)
            {
                var bestChildKvp = LocalSearch(c, forest, bestLoss);
                configurations.Add(bestChildKvp);
            }

            // Additional set of random configurations to choose from during local search.
            var randomParameters = Enumerable.Range(0, _numRandomEISearchConfigurations).Select(i => _randomTuner.Propose(new TrialSettings()));
            var randomConfigurations = randomParameters.Select(parameter => new Tuple<double, Parameter>(EvaluateConfigurationsByEI(forest, bestLoss, parameter), parameter));

            var orderedConfigurations = configurations.Concat(randomConfigurations).OrderByDescending(p => p.Item1);
            var comparer = Parameter.FromInt(0);
            var retainedConfigs = new HashSet<Parameter>(orderedConfigurations.Select(x => x.Item2), comparer);

            // remove configurations matching previous run
            foreach (var previousRun in _histories)
            {
                retainedConfigs.Remove(previousRun.TrialSettings.Parameter);
            }

            return retainedConfigs.Take(numOfCandidates).ToArray();
        }

        /// <summary>
        /// Performs a local one-mutation neighborhood greedy search.
        /// </summary>
        /// <param name="startParameter">Starting parameter set configuration.</param>
        /// <param name="forest">Trained forest, for evaluation of points.</param>
        /// <param name="bestLoss">Best performance seen thus far.</param>
        /// <returns></returns>
        private Tuple<double, Parameter> LocalSearch(Parameter startParameter, FastForestRegressionModelParameters forest, double bestLoss)
        {
            try
            {
                double currentBestEI = EvaluateConfigurationsByEI(forest, bestLoss, startParameter);
                var currentBestConfig = startParameter;

                for (; ; )
                {
                    Parameter[] neighborhood = GetOneMutationNeighborhood(currentBestConfig);
                    var eis = neighborhood.Select(p => EvaluateConfigurationsByEI(forest, bestLoss, p)).ToArray();
                    var maxIndex = eis.ArgMax();
                    if (Math.Abs(eis[maxIndex] - currentBestEI) < _epsilon)
                        break;
                    else
                    {
                        currentBestConfig = neighborhood[maxIndex];
                        currentBestEI = eis[maxIndex];
                    }
                }

                return new Tuple<double, Parameter>(currentBestEI, currentBestConfig);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException("SMAC sweeper localSearch threw exception", e);
            }
        }

        private Parameter[] GetOneMutationNeighborhood(Parameter currentBestConfig)
        {
            var neighborhood = new List<Parameter>();
            var features = _searchSpace.MappingToFeatureSpace(currentBestConfig);
            for (int d = 0; d != _searchSpace.FeatureSpaceDim; ++d)
            {
                var newFeatures = features.Select(x => x).ToArray();
                if (_searchSpace.Step[d] is int step)
                {
                    // if step is not null, it means the parameter on that index is discrete.
                    // in that case, to sample a new value, we need to add the current feature value with 1/step
                    var nextStep = features[d] + (1.0 / step);
                    if (nextStep > 1)
                    {
                        nextStep -= 1;
                    }
                    newFeatures[d] = nextStep;
                    neighborhood.Add(_searchSpace.SampleFromFeatureSpace(newFeatures));
                }
                else
                {
                    // if step is null, it means the parameter on that index is numeric.
                    // create k neighbours for that value
                    for (int j = 0; j != _numNeighborsForNumericalParams; ++j)
                    {
                        var mu = features[d];
                        var sigma = 0.2;

                        while (true)
                        {
                            // sample normal(mu, sigma) from [0,1] uniform using box-muller transform
                            var u1 = _rnd.NextDouble();
                            var u2 = _rnd.NextDouble();
                            var newFeatured = mu + sigma * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                            if (newFeatured >= 0 && newFeatured <= 1)
                            {
                                newFeatures[d] = newFeatured;
                                break;
                            }
                        }
                        neighborhood.Add(_searchSpace.SampleFromFeatureSpace(newFeatures));
                    }
                }
            }

            return neighborhood.ToArray();
        }

        private double EvaluateConfigurationsByEI(FastForestRegressionModelParameters forest, double bestVal, Parameter parameter)
        {
            double[] leafPredictions = GetForestRegressionLeafValues(forest, parameter);
            double[] forestStatistics = ComputeForestStats(leafPredictions);
            return ComputeEI(bestVal, forestStatistics);
        }

        /// <summary>
        /// Goes through forest to extract the set of leaf values associated with filtering each configuration.
        /// </summary>
        /// <param name="forest">Trained forest predictor, used for filtering configs.</param>
        /// <param name="parameter">Parameter configuration.</param>
        /// <returns>1D array where rows correspond to configurations, and columns to the predicted leaf values.</returns>
        private double[] GetForestRegressionLeafValues(FastForestRegressionModelParameters forest, Parameter parameter)
        {
            List<double> leafValues = new List<double>();
            var transformedParams = _searchSpace.MappingToFeatureSpace(parameter).Select(p => Convert.ToSingle(p)).ToArray();
            for (var treeId = 0; treeId < forest.TrainedTreeEnsemble.Trees.Count; treeId++)
            {
                var features = new VBuffer<float>(transformedParams.Length, transformedParams);
                List<int> path = null;
                var leafId = forest.GetLeaf(treeId, features, ref path);
                var leafValue = forest.GetLeafValue(treeId, leafId);
                leafValues.Add(leafValue);
            }

            return leafValues.ToArray();
        }

        /// <summary>
        /// Computes the empirical means and standard deviations for trees in the forest for each configuration.
        /// </summary>
        /// <param name="leafValues">The sets of leaf values from which the means and standard deviations are computed.</param>
        /// <returns>A 2D array with one row per set of tree values, and the columns being mean and stddev, respectively.</returns>
        private double[] ComputeForestStats(double[] leafValues)
        {
            double[] meansAndStdDevs = new double[2];

            // Computes the empirical mean and empirical std dev from the leaf prediction values.\double[] row = new double[2];
            meansAndStdDevs[0] = VectorUtils.GetMean(leafValues);
            meansAndStdDevs[1] = VectorUtils.GetStandardDeviation(leafValues);

            return meansAndStdDevs;
        }

        private double ComputeEI(double bestLoss, double[] forestStatistics)
        {
            double empMean = forestStatistics[0];
            double empStdDev = forestStatistics[1];
            double centered = bestLoss - empMean;
            if (empStdDev == 0)
            {
                return centered;
            }
            double ztrans = centered / empStdDev;
            return centered * SweeperProbabilityUtils.StdNormalCdf(ztrans) + empStdDev * SweeperProbabilityUtils.StdNormalPdf(ztrans);
        }

        public void Update(TrialResult result)
        {
            _histories.Add(result);
        }
    }
}
