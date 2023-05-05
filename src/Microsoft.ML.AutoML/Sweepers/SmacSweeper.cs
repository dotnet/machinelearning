// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Float = System.Single;

namespace Microsoft.ML.AutoML
{
    //REVIEW: Figure out better way to do this. could introduce a base class for all smart sweepers,
    //encapsulating common functionality. This seems like a good plan to pursue.
    internal sealed class SmacSweeper : ISweeper
    {
        public sealed class Arguments
        {
            // Swept parameters
            public IValueGenerator[] SweptParameters;

            // Seed for the random number generator for the first batch sweeper
            public int RandomSeed;

            // If iteration point is outside parameter definitions, should it be projected?
            public bool ProjectInBounds;

            // Number of regression trees in forest
            public int NumOfTrees;

            // Minimum number of data points required to be in a node if it is to be split further
            public int NMinForSplit;

            // Number of points to use for random initialization
            public int NumberInitialPopulation;

            // Number of search parents to use for local search in maximizing EI acquisition function
            public int LocalSearchParentCount;

            // Number of random configurations when maximizing EI acquisition function
            public int NumRandomEISearchConfigurations;

            // Fraction of eligible dimensions to split on (i.e., split ratio)
            public Float SplitRatio;

            // Epsilon threshold for ending local searches
            public Float Epsilon;

            // Number of neighbors to sample for locally searching each numerical parameter
            public int NumNeighborsForNumericalParams;

            public Arguments()
            {
                ProjectInBounds = true;
                NumOfTrees = 10;
                NMinForSplit = 2;
                NumberInitialPopulation = 20;
                LocalSearchParentCount = 10;
                NumRandomEISearchConfigurations = 10000;
                SplitRatio = 0.8f;
                Epsilon = 0.00001f;
                NumNeighborsForNumericalParams = 4;
            }
        }

        private readonly ISweeper _randomSweeper;
        private readonly Arguments _args;
        private readonly MLContext _context;

        private readonly IValueGenerator[] _sweepParameters;

        public SmacSweeper(MLContext context, Arguments args)
        {
            _context = context;
            _args = args;
            _sweepParameters = args.SweptParameters;
            _randomSweeper = new UniformRandomSweeper(new SweeperBase.ArgumentsBase(), _sweepParameters);
        }

        public ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            int numOfCandidates = maxSweeps;

            // Initialization: Will enter here on first iteration and use the default (random)
            // sweeper to generate initial candidates.
            int numRuns = previousRuns == null ? 0 : previousRuns.Count();
            if (numRuns < _args.NumberInitialPopulation)
                return _randomSweeper.ProposeSweeps(Math.Min(numOfCandidates, _args.NumberInitialPopulation - numRuns), previousRuns);

            // Only retain viable runs
            List<IRunResult> viableRuns = new List<IRunResult>();
            foreach (RunResult run in previousRuns)
            {
                if (run != null && run.HasMetricValue)
                    viableRuns.Add(run);
            }

            // Fit Random Forest Model on previous run data.
            var forestPredictor = FitModel(viableRuns);

            // Using acquisition function and current best, get candidate configuration(s).
            return GenerateCandidateConfigurations(numOfCandidates, viableRuns, forestPredictor);
        }

        private FastForestRegressionModelParameters FitModel(IEnumerable<IRunResult> previousRuns)
        {
            Single[] targets = new Single[previousRuns.Count()];
            Single[][] features = new Single[previousRuns.Count()][];

            int i = 0;
            foreach (RunResult r in previousRuns)
            {
                features[i] = SweeperProbabilityUtils.ParameterSetAsFloatArray(_sweepParameters, r.ParameterSet, true);
                targets[i] = (Float)r.MetricValue;
                i++;
            }

            ArrayDataViewBuilder dvBuilder = new ArrayDataViewBuilder(_context);
            dvBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single, targets);
            dvBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single, features);

            IDataView data = dvBuilder.GetDataView();
            Runtime.Contracts.Assert(data.GetRowCount() == targets.Length, "This data view will have as many rows as there have been evaluations");

            // Set relevant random forest arguments.
            // Train random forest.
            var trainer = _context.Regression.Trainers.FastForest(new FastForestRegressionTrainer.Options()
            {
                FeatureFraction = _args.SplitRatio,
                NumberOfTrees = _args.NumOfTrees,
                MinimumExampleCountPerLeaf = _args.NMinForSplit
            });
            var predictor = trainer.Fit(data).Model;

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
        private ParameterSet[] GenerateCandidateConfigurations(int numOfCandidates, IEnumerable<IRunResult> previousRuns, FastForestRegressionModelParameters forest)
        {
            // Get k best previous runs ParameterSets.
            ParameterSet[] bestKParamSets = GetKBestConfigurations(previousRuns, _args.LocalSearchParentCount);

            // Perform local searches using the k best previous run configurations.
            ParameterSet[] eiChallengers = GreedyPlusRandomSearch(bestKParamSets, forest, (int)Math.Ceiling(numOfCandidates / 2.0F), previousRuns);

            // Generate another set of random configurations to interleave.
            ParameterSet[] randomChallengers = _randomSweeper.ProposeSweeps(numOfCandidates - eiChallengers.Length, previousRuns);

            // Return interleaved challenger candidates with random candidates. Since the number of candidates from either can be less than
            // the number asked for, since we only generate unique candidates, and the number from either method may vary considerably.
            ParameterSet[] configs = new ParameterSet[eiChallengers.Length + randomChallengers.Length];
            Array.Copy(eiChallengers, 0, configs, 0, eiChallengers.Length);
            Array.Copy(randomChallengers, 0, configs, eiChallengers.Length, randomChallengers.Length);

            return configs;
        }

        /// <summary>
        /// Does a mix of greedy local search around best performing parameter sets, while throwing random parameter sets into the mix.
        /// </summary>
        /// <param name="parents">Beginning locations for local greedy search.</param>
        /// <param name="forest">Trained random forest, used later for evaluating parameters.</param>
        /// <param name="numOfCandidates">Number of candidate configurations returned by the method (top K).</param>
        /// <param name="previousRuns">Historical run results.</param>
        /// <returns>Array of parameter sets, which will then be evaluated.</returns>
        private ParameterSet[] GreedyPlusRandomSearch(ParameterSet[] parents, FastForestRegressionModelParameters forest, int numOfCandidates, IEnumerable<IRunResult> previousRuns)
        {
            RunResult bestRun = (RunResult)previousRuns.Max();
            RunResult worstRun = (RunResult)previousRuns.Min();
            double bestVal = bestRun.MetricValue;

            HashSet<Tuple<double, ParameterSet>> configurations = new HashSet<Tuple<double, ParameterSet>>();

            // Perform local search.
            foreach (ParameterSet c in parents)
            {
                Tuple<double, ParameterSet> bestChildKvp = LocalSearch(c, forest, bestVal, _args.Epsilon, bestRun.IsMetricMaximizing);
                configurations.Add(bestChildKvp);
            }

            // Additional set of random configurations to choose from during local search.
            ParameterSet[] randomConfigs = _randomSweeper.ProposeSweeps(_args.NumRandomEISearchConfigurations, previousRuns);
            double[] randomEIs = EvaluateConfigurationsByEI(forest, bestVal, randomConfigs, bestRun.IsMetricMaximizing);
            Runtime.Contracts.Assert(randomConfigs.Length == randomEIs.Length);

            for (int i = 0; i < randomConfigs.Length; i++)
                configurations.Add(new Tuple<double, ParameterSet>(randomEIs[i], randomConfigs[i]));

            IOrderedEnumerable<Tuple<double, ParameterSet>> bestConfigurations = configurations.OrderByDescending(x => x.Item1);

            var retainedConfigs = new HashSet<ParameterSet>(bestConfigurations.Select(x => x.Item2));

            // remove configurations matching previous run
            foreach (var previousRun in previousRuns)
            {
                retainedConfigs.Remove(previousRun.ParameterSet);
            }

            return retainedConfigs.Take(numOfCandidates).ToArray();
        }

        /// <summary>
        /// Performs a local one-mutation neighborhood greedy search.
        /// </summary>
        /// <param name="parent">Starting parameter set configuration.</param>
        /// <param name="forest">Trained forest, for evaluation of points.</param>
        /// <param name="bestVal">Best performance seen thus far.</param>
        /// <param name="epsilon">Threshold for when to stop the local search.</param>
        /// <param name="isMetricMaximizing">Whether SMAC should aim to maximize (vs minimize) metric.</param>
        /// <returns></returns>
        private Tuple<double, ParameterSet> LocalSearch(ParameterSet parent, FastForestRegressionModelParameters forest, double bestVal, double epsilon, bool isMetricMaximizing)
        {
            try
            {
                double currentBestEI = EvaluateConfigurationsByEI(forest, bestVal, new ParameterSet[] { parent }, isMetricMaximizing)[0];
                ParameterSet currentBestConfig = parent;

                for (; ; )
                {
                    ParameterSet[] neighborhood = GetOneMutationNeighborhood(currentBestConfig);
                    double[] eis = EvaluateConfigurationsByEI(forest, bestVal, neighborhood, isMetricMaximizing);
                    int bestIndex = eis.ArgMax();
                    if (eis[bestIndex] - currentBestEI < _args.Epsilon)
                        break;
                    else
                    {
                        currentBestConfig = neighborhood[bestIndex];
                        currentBestEI = eis[bestIndex];
                    }
                }

                return new Tuple<double, ParameterSet>(currentBestEI, currentBestConfig);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException("SMAC sweeper localSearch threw exception", e);
            }
        }

        /// <summary>
        /// Computes a single-mutation neighborhood (one parameter at a time) for a given configuration. For
        /// numeric parameters, samples K mutations (i.e., creates K neighbors based on that parameter).
        /// </summary>
        /// <param name="parent">Starting configuration.</param>
        /// <returns>A set of configurations that each differ from parent in exactly one parameter.</returns>
        private ParameterSet[] GetOneMutationNeighborhood(ParameterSet parent)
        {
            List<ParameterSet> neighbors = new List<ParameterSet>();
            SweeperProbabilityUtils spu = new SweeperProbabilityUtils();

            for (int i = 0; i < _sweepParameters.Length; i++)
            {
                // This allows us to query possible values of this parameter.
                IValueGenerator sweepParam = _sweepParameters[i];

                // This holds the actual value for this parameter, chosen in this parameter set.
                IParameterValue pset = parent[sweepParam.Name];

                Runtime.Contracts.Assert(pset != null);

                DiscreteValueGenerator parameterDiscrete = sweepParam as DiscreteValueGenerator;
                if (parameterDiscrete != null)
                {
                    // Create one neighbor for every discrete parameter.
                    Float[] neighbor = SweeperProbabilityUtils.ParameterSetAsFloatArray(_sweepParameters, parent, false);

                    int hotIndex = -1;
                    for (int j = 0; j < parameterDiscrete.Count; j++)
                    {
                        if (parameterDiscrete[j].Equals(pset))
                        {
                            hotIndex = j;
                            break;
                        }
                    }

                    Runtime.Contracts.Assert(hotIndex >= 0);

                    Random r = new Random();
                    int randomIndex = r.Next(0, parameterDiscrete.Count - 1);
                    randomIndex += randomIndex >= hotIndex ? 1 : 0;
                    neighbor[i] = randomIndex;
                    neighbors.Add(SweeperProbabilityUtils.FloatArrayAsParameterSet(_sweepParameters, neighbor, false));
                }
                else
                {
                    INumericValueGenerator parameterNumeric = sweepParam as INumericValueGenerator;
                    Runtime.Contracts.Assert(parameterNumeric != null, "SMAC sweeper can only sweep over discrete and numeric parameters");

                    // Create k neighbors (typically 4) for every numerical parameter.
                    for (int j = 0; j < _args.NumNeighborsForNumericalParams; j++)
                    {
                        Float[] neigh = SweeperProbabilityUtils.ParameterSetAsFloatArray(_sweepParameters, parent, false);
                        double newVal = spu.NormalRVs(1, neigh[i], 0.2)[0];
                        while (newVal <= 0.0 || newVal >= 1.0)
                            newVal = spu.NormalRVs(1, neigh[i], 0.2)[0];
                        neigh[i] = (Float)newVal;
                        ParameterSet neighbor = SweeperProbabilityUtils.FloatArrayAsParameterSet(_sweepParameters, neigh, false);
                        neighbors.Add(neighbor);
                    }
                }
            }
            return neighbors.ToArray();
        }

        /// <summary>
        /// Goes through forest to extract the set of leaf values associated with filtering each configuration.
        /// </summary>
        /// <param name="forest">Trained forest predictor, used for filtering configs.</param>
        /// <param name="configs">Parameter configurations.</param>
        /// <returns>2D array where rows correspond to configurations, and columns to the predicted leaf values.</returns>
        private double[][] GetForestRegressionLeafValues(FastForestRegressionModelParameters forest, ParameterSet[] configs)
        {
            List<double[]> datasetLeafValues = new List<double[]>();
            foreach (ParameterSet config in configs)
            {
                List<double> leafValues = new List<double>();
                for (var treeId = 0; treeId < forest.TrainedTreeEnsemble.Trees.Count; treeId++)
                {
                    Float[] transformedParams = SweeperProbabilityUtils.ParameterSetAsFloatArray(_sweepParameters, config, true);
                    VBuffer<Float> features = new VBuffer<Float>(transformedParams.Length, transformedParams);
                    var leafId = GetLeaf(forest, treeId, features);
                    var leafValue = GetLeafValue(forest, treeId, leafId);
                    leafValues.Add(leafValue);
                }
                datasetLeafValues.Add(leafValues.ToArray());
            }
            return datasetLeafValues.ToArray();
        }

        // Todo: Remove the reflection below for TreeTreeEnsembleModelParameters methods GetLeaf and GetLeafValue.
        // Long-term, replace with tree featurizer once it becomes available
        // Tracking issue -- https://github.com/dotnet/machinelearning-automl/issues/342
        private static readonly MethodInfo _getLeafMethod = typeof(TreeEnsembleModelParameters).GetMethod("GetLeaf", BindingFlags.NonPublic | BindingFlags.Instance);
        private static readonly MethodInfo _getLeafValueMethod = typeof(TreeEnsembleModelParameters).GetMethod("GetLeafValue", BindingFlags.NonPublic | BindingFlags.Instance);

        private static int GetLeaf(TreeEnsembleModelParameters model, int treeId, VBuffer<Float> features)
        {
            List<int> path = null;
            return (int)_getLeafMethod.Invoke(model, new object[] { treeId, features, path });
        }

        private static float GetLeafValue(TreeEnsembleModelParameters model, int treeId, int leafId)
        {
            return (float)_getLeafValueMethod.Invoke(model, new object[] { treeId, leafId });
        }

        /// <summary>
        /// Computes the empirical means and standard deviations for trees in the forest for each configuration.
        /// </summary>
        /// <param name="leafValues">The sets of leaf values from which the means and standard deviations are computed.</param>
        /// <returns>A 2D array with one row per set of tree values, and the columns being mean and stddev, respectively.</returns>
        private double[][] ComputeForestStats(double[][] leafValues)
        {
            // Computes the empirical mean and empirical std dev from the leaf prediction values.
            double[][] meansAndStdDevs = new double[leafValues.Length][];
            for (int i = 0; i < leafValues.Length; i++)
            {
                double[] row = new double[2];
                row[0] = VectorUtils.GetMean(leafValues[i]);
                row[1] = VectorUtils.GetStandardDeviation(leafValues[i]);
                meansAndStdDevs[i] = row;
            }
            return meansAndStdDevs;
        }

        private double[] EvaluateConfigurationsByEI(FastForestRegressionModelParameters forest, double bestVal, ParameterSet[] configs, bool isMetricMaximizing)
        {
            double[][] leafPredictions = GetForestRegressionLeafValues(forest, configs);
            double[][] forestStatistics = ComputeForestStats(leafPredictions);
            return ComputeEIs(bestVal, forestStatistics, isMetricMaximizing);
        }

        private ParameterSet[] GetKBestConfigurations(IEnumerable<IRunResult> previousRuns, int k = 10)
        {
            // NOTE: Should we change this to rank according to EI (using forest), instead of observed performance?

            SortedSet<RunResult> bestK = new SortedSet<RunResult>();

            foreach (RunResult r in previousRuns)
            {
                RunResult worst = bestK.Min();

                if (bestK.Count < k || r.CompareTo(worst) > 0)
                    bestK.Add(r);

                if (bestK.Count > k)
                    bestK.Remove(worst);
            }

            // Extract the ParamaterSets and return.
            List<ParameterSet> outSet = new List<ParameterSet>();
            foreach (RunResult r in bestK)
                outSet.Add(r.ParameterSet);
            return outSet.ToArray();
        }

        private double ComputeEI(double bestVal, double[] forestStatistics, bool isMetricMaximizing)
        {
            double empMean = forestStatistics[0];
            double empStdDev = forestStatistics[1];
            double centered = empMean - bestVal;
            if (!isMetricMaximizing)
            {
                centered *= -1;
            }
            if (empStdDev == 0)
            {
                return centered;
            }
            double ztrans = centered / empStdDev;
            return centered * SweeperProbabilityUtils.StdNormalCdf(ztrans) + empStdDev * SweeperProbabilityUtils.StdNormalPdf(ztrans);
        }

        private double[] ComputeEIs(double bestVal, double[][] forestStatistics, bool isMetricMaximizing)
        {
            double[] eis = new double[forestStatistics.Length];
            for (int i = 0; i < forestStatistics.Length; i++)
                eis[i] = ComputeEI(bestVal, forestStatistics[i], isMetricMaximizing);
            return eis;
        }
    }
}
