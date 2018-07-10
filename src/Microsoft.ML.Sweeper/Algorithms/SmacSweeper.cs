// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Sweeper;

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Sweeper.Algorithms;

[assembly: LoadableClass(typeof(SmacSweeper), typeof(SmacSweeper.Arguments), typeof(SignatureSweeper),
    "SMAC Sweeper", "SMACSweeper", "SMAC")]

namespace Microsoft.ML.Runtime.Sweeper
{
    //REVIEW: Figure out better way to do this. could introduce a base class for all smart sweepers,
    //encapsulating common functionality. This seems like a good plan to persue.
    public sealed class SmacSweeper : ISweeper
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Swept parameters", ShortName = "p")]
            public SubComponent<IValueGenerator, SignatureSweeperParameter>[] SweptParameters;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random number generator for the first batch sweeper", ShortName = "seed")]
            public int RandomSeed;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If iteration point is outside parameter definitions, should it be projected?", ShortName = "project")]
            public bool ProjectInBounds = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of regression trees in forest", ShortName = "numtrees")]
            public int NumOfTrees = 10;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum number of data points required to be in a node if it is to be split further", ShortName = "nmin")]
            public int NMinForSplit = 2;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of points to use for random initialization", ShortName = "nip")]
            public int NumberInitialPopulation = 20;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of search parents to use for local search in maximizing EI acquisition function", ShortName = "lsp")]
            public int LocalSearchParentCount = 10;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of random configurations when maximizing EI acquisition function", ShortName = "nrcan")]
            public int NumRandomEISearchConfigurations = 10000;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Fraction of eligible dimensions to split on (i.e., split ratio)", ShortName = "sr")]
            public Float SplitRatio = (Float)0.8;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Epsilon threshold for ending local searches", ShortName = "eps")]
            public Float Epsilon = (Float)0.00001;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of neighbors to sample for locally searching each numerical parameter", ShortName = "nnnp")]
            public int NumNeighborsForNumericalParams = 4;
        }

        private readonly ISweeper _randomSweeper;
        private readonly Arguments _args;
        private readonly IHost _host;

        private readonly IValueGenerator[] _sweepParameters;

        public SmacSweeper(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("Sweeper");

            _host.CheckUserArg(args.NumOfTrees > 0, nameof(args.NumOfTrees), "parameter must be greater than 0");
            _host.CheckUserArg(args.NMinForSplit > 1, nameof(args.NMinForSplit), "parameter must be greater than 1");
            _host.CheckUserArg(args.SplitRatio > 0 && args.SplitRatio <= 1, nameof(args.SplitRatio), "parameter must be in range (0,1].");
            _host.CheckUserArg(args.NumberInitialPopulation > 1, nameof(args.NumberInitialPopulation), "parameter must be greater than 1");
            _host.CheckUserArg(args.LocalSearchParentCount > 0, nameof(args.LocalSearchParentCount), "parameter must be greater than 0");
            _host.CheckUserArg(args.NumRandomEISearchConfigurations > 0, nameof(args.NumRandomEISearchConfigurations), "parameter must be greater than 0");
            _host.CheckUserArg(args.NumNeighborsForNumericalParams > 0, nameof(args.NumNeighborsForNumericalParams), "parameter must be greater than 0");

            _args = args;
            _host.CheckUserArg(Utils.Size(args.SweptParameters) > 0, nameof(args.SweptParameters), "SMAC sweeper needs at least one parameter to sweep over");
            _sweepParameters = args.SweptParameters.Select(p => p.CreateInstance(_host)).ToArray();
            _randomSweeper = new UniformRandomSweeper(env, new SweeperBase.ArgumentsBase(), _sweepParameters);
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
            FastForestRegressionPredictor forestPredictor = FitModel(viableRuns);

            // Using acquisition function and current best, get candidate configuration(s).
            return GenerateCandidateConfigurations(numOfCandidates, viableRuns, forestPredictor);
        }

        private FastForestRegressionPredictor FitModel(IEnumerable<IRunResult> previousRuns)
        {
            Single[] targets = new Single[previousRuns.Count()];
            Single[][] features = new Single[previousRuns.Count()][];

            int i = 0;
            foreach (RunResult r in previousRuns)
            {
                features[i] = SweeperProbabilityUtils.ParameterSetAsFloatArray(_host, _sweepParameters, r.ParameterSet, true);
                targets[i] = (Float)r.MetricValue;
                i++;
            }

            ArrayDataViewBuilder dvBuilder = new ArrayDataViewBuilder(_host);
            dvBuilder.AddColumn("Label", NumberType.Float, targets);
            dvBuilder.AddColumn("Features", NumberType.Float, features);

            IDataView view = dvBuilder.GetDataView();
            _host.Assert(view.GetRowCount() == targets.Length, "This data view will have as many rows as there have been evaluations");
            RoleMappedData data = TrainUtils.CreateExamples(view, "Label", "Features");

            using (IChannel ch = _host.Start("Single training"))
            {
                // Set relevant random forest arguments.
                FastForestRegression.Arguments args = new FastForestRegression.Arguments();
                args.FeatureFraction = _args.SplitRatio;
                args.NumTrees = _args.NumOfTrees;
                args.MinDocumentsInLeafs = _args.NMinForSplit;

                // Train random forest.
                FastForestRegression trainer = new FastForestRegression(_host, args);
                trainer.Train(data);
                FastForestRegressionPredictor predictor = trainer.CreatePredictor();

                // Return random forest predictor.
                ch.Done();
                return predictor;
            }
        }

        /// <summary>
        /// Generates a set of candidate configurations to sweep through, based on a combination of random and local
        /// search, as outlined in Hutter et al - Sequential Model-Based Optimization for General Algorithm Conﬁguration.
        /// Makes use of class private members which determine how many candidates are returned. This number will include
        /// random configurations interleaved (per the paper), and thus will be double the specified value.
        /// </summary>
        /// <param name="numOfCandidates">Number of candidate solutions to return.</param>
        /// <param name="previousRuns">History of previously evaluated points, with their emprical performance values.</param>
        /// <param name="forest">Trained random forest ensemble. Used in evaluating the candidates.</param>
        /// <returns>An array of ParamaterSets which are the candidate configurations to sweep.</returns>
        private ParameterSet[] GenerateCandidateConfigurations(int numOfCandidates, IEnumerable<IRunResult> previousRuns, FastForestRegressionPredictor forest)
        {
            ParameterSet[] configs = new ParameterSet[numOfCandidates];

            // Get k best previous runs ParameterSets.
            ParameterSet[] bestKParamSets = GetKBestConfigurations(previousRuns, forest, _args.LocalSearchParentCount);

            // Perform local searches using the k best previous run configurations.
            ParameterSet[] eiChallengers = GreedyPlusRandomSearch(bestKParamSets, forest, (int)Math.Ceiling(numOfCandidates / 2.0F), previousRuns);

            // Generate another set of random configurations to interleave
            ParameterSet[] randomChallengers = _randomSweeper.ProposeSweeps(numOfCandidates - eiChallengers.Length, previousRuns);

            // Return interleaved challenger candidates with random candidates
            for (int j = 0; j < configs.Length; j++)
                configs[j] = j % 2 == 0 ? eiChallengers[j / 2] : randomChallengers[j / 2];

            return configs;
        }

        private ParameterSet[] TreeOrderedCandidatesSearch(FastForestRegressionPredictor forest, int numOfCandidates, IEnumerable<IRunResult> previousRuns)
        {
            // Step 1: Get ordered list of all leaf values.
            SortedList<double, Tuple<int, int>> leafValueList = new SortedList<double, Tuple<int, int>>(Comparer<double>.Create((x, y) => y.CompareTo(x)));
            for (int i = 0; i < forest.TrainedEnsemble.NumTrees; i++)
            {
                RegressionTree t = forest.TrainedEnsemble.GetTreeAt(i);
                for (int j = 0; j < t.NumLeaves; j++)
                {
                    double val = t.LeafValue(j);
                    while (leafValueList.ContainsKey(val))
                        val += Double.Epsilon;
                    leafValueList.Add(val, Tuple.Create(i, j));
                }
            }
            return null;
        }

        /// <summary>
        /// Does a mix of greedy local search around best performing parameter sets, while throwing random parameter sets into the mix.
        /// </summary>
        /// <param name="parents">Beginning locations for local greedy search.</param>
        /// <param name="forest">Trained random forest, used later for evaluating parameters.</param>
        /// <param name="numOfCandidates">Number of candidate configurations returned by the method (top K).</param>
        /// <param name="previousRuns">Historical run results.</param>
        /// <returns>Array of parameter sets, which will then be evaluated.</returns>
        private ParameterSet[] GreedyPlusRandomSearch(ParameterSet[] parents, FastForestRegressionPredictor forest, int numOfCandidates, IEnumerable<IRunResult> previousRuns)
        {
            // REVIEW: The IsMetricMaximizing flag affects the comparator, so that 
            // performing Max() should get the best, regardless of if it is maximizing or
            // minimizing.
            RunResult bestRun = (RunResult)previousRuns.Max();
            RunResult worstRun = (RunResult)previousRuns.Min();
            double bestVal = bestRun.IsMetricMaximizing ? bestRun.MetricValue : worstRun.MetricValue - bestRun.MetricValue;

            HashSet<Tuple<double, ParameterSet>> configurations = new HashSet<Tuple<double, ParameterSet>>();

            // Perform local search.
            foreach (ParameterSet c in parents)
            {
                Tuple<double, ParameterSet> bestChildKvp = LocalSearch(c, forest, bestVal, _args.Epsilon);
                configurations.Add(bestChildKvp);
            }

            // Additional set of random configurations to choose from during local search.
            ParameterSet[] randomConfigs = _randomSweeper.ProposeSweeps(_args.NumRandomEISearchConfigurations, previousRuns);
            double[] randomEIs = EvaluateConfigurationsByEI(forest, bestVal, randomConfigs);
            _host.Assert(randomConfigs.Length == randomEIs.Length);

            for (int i = 0; i < randomConfigs.Length; i++)
                configurations.Add(new Tuple<double, ParameterSet>(randomEIs[i], randomConfigs[i]));

            HashSet<ParameterSet> retainedConfigs = new HashSet<ParameterSet>();
            IOrderedEnumerable<Tuple<double, ParameterSet>> bestConfigurations = configurations.OrderByDescending(x => x.Item1);

            foreach (Tuple<double, ParameterSet> t in bestConfigurations.Take(numOfCandidates))
                retainedConfigs.Add(t.Item2);

            return retainedConfigs.ToArray();
        }

        /// <summary>
        /// Performs a local one-mutation neighborhood greedy search.
        /// </summary>
        /// <param name="parent">Starting parameter set configuration.</param>
        /// <param name="forest">Trained forest, for evaluation of points.</param>
        /// <param name="bestVal">Best performance seen thus far.</param>
        /// <param name="epsilon">Threshold for when to stop the local search.</param>
        /// <returns></returns>
        private Tuple<double, ParameterSet> LocalSearch(ParameterSet parent, FastForestRegressionPredictor forest, double bestVal, double epsilon)
        {
            try
            {
                double currentBestEI = EvaluateConfigurationsByEI(forest, bestVal, new ParameterSet[] { parent })[0];
                ParameterSet currentBestConfig = parent;

                for (; ; )
                {
                    ParameterSet[] neighborhood = GetOneMutationNeighborhood(currentBestConfig);
                    double[] eis = EvaluateConfigurationsByEI(forest, bestVal, neighborhood);
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
                throw _host.Except(e, "SMAC sweeper localSearch threw exception");
            }
        }

        /// <summary>
        /// Computes a single-mutation neighborhood (one param at a time) for a given configuration. For
        /// numeric parameters, samples K mutations (i.e., creates K neighbors based on that paramater).
        /// </summary>
        /// <param name="parent">Starting configuration.</param>
        /// <returns>A set of configurations that each differ from parent in exactly one parameter.</returns>
        private ParameterSet[] GetOneMutationNeighborhood(ParameterSet parent)
        {
            List<ParameterSet> neighbors = new List<ParameterSet>();
            SweeperProbabilityUtils spu = new SweeperProbabilityUtils(_host);

            for (int i = 0; i < _sweepParameters.Length; i++)
            {
                // This allows us to query possible values of this parameter.
                IValueGenerator sweepParam = _sweepParameters[i];

                // This holds the actual value for this parameter, chosen in this parameter set.
                IParameterValue pset = parent[sweepParam.Name];

                _host.AssertValue(pset);

                DiscreteValueGenerator parameterDiscrete = sweepParam as DiscreteValueGenerator;
                if (parameterDiscrete != null)
                {
                    // Create one neighbor for every discrete parameter.
                    Float[] neighbor = SweeperProbabilityUtils.ParameterSetAsFloatArray(_host, _sweepParameters, parent, false);

                    int hotIndex = -1;
                    for (int j = 0; j < parameterDiscrete.Count; j++)
                    {
                        if (parameterDiscrete[j].Equals(pset))
                        {
                            hotIndex = j;
                            break;
                        }
                    }

                    _host.Assert(hotIndex >= 0);

                    Random r = new Random();
                    int randomIndex = r.Next(0, parameterDiscrete.Count - 1);
                    randomIndex += randomIndex >= hotIndex ? 1 : 0;
                    neighbor[i] = randomIndex;
                    neighbors.Add(SweeperProbabilityUtils.FloatArrayAsParameterSet(_host, _sweepParameters, neighbor, false));
                }
                else
                {
                    INumericValueGenerator parameterNumeric = sweepParam as INumericValueGenerator;
                    _host.Check(parameterNumeric != null, "SMAC sweeper can only sweep over discrete and numeric parameters");

                    // Create k neighbors (typically 4) for every numerical parameter.
                    for (int j = 0; j < _args.NumNeighborsForNumericalParams; j++)
                    {
                        Float[] neigh = SweeperProbabilityUtils.ParameterSetAsFloatArray(_host, _sweepParameters, parent, false);
                        double newVal = spu.NormalRVs(1, neigh[i], 0.2)[0];
                        while (newVal <= 0.0 || newVal >= 1.0)
                            newVal = spu.NormalRVs(1, neigh[i], 0.2)[0];
                        neigh[i] = (Float)newVal;
                        ParameterSet neighbor = SweeperProbabilityUtils.FloatArrayAsParameterSet(_host, _sweepParameters, neigh, false);
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
        private double[][] GetForestRegressionLeafValues(FastForestRegressionPredictor forest, ParameterSet[] configs)
        {
            List<double[]> datasetLeafValues = new List<double[]>();
            var e = forest.TrainedEnsemble;
            foreach (ParameterSet config in configs)
            {
                List<double> leafValues = new List<double>();
                foreach (RegressionTree t in e.Trees)
                {
                    Float[] transformedParams = SweeperProbabilityUtils.ParameterSetAsFloatArray(_host, _sweepParameters, config, true);
                    VBuffer<Float> features = new VBuffer<Float>(transformedParams.Length, transformedParams);
                    leafValues.Add((Float)t.LeafValues[t.GetLeaf(ref features)]);
                }
                datasetLeafValues.Add(leafValues.ToArray());
            }
            return datasetLeafValues.ToArray();
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

        private double[] EvaluateConfigurationsByEI(FastForestRegressionPredictor forest, double bestVal, ParameterSet[] configs)
        {
            double[][] leafPredictions = GetForestRegressionLeafValues(forest, configs);
            double[][] forestStatistics = ComputeForestStats(leafPredictions);
            return ComputeEIs(bestVal, forestStatistics);
        }

        private ParameterSet[] GetKBestConfigurations(IEnumerable<IRunResult> previousRuns, FastForestRegressionPredictor forest, int k = 10)
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

        private double ComputeEI(double bestVal, double[] forestStatistics)
        {
            double empMean = forestStatistics[0];
            double empStdDev = forestStatistics[1];
            double centered = empMean - bestVal;
            double ztrans = centered / empStdDev;
            return centered * SweeperProbabilityUtils.StdNormalCdf(ztrans) + empStdDev * SweeperProbabilityUtils.StdNormalPdf(ztrans);
        }

        private double[] ComputeEIs(double bestVal, double[][] forestStatistics)
        {
            double[] eis = new double[forestStatistics.Length];
            for (int i = 0; i < forestStatistics.Length; i++)
                eis[i] = ComputeEI(bestVal, forestStatistics[i]);
            return eis;
        }

        // *********** Utility Functions *******************

        private ParameterSet UpdateParameterSet(ParameterSet original, IParameterValue newParam)
        {
            List<IParameterValue> parameters = new List<IParameterValue>();
            for (int i = 0; i < _sweepParameters.Length; i++)
            {
                if (_sweepParameters[i].Name.Equals(newParam.Name))
                    parameters.Add(newParam);
                else
                {
                    parameters.Add(original[_sweepParameters[i].Name]);
                }
            }

            return new ParameterSet(parameters);
        }

        private Float ParameterAsFloat(ParameterSet parameterSet, int index)
        {
            _host.Assert(parameterSet.Count == _sweepParameters.Length);
            _host.Assert(index >= 0 && index <= _sweepParameters.Length);

            var sweepParam = _sweepParameters[index];
            var pset = parameterSet[sweepParam.Name];
            _host.AssertValue(pset);

            var parameterDiscrete = sweepParam as DiscreteValueGenerator;
            if (parameterDiscrete != null)
            {
                int hotIndex = -1;
                for (int j = 0; j < parameterDiscrete.Count; j++)
                {
                    if (parameterDiscrete[j].Equals(pset))
                    {
                        hotIndex = j;
                        break;
                    }
                }
                _host.Assert(hotIndex >= 0);

                return hotIndex;
            }
            else
            {
                var parameterNumeric = sweepParam as INumericValueGenerator;
                _host.Check(parameterNumeric != null, "SMAC sweeper can only sweep over discrete and numeric parameters");

                // Normalizing all numeric parameters to [0,1] range.
                return parameterNumeric.NormalizeValue(pset);
            }
        }
    }
}
