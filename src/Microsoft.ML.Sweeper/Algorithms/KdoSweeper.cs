// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Sweeper.Algorithms;

[assembly: LoadableClass(typeof(KdoSweeper), typeof(KdoSweeper.Arguments), typeof(SignatureSweeper),
    "KDO Sweeper", "KDOSweeper", "KDO")]

namespace Microsoft.ML.Runtime.Sweeper.Algorithms
{
    /// <summary>
    /// Kernel Density Optimization (KDO) is a sequential model-based optimization method originally developed by George D. Montanez (me).
    /// The search space consists of a unit hypercube, with one dimension per hyperparameter (it is a spatial method, so scaling the dimensions
    /// to the unit hypercube is critical). The idea is that the exploration of the cube to find good values is performed by creating an approximate
    /// (and biased) kernel density estimate of the space (where density corresponds to metric performance), concentrating mass in regions of better
    /// performance, then drawing samples from the pdf.
    ///
    /// To trade off exploration versus exploitation, an fitness proportional mutation scheme is used. Uniform random points are selected during
    /// initialization and during the runs (parameter controls how often). A Gaussian model is fit to the distribution of performance values, and
    /// each evaluated point in the history is given a value between 0 and 1 corresponding to the CDF evaluation of its performance under the
    /// Gaussian. Points with low quantile values are mutated more strongly than those with higher values, which allows the method to hone in
    /// precisely when approaching really good regions.
    ///
    /// Categorical parameters are handled by forming a categorical distribution on possible values weighted by observed performance of each value,
    /// taken independently.
    /// </summary>

    public sealed class KdoSweeper : ISweeper
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Swept parameters", ShortName = "p", SignatureType = typeof(SignatureSweeperParameter))]
            public IComponentFactory<IValueGenerator>[] SweptParameters;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random number generator for the first batch sweeper", ShortName = "seed")]
            public int RandomSeed;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If iteration point is outside parameter definitions, should it be projected?", ShortName = "project")]
            public bool ProjectInBounds = true;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of points to use for random initialization", ShortName = "nip")]
            public int NumberInitialPopulation = 20;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum mutation spread", ShortName = "mms")]
            public double MinimumMutationSpread = 0.001;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum length of history to retain", ShortName = "hlen")]
            public int HistoryLength = 20;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, draws samples from independent Beta distributions, rather than multivariate Gaussian", ShortName = "beta")]
            public bool Beta = false;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, uses simpler mutation and concentration model")]
            public bool Simple = false;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Proportion of trials, between 0 and 1, that are uniform random draws", ShortName = "prand")]
            public double ProportionRandom = 0.05;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum power for rescaling (the larger the number, the stronger the exploitation of good points)", ShortName = "wrp")]
            public double WeightRescalingPower = 30;

            // REVIEW: this parameter should be removed as soon as we test the new method (as Prabhat Roy is currently doing 9/18/2017). It is here
            // to allow him to continue to run existing tests in progress using the previous behavior, but should be removed once we're sure this new change
            // doesn't degrade performance.
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "(Deprecated) Use legacy discrete parameter behavior.", ShortName = "legacy", Hide = true, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public bool LegacyDpBehavior = false;
        }

        private readonly ISweeper _randomSweeper;
        private readonly ISweeper _redundantSweeper;
        private readonly Arguments _args;
        private readonly IHost _host;

        private readonly IValueGenerator[] _sweepParameters;
        private readonly SweeperProbabilityUtils _spu;
        private readonly SortedSet<Float[]> _alreadySeenConfigs;
        private readonly List<ParameterSet> _randomParamSets;

        public KdoSweeper(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("Sweeper");

            _host.CheckUserArg(args.NumberInitialPopulation > 1, nameof(args.NumberInitialPopulation), "Must be greater than 1");
            _host.CheckUserArg(args.HistoryLength > 1, nameof(args.HistoryLength), "Must be greater than 1");
            _host.CheckUserArg(args.MinimumMutationSpread >= 0, nameof(args.MinimumMutationSpread), "Must be nonnegative");
            _host.CheckUserArg(0 <= args.ProportionRandom && args.ProportionRandom <= 1, nameof(args.ProportionRandom), "Must be in [0, 1]");
            _host.CheckUserArg(args.WeightRescalingPower >= 1, nameof(args.WeightRescalingPower), "Must be greater or equal to 1");

            _args = args;
            _host.CheckUserArg(Utils.Size(args.SweptParameters) > 0, nameof(args.SweptParameters), "KDO sweeper needs at least one parameter to sweep over");
            _sweepParameters = args.SweptParameters.Select(p => p.CreateComponent(_host)).ToArray();
            _randomSweeper = new UniformRandomSweeper(env, new SweeperBase.ArgumentsBase(), _sweepParameters);
            _redundantSweeper = new UniformRandomSweeper(env, new SweeperBase.ArgumentsBase { Retries = 0 }, _sweepParameters);
            _spu = new SweeperProbabilityUtils(_host);
            _alreadySeenConfigs = new SortedSet<Float[]>(new FloatArrayComparer());
            _randomParamSets = new List<ParameterSet>();
        }

        public ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            int numOfCandidates = maxSweeps;
            var prevRuns = previousRuns?.ToArray() ?? new IRunResult[0];
            var numSweeps = Math.Min(numOfCandidates, _args.NumberInitialPopulation - prevRuns.Length);

            // Initialization: Will enter here on first iteration and use the default (random)
            // sweeper to generate initial candidates.
            if (prevRuns.Length < _args.NumberInitialPopulation)
            {
                ParameterSet[] rcs;
                int attempts = 0;
                do
                {
                    rcs = _randomSweeper.ProposeSweeps(numSweeps, prevRuns);
                    attempts++;
                } while (rcs.Length < 1 && attempts < 100);

                // If failed to grab a new parameter set, generate random (and possibly redundant) one.
                if (rcs.Length == 0)
                    rcs = _redundantSweeper.ProposeSweeps(numSweeps, prevRuns);

                foreach (ParameterSet ps in rcs)
                    _randomParamSets.Add(ps);

                return rcs;
            }

            // Only retain viable runs
            var viableRuns = prevRuns.Cast<RunResult>().Where(run => run != null && run.HasMetricValue).Cast<IRunResult>().ToArray();

            // Make sure we have a metric
            if (viableRuns.Length == 0 && prevRuns.Length > 0)
            {
                // I'm not sure if this is too much detail, but it might be.
                string errorMessage = $"Error: Sweep run results are missing metric values. \n\n" +
                                      $"NOTE: Default metric of 'AUC' only viable for binary classification problems. \n" +
                                      $"Please include an evaluator (ev) component with an appropriate metric specified for your task type.\n\n" +
                                       "Example RSP using alternate metric (i.e., AccuracyMicro):\nrunner=Local{\n\tev=Tlc{m=AccuracyMicro}\n\tpattern={...etc...}\n}";
                throw _host.Except(new Exception(errorMessage), errorMessage);
            }

            return GenerateCandidateConfigurations(numOfCandidates, viableRuns);
        }

        /// <summary>
        /// REVIEW: Assumes metric is between 0.0 and 1.0. Will not work with metrics that have values outside this range.
        /// </summary>
        private ParameterSet[] GenerateCandidateConfigurations(int numOfCandidates, IRunResult[] previousRuns)
        {
            _host.Assert(Utils.Size(previousRuns) > 1);
            IRunResult[] history = previousRuns;
            int totalHistoryLength = history.Length;

            // Reduce length of history if necessary.
            if (history.Length > _args.HistoryLength)
                history = TruncateHistory(history);

            double[] randomVals = ExtractRandomRunValues(previousRuns);
            double rMean = VectorUtils.GetMean(randomVals);
            // Add a small amount of variance for unlikely edge cases when all values were identical (i.e., zero variance).
            // Should not happen, but adding a small variance ensures it will never cause problems if it does.
            double rVar = Math.Pow(VectorUtils.GetStandardDeviation(randomVals), 2) + 1e-10;
            double[] weights = HistoryToWeights(history, totalHistoryLength, rMean, rVar);
            int[] parentIndicies = SampleCategoricalDist(numOfCandidates, weights);
            return GenerateChildConfigurations(history, parentIndicies, weights, previousRuns, rMean, rVar);
        }

        private ParameterSet[] GenerateChildConfigurations(IRunResult[] history, int[] parentIndicies, double[] weights, IRunResult[] previousRuns, double rMean, double rVar)
        {
            _host.Assert(history.Length == weights.Length && parentIndicies.Max() < history.Length);
            List<ParameterSet> children = new List<ParameterSet>();

            for (int i = 0; i < parentIndicies.Length; i++)
            {
                RunResult parent = (RunResult)history[parentIndicies[i]];
                children.Add(SampleChild(parent.ParameterSet, parent.MetricValue, history.Length, previousRuns, rMean, rVar, parent.IsMetricMaximizing));
            }

            return children.ToArray();
        }

        /// <summary>
        /// Sample child configuration from configuration centered at parent, using fitness proportional mutation.
        /// </summary>
        /// <param name="parent">Starting parent configuration (used as mean in multivariate Gaussian).</param>
        /// <param name="fitness">Numeric value indicating how good a configuration parent is.</param>
        /// <param name="n">Count of how many items currently in history.</param>
        /// <param name="previousRuns">Run history.</param>
        /// <param name="rMean">Mean metric value of previous random runs.</param>
        /// <param name="rVar">Metric value empirical variance of previous random runs.</param>
        /// <param name="isMetricMaximizing">Flag for if we are minimizing or maximizing values.</param>
        /// <returns>A mutated version of parent (i.e., point sampled near parent).</returns>
        private ParameterSet SampleChild(ParameterSet parent, double fitness, int n, IRunResult[] previousRuns, double rMean, double rVar, bool isMetricMaximizing)
        {
            Float[] child = SweeperProbabilityUtils.ParameterSetAsFloatArray(_host, _sweepParameters, parent, false);
            List<int> numericParamIndices = new List<int>();
            List<double> numericParamValues = new List<double>();
            int loopCount = 0;

            // Interleave uniform random samples, according to proportion defined.
            if (_spu.SampleUniform() <= _args.ProportionRandom)
            {
                ParameterSet ps = _randomSweeper.ProposeSweeps(1)[0];
                _randomParamSets.Add(ps);
                return ps;
            }

            do
            {
                for (int i = 0; i < _sweepParameters.Length; i++)
                {
                    // This allows us to query possible values of this parameter.
                    var sweepParam = _sweepParameters[i];

                    if (sweepParam is DiscreteValueGenerator parameterDiscrete)
                    {
                        // Sample categorical parameter.
                        double[] categoryWeights = _args.LegacyDpBehavior
                            ? CategoriesToWeightsOld(parameterDiscrete, previousRuns)
                            : CategoriesToWeights(parameterDiscrete, previousRuns);
                        child[i] = SampleCategoricalDist(1, categoryWeights)[0];
                    }
                    else
                    {
                        var parameterNumeric = sweepParam as INumericValueGenerator;
                        _host.Check(parameterNumeric != null, "KDO sweeper can only sweep over discrete and numeric parameters");
                        numericParamIndices.Add(i);
                        numericParamValues.Add(child[i]);
                    }
                }

                if (numericParamIndices.Count > 0)
                {
                    if (!_args.Beta)
                    {
                        // Sample point from multivariate Gaussian, centered on parent values, with mutation proportional to fitness.
                        double[] mu = numericParamValues.ToArray();
                        double correctedVal = isMetricMaximizing
                            ? 1.0 - SweeperProbabilityUtils.NormalCdf(fitness, rMean, rVar)
                            : 1.0 - SweeperProbabilityUtils.NormalCdf(2 * rMean - fitness, rMean, rVar);
                        double bandwidthScale = Math.Max(_args.MinimumMutationSpread, correctedVal);
                        double[] stddevs = Enumerable.Repeat(_args.Simple ? 0.2 : bandwidthScale, mu.Length).ToArray();
                        double[][] bandwidthMatrix = BuildBandwidthMatrix(n, stddevs);
                        double[] sampledPoint = SampleDiagonalCovMultivariateGaussian(1, mu, bandwidthMatrix)[0];
                        for (int j = 0; j < sampledPoint.Length; j++)
                            child[numericParamIndices[j]] = (Float)Corral(sampledPoint[j]);
                    }
                    else
                    {
                        // If Beta flag set, sample from independent Beta distributions instead.
                        SysRandom rng = new SysRandom();
                        double alpha = 1 + 15 * fitness;
                        foreach (int index in numericParamIndices)
                        {
                            const double epsCutoff = 1e-10;
                            double eps = Math.Min(Math.Max(child[index], epsCutoff), 1 - epsCutoff);
                            double beta = alpha / eps - alpha;
                            child[index] = (Float)Stats.SampleFromBeta(rng, alpha, beta);
                        }
                    }
                }

                // Don't get stuck at local point.
                loopCount++;
                if (loopCount > 10)
                    return _randomSweeper.ProposeSweeps(1, null)[0];
            } while (_alreadySeenConfigs.Contains(child));

            _alreadySeenConfigs.Add(child);
            return SweeperProbabilityUtils.FloatArrayAsParameterSet(_host, _sweepParameters, child, false);
        }

        private double Corral(double v)
        {
            if (v > 1)
                return 1;
            return v < 0 ? 0 : v;
        }

        /// <summary>
        /// Creates a diagonal rule-of-thumb kernel bandwidth matrix.
        /// </summary>
        /// <param name="n">Number of items in history (just acts as a regularization parameter in KDO).</param>
        /// <param name="stddevs">Array of per feature standard deviations.</param>
        /// <returns>A matrix of bandwidth values, for use in kernel density estimation.</returns>
        private double[][] BuildBandwidthMatrix(int n, double[] stddevs)
        {
            int d = stddevs.Length;
            double[][] bandwidthMatrix = new double[d][];
            double p1 = 1.0 / (d + 4);
            double p2 = Math.Pow((4.0 / (d + 2)), p1);

            for (int i = 0; i < d; i++)
            {
                // Silverman's rule-of-thumb.
                bandwidthMatrix[i] = new double[d];
                bandwidthMatrix[i][i] = p2 * stddevs[i] * Math.Pow(n, -p1);
            }

            return bandwidthMatrix;
        }

        /// <summary>
        /// Converts a set of history into a set of weights, one for each run in the history.
        /// </summary>
        /// <param name="history">Input set of historical runs.</param>
        /// <param name="n">Number of total runs (history may be truncated)</param>
        /// <param name="rMean">Mean metric value of previous random runs.</param>
        /// <param name="rVar">Metric value empirical variance of previous random runs.</param>
        /// <returns>Array of weights.</returns>
        private double[] HistoryToWeights(IRunResult[] history, int n, double rMean, double rVar)
        {
            // Extract weights and normalize.
            double[] weights = new double[history.Length];

            for (int i = 0; i < history.Length; i++)
                weights[i] = (double)history[i].MetricValue;

            // Fitness proportional scaling constant.
            bool isMinimizing = history.Length > 0 && !history[0].IsMetricMaximizing;
            double currentMaxPerf = isMinimizing ? SweeperProbabilityUtils.NormalCdf(2 * rMean - weights.Min(), rMean, rVar) : SweeperProbabilityUtils.NormalCdf(weights.Max(), rMean, rVar);

            // Normalize weights to sum to one. Automatically Takes care of case where all are equal to zero.
            weights = isMinimizing ? SweeperProbabilityUtils.InverseNormalize(weights) : SweeperProbabilityUtils.Normalize(weights);

            // Scale weights. (Concentrates mass on good points, depending on how good the best currently is.)
            for (int i = 0; i < weights.Length; i++)
                weights[i] = _args.Simple ? Math.Pow(weights[i], Math.Min(Math.Sqrt(n), 100)) : Math.Pow(weights[i], _args.WeightRescalingPower * currentMaxPerf);

            weights = SweeperProbabilityUtils.Normalize(weights);

            return weights;
        }

        private double[] ExtractRandomRunValues(IEnumerable<IRunResult> previousRuns)
        {
            return (from RunResult r in previousRuns where _randomParamSets.Contains(r.ParameterSet) select r.MetricValue).ToArray();
        }

        /// <summary>
        /// New version of CategoryToWeights method, which fixes an issue where we could
        /// potentially assign a lot of mass to bad categories.
        /// </summary>
        private double[] CategoriesToWeights(DiscreteValueGenerator param, IRunResult[] previousRuns)
        {
            double[] weights = new double[param.Count];
            Dictionary<string, int> labelToIndex = new Dictionary<string, int>();
            int[] counts = new int[param.Count];

            // Map categorical values to their index.
            for (int j = 0; j < param.Count; j++)
                labelToIndex[param[j].ValueText] = j;

            // Add mass according to performance
            bool isMaximizing = true;
            foreach (RunResult r in previousRuns)
            {
                weights[labelToIndex[r.ParameterSet[param.Name].ValueText]] += r.MetricValue;
                counts[labelToIndex[r.ParameterSet[param.Name].ValueText]]++;
                isMaximizing = r.IsMetricMaximizing;
            }

            // Take average mass for each category
            for (int i = 0; i < weights.Length; i++)
                weights[i] /= (counts[i] > 0 ? counts[i] : 1);

            // If any learner has not been seen, default it's average to
            // best value to encourage exploration of untried algorithms.
            double bestVal = isMaximizing ?
                previousRuns.Cast<RunResult>().Where(r => r.HasMetricValue).Max(r => r.MetricValue) :
                previousRuns.Cast<RunResult>().Where(r => r.HasMetricValue).Min(r => r.MetricValue);
            for (int i = 0; i < weights.Length; i++)
                weights[i] += counts[i] == 0 ? bestVal : 0;

            // Normalize weights to sum to one and return
            return isMaximizing ? SweeperProbabilityUtils.Normalize(weights) : SweeperProbabilityUtils.InverseNormalize(weights);
        }

        /// <summary>
        /// REVIEW: This was the original CategoriesToWeights function. Should be deprecated once we can validate the new function works
        /// better. It contains a subtle issue, such that categories with poor performance but which are seen a lot will have
        /// high weight. New function addresses this issue, while also improving exploration capability of algorithm.
        /// </summary>
        /// <param name="param"></param>
        /// <param name="previousRuns"></param>
        /// <returns></returns>
        private double[] CategoriesToWeightsOld(DiscreteValueGenerator param, IEnumerable<IRunResult> previousRuns)
        {
            double[] weights = new double[param.Count];
            Dictionary<string, int> labelToIndex = new Dictionary<string, int>();

            // Map categorical values to their index.
            for (int j = 0; j < param.Count; j++)
                labelToIndex[param[j].ValueText] = j;

            // Add pseudo-observations, to account for unobserved parameter settings.
            for (int i = 0; i < weights.Length; i++)
                weights[i] = 0.1;

            // Sum up the results for each category value.
            bool isMaximizing = true;
            foreach (RunResult r in previousRuns)
            {
                weights[labelToIndex[r.ParameterSet[param.Name].ValueText]] += r.MetricValue;
                isMaximizing = r.IsMetricMaximizing;
            }

            // Normalize weights to sum to one and return
            return isMaximizing ? SweeperProbabilityUtils.Normalize(weights) : SweeperProbabilityUtils.InverseNormalize(weights);
        }

        /// <summary>
        /// Keep only the top K results from the history.
        /// </summary>
        /// <param name="history">set of all history.</param>
        /// <returns>The best K points contained in the history.</returns>
        private IRunResult[] TruncateHistory(IRunResult[] history)
        {
            SortedSet<RunResult> bestK = new SortedSet<RunResult>();

            foreach (RunResult r in history)
            {
                RunResult worst = bestK.Min();

                if (bestK.Count < _args.HistoryLength || r.CompareTo(worst) > 0)
                    bestK.Add(r);

                if (bestK.Count > _args.HistoryLength)
                    bestK.Remove(worst);
            }

            return bestK.ToArray();
        }

        private int[] SampleCategoricalDist(int numSamples, double[] weights)
        {
            _host.AssertNonEmpty(weights);
            _host.Assert(weights.Sum() > 0);
            return _spu.SampleCategoricalDistribution(numSamples, weights);
        }

        private double[][] SampleDiagonalCovMultivariateGaussian(int numRVs, double[] mu, double[][] diagonalCovariance)
        {
            // Perform checks to ensure covariance has correct form (square diagonal with dimension d).
            int d = mu.Length;
            _host.Assert(d > 0 && diagonalCovariance.Length == d);
            for (int i = 0; i < d; i++)
            {
                _host.Assert(diagonalCovariance[i].Length == d);
                for (int j = 0; j < d; j++)
                {
                    _host.Assert((i == j && diagonalCovariance[i][j] >= 0) || diagonalCovariance[i][j] == 0);
                }
            }

            // Create transform matrix
            double[][] a = new double[d][];
            for (int i = 0; i < d; i++)
            {
                a[i] = new double[d];
                for (int j = 0; j < d; j++)
                    a[i][j] = i + j == d - 1 ? Math.Sqrt(diagonalCovariance[i][i]) : 0;
            }

            // Sample points
            double[][] points = new double[numRVs][];
            for (int i = 0; i < points.Length; i++)
            {
                // Generate vector of independent standard normal RVs.
                points[i] = VectorTransformAdd(mu, _spu.NormalRVs(mu.Length, 0, 1), a);
            }

            return points;
        }

        private double[] VectorTransformAdd(double[] m, double[] z, double[][] a)
        {
            int d = m.Length;
            double[] result = new double[d];
            for (int i = 0; i < d; i++)
            {
                result[i] = m[i];
                for (int j = 0; j < d; j++)
                    result[i] += a[i][j] * z[j];
            }
            return result;
        }

        private sealed class FloatArrayComparer : IComparer<Float[]>
        {
            public int Compare(Float[] x, Float[] y)
            {
                if (x.Length != y.Length)
                    return x.Length > y.Length ? 1 : -1;

                for (int i = 0; i < x.Length; i++)
                {
                    if (x[i] != y[i])
                        return 1;
                }

                return 0;
            }
        }
    }
}
