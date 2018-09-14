// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Internal.Internallearn;

// REVIEW: Do we really need all these names?
[assembly: LoadableClass(FastTreeRankingTrainer.Summary, typeof(FastTreeRankingTrainer), typeof(FastTreeRankingTrainer.Arguments),
    new[] { typeof(SignatureRankerTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastTreeRankingTrainer.UserNameValue,
    FastTreeRankingTrainer.LoadNameValue,
    FastTreeRankingTrainer.ShortName,

    // FastRank names
    "FastRankRanking",
    "FastRankRankingWrapper",
    "rank",
    "frrank",
    "btrank")]

[assembly: LoadableClass(typeof(FastTreeRankingPredictor), null, typeof(SignatureLoadModel),
    "FastTree Ranking Executor",
    FastTreeRankingPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(FastTree), null, typeof(SignatureEntryPointModule), "FastTree")]

namespace Microsoft.ML.Runtime.FastTree
{
    /// <include file='doc.xml' path='doc/members/member[@name="FastTree"]/*' />
    public sealed partial class FastTreeRankingTrainer
        : BoostingFastTreeTrainerBase<FastTreeRankingTrainer.Arguments, RankingPredictionTransformer<FastTreeRankingPredictor>, FastTreeRankingPredictor>,
          IHasLabelGains
    {
        internal const string LoadNameValue = "FastTreeRanking";
        internal const string UserNameValue = "FastTree (Boosted Trees) Ranking";
        internal const string Summary = "Trains gradient boosted decision trees to the LambdaRank quasi-gradient.";
        internal const string ShortName = "ftrank";

        private IEnsembleCompressor<short> _ensembleCompressor;
        private Test _specialTrainSetTest;
        private TestHistory _firstTestSetHistory;

        /// <summary>
        /// The prediction kind for this trainer.
        /// </summary>
        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        private readonly SchemaShape.Column[] _outputColumns;

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeRankingTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="groupIdColumn">The name for the column containing the group ID. </param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public FastTreeRankingTrainer(IHostEnvironment env, string labelColumn, string featureColumn, string groupIdColumn,
            string weightColumn = null, Action<Arguments> advancedSettings = null)
            : base(env, MakeLabelColumn(labelColumn), featureColumn, weightColumn, groupIdColumn, advancedSettings: advancedSettings)
        {
            _outputColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false)
            };
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeRankingTrainer"/> by using the legacy <see cref="Arguments"/> class.
        /// </summary>
        public FastTreeRankingTrainer(IHostEnvironment env, Arguments args)
                : base(env, args, MakeLabelColumn(args.LabelColumn))
        {
            _outputColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false)
            };
        }

        protected override float GetMaxLabel()
        {
            return GetLabelGains().Length - 1;
        }

        protected override FastTreeRankingPredictor TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;
            ValidData = context.ValidationSet;

            using (var ch = Host.Start("Training"))
            {
                var maxLabel = GetLabelGains().Length - 1;
                ConvertData(trainData);
                TrainCore(ch);
                FeatureCount = trainData.Schema.Feature.Type.ValueCount;
                ch.Done();
            }
            return new FastTreeRankingPredictor(Host, TrainedEnsemble, FeatureCount, InnerArgs);
        }

        public Double[] GetLabelGains()
        {
            try
            {
                Host.AssertValue(Args.CustomGains);
                return Args.CustomGains.Split(',').Select(k => Convert.ToDouble(k.Trim())).ToArray();
            }
            catch (Exception ex)
            {
                if (ex is FormatException || ex is OverflowException)
                    throw Host.Except(ex, "Error in the format of custom gains. Inner exception is {0}", ex.Message);
                throw;
            }
        }

        protected override void CheckArgs(IChannel ch)
        {
            if (!string.IsNullOrEmpty(Args.CustomGains))
            {
                var stringGain = Args.CustomGains.Split(',');
                if (stringGain.Length < 5)
                {
                    throw ch.ExceptUserArg(nameof(Args.CustomGains),
                        "{0} an invalid number of gain levels. We require at least 5. Make certain they're comma separated.",
                        stringGain.Length);
                }
                Double[] gain = new Double[stringGain.Length];
                for (int i = 0; i < stringGain.Length; ++i)
                {
                    if (!Double.TryParse(stringGain[i], out gain[i]))
                    {
                        throw ch.ExceptUserArg(nameof(Args.CustomGains),
                            "Could not parse '{0}' as a floating point number", stringGain[0]);
                    }
                }
                DcgCalculator.LabelGainMap = gain;
                Dataset.DatasetSkeleton.LabelGainMap = gain;
            }

            ch.CheckUserArg((Args.EarlyStoppingRule == null && !Args.EnablePruning) || (Args.EarlyStoppingMetrics == 1 || Args.EarlyStoppingMetrics == 3), nameof(Args.EarlyStoppingMetrics),
                "earlyStoppingMetrics should be 1 or 3.");

            base.CheckArgs(ch);
        }

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
        {
            return new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);
        }

        protected override void Initialize(IChannel ch)
        {
            base.Initialize(ch);
            if (Args.CompressEnsemble)
            {
                _ensembleCompressor = new LassoBasedEnsembleCompressor();
                _ensembleCompressor.Initialize(Args.NumTrees, TrainSet, TrainSet.Ratings, Args.RngSeed);
            }
        }

        protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch)
        {
            return new LambdaRankObjectiveFunction(TrainSet, TrainSet.Ratings, Args, ParallelTraining);
        }

        protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            if (Args.UseLineSearch)
            {
                _specialTrainSetTest = new FastNdcgTest(optimizationAlgorithm.TrainingScores, TrainSet.Ratings, Args.SortingAlgorithm, Args.EarlyStoppingMetrics);
                optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(_specialTrainSetTest, 0, Args.NumPostBracketSteps, Args.MinStepSize);
            }
            return optimizationAlgorithm;
        }

        protected override BaggingProvider CreateBaggingProvider()
        {
            Host.Assert(Args.BaggingSize > 0);
            return new RankingBaggingProvider(TrainSet, Args.NumLeaves, Args.RngSeed, Args.BaggingTrainFraction);
        }

        protected override void PrepareLabels(IChannel ch)
        {
        }

        protected override Test ConstructTestForTrainingData()
        {
            return new NdcgTest(ConstructScoreTracker(TrainSet), TrainSet.Ratings, Args.SortingAlgorithm);
        }

        protected override void InitializeTests()
        {
            if (Args.TestFrequency != int.MaxValue)
            {
                AddFullTests();
            }

            if (Args.PrintTestGraph)
            {
                // If FirstTestHistory is null (which means the tests were not intialized due to /tf==infinity)
                // We need initialize first set for graph printing
                // Adding to a tests would result in printing the results after final iteration
                if (_firstTestSetHistory == null)
                {
                    var firstTestSetTest = CreateFirstTestSetTest();
                    _firstTestSetHistory = new TestHistory(firstTestSetTest, 0);
                }
            }

            // Tests for early stopping.
            TrainTest = CreateSpecialTrainSetTest();
            if (ValidSet != null)
                ValidTest = CreateSpecialValidSetTest();

            if (Args.PrintTrainValidGraph && Args.EnablePruning && _specialTrainSetTest == null)
            {
                _specialTrainSetTest = CreateSpecialTrainSetTest();
            }

            if (Args.EnablePruning && ValidTest != null)
            {
                if (!Args.UseTolerantPruning)
                {
                    //use simple eraly stopping condition
                    PruningTest = new TestHistory(ValidTest, 0);
                }
                else
                {
                    //use tolerant stopping condition
                    PruningTest = new TestWindowWithTolerance(ValidTest, 0, Args.PruningWindowSize, Args.PruningThreshold);
                }
            }
        }

        private void AddFullTests()
        {
            Tests.Add(CreateStandardTest(TrainSet));

            if (ValidSet != null)
            {
                Test test = CreateStandardTest(ValidSet);
                Tests.Add(test);
            }

            for (int t = 0; TestSets != null && t < TestSets.Length; ++t)
            {
                Test test = CreateStandardTest(TestSets[t]);
                if (t == 0)
                {
                    _firstTestSetHistory = new TestHistory(test, 0);
                }

                Tests.Add(test);
            }
        }

        protected override void PrintIterationMessage(IChannel ch, IProgressChannel pch)
        {
            // REVIEW: Shift to using progress channels to report this information.
#if OLD_TRACE
            // This needs to be executed every iteration.
            if (PruningTest != null)
            {
                if (PruningTest is TestWindowWithTolerance)
                {
                    if (PruningTest.BestIteration != -1)
                        ch.Info("Iteration {0} \t(Best tolerated validation moving average NDCG@{1} {2}:{3:00.00}~{4:00.00})",
                                Ensemble.NumTrees,
                                _args.earlyStoppingMetrics,
                                PruningTest.BestIteration,
                                100 * (PruningTest as TestWindowWithTolerance).BestAverageValue,
                                100 * (PruningTest as TestWindowWithTolerance).CurrentAverageValue);
                    else
                        ch.Info("Iteration {0}", Ensemble.NumTrees);
                }
                else
                {
                    ch.Info("Iteration {0} \t(best validation NDCG@{1} {2}:{3:00.00}>{4:00.00})",
                            Ensemble.NumTrees,
                            _args.earlyStoppingMetrics,
                            PruningTest.BestIteration,
                            100 * PruningTest.BestResult.FinalValue,
                            100 * PruningTest.ComputeTests().First().FinalValue);
                }
            }
            else
                base.PrintIterationMessage(ch, pch);
#else
            base.PrintIterationMessage(ch, pch);
#endif
        }

        protected override void ComputeTests()
        {
            if (_firstTestSetHistory != null)
                _firstTestSetHistory.ComputeTests();

            if (_specialTrainSetTest != null)
                _specialTrainSetTest.ComputeTests();

            if (PruningTest != null)
                PruningTest.ComputeTests();
        }

        protected override string GetTestGraphLine()
        {
            StringBuilder lineBuilder = new StringBuilder();

            lineBuilder.AppendFormat("Eval:\tnet.{0:D8}.ini", Ensemble.NumTrees - 1);

            foreach (var r in _firstTestSetHistory.ComputeTests())
            {
                lineBuilder.AppendFormat("\t{0:0.0000}", r.FinalValue);
            }

            double trainTestResult = 0.0;
            double validTestResult = 0.0;

            // We only print non-zero train&valid graph if earlyStoppingTruncation!=0
            // In case /es is not set, we print 0 for train and valid graph NDCG
            // Let's keeping this behaviour for backward compatibility with previous FR version
            // Ideally /graphtv should enforce non-zero /es in the commandline validation
            if (_specialTrainSetTest != null)
            {
                trainTestResult = _specialTrainSetTest.ComputeTests().First().FinalValue;
            }

            if (PruningTest != null)
            {
                validTestResult = PruningTest.ComputeTests().First().FinalValue;
            }

            lineBuilder.AppendFormat("\t{0:0.0000}\t{1:0.0000}", trainTestResult, validTestResult);

            return lineBuilder.ToString();
        }

        protected override void Train(IChannel ch)
        {
            base.Train(ch);
            // Print final last iteration.
            // Note that trainNDCG printed in graph will be from copy of a value from previous iteration
            // and will diffre slightly from the proper final value computed by FullTest.
            // We cannot compute the final NDCG here due to the fact we use FastNDCGTestForTrainSet computing NDCG based on label sort saved during gradient computation (and we don;t have gradients for n+1 iteration)
            // Keeping it in sync with original FR code
            PrintTestGraph(ch);
        }

        protected override void CustomizedTrainingIteration(RegressionTree tree)
        {
            Contracts.AssertValueOrNull(tree);
            if (tree != null && Args.CompressEnsemble)
            {
                double[] trainOutputs = Ensemble.GetTreeAt(Ensemble.NumTrees - 1).GetOutputs(TrainSet);
                _ensembleCompressor.SetTreeScores(Ensemble.NumTrees - 1, trainOutputs);
            }
        }

        /// <summary>
        /// Create standard test for dataset.
        /// </summary>
        /// <param name="dataset">dataset used for testing</param>
        /// <returns>standard test for the dataset</returns>
        private Test CreateStandardTest(Dataset dataset)
        {
            if (Utils.Size(dataset.MaxDcg) == 0)
                dataset.Skeleton.RecomputeMaxDcg(10);

            return new NdcgTest(
                ConstructScoreTracker(dataset),
                dataset.Ratings,
                Args.SortingAlgorithm);
        }

        /// <summary>
        /// Create the special test for train set.
        /// </summary>
        /// <returns>test for train set</returns>
        private Test CreateSpecialTrainSetTest()
        {
            return new FastNdcgTestForTrainSet(
                OptimizationAlgorithm.TrainingScores,
                OptimizationAlgorithm.ObjectiveFunction as LambdaRankObjectiveFunction,
                TrainSet.Ratings,
                Args.SortingAlgorithm,
                Args.EarlyStoppingMetrics);
        }

        /// <summary>
        /// Create the special test for valid set.
        /// </summary>
        /// <returns>test for train set</returns>
        private Test CreateSpecialValidSetTest()
        {
            return new FastNdcgTest(
                ConstructScoreTracker(ValidSet),
                ValidSet.Ratings,
                Args.SortingAlgorithm,
                Args.EarlyStoppingMetrics);
        }

        /// <summary>
        /// Create the test for the first test set.
        /// </summary>
        /// <returns>test for the first test set</returns>
        private Test CreateFirstTestSetTest()
        {
            return CreateStandardTest(TestSets[0]);
        }

        /// <summary>
        /// Get the header of test graph
        /// </summary>
        /// <returns>Test graph header</returns>
        protected override string GetTestGraphHeader()
        {
            StringBuilder headerBuilder = new StringBuilder("Eval:\tFileName\tNDCG@1\tNDCG@2\tNDCG@3\tNDCG@4\tNDCG@5\tNDCG@6\tNDCG@7\tNDCG@8\tNDCG@9\tNDCG@10");

            if (Args.PrintTrainValidGraph)
            {
                headerBuilder.Append("\tNDCG@20\tNDCG@40");
                headerBuilder.AppendFormat(
                    "\nNote: Printing train NDCG@{0} as NDCG@20 and validation NDCG@{0} as NDCG@40..\n",
                    Args.EarlyStoppingMetrics);
            }

            return headerBuilder.ToString();
        }

        protected override RankingPredictionTransformer<FastTreeRankingPredictor> MakeTransformer(FastTreeRankingPredictor model, ISchema trainSchema)
        => new RankingPredictionTransformer<FastTreeRankingPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema) => _outputColumns;

        public sealed class LambdaRankObjectiveFunction : ObjectiveFunctionBase, IStepSearch
        {
            private readonly short[] _labels;

            private enum DupeIdInfo
            {
                NoInformation = 0,
                Unique = 1,
                FormatNotSupported = 1000000,
                Code404 = 1000001
            };

            // precomputed arrays
            private readonly double[] _inverseMaxDcgt;
            private readonly double[] _discount;
            private readonly int[] _oneTwoThree;

            private int[][] _labelCounts;

            // reusable memory, technical stuff
            private int[][] _permutationBuffers;
            private DcgPermutationComparer[] _comparers;

            //gains
            private double[] _gain;
            private double[] _gainLabels;

            // parameters
            private int _maxDcgTruncationLevel;
            private bool _trainDcg;
            // A lookup table for the sigmoid used in the lambda calculation
            // Note: Is built for a specific sigmoid parameter, so assumes this will be constant throughout computation
            private double[] _sigmoidTable;
            private double _minScore;       // Computed: range of scores covered in table
            private double _maxScore;
            private double _minSigmoid;
            private double _maxSigmoid;
            private double _scoreToSigmoidTableFactor;
            private const double _expAsymptote = -50;     // exp( x < expAsymptote ) is assumed to be 0
            private const int _sigmoidBins = 1000000;         // Number of bins in the lookup table

            // Secondary gains, currently not used in any way.
#pragma warning disable 0649
            private double _secondaryMetricShare;
            private double[] _secondaryInverseMaxDcgt;
            private double[] _secondaryGains;
#pragma warning restore 0649

            // Baseline risk.
            private static int _iteration = 0; // This is a static class global member which keeps track of the iterations.
            private double[] _baselineDcg;
            private double[] _baselineAlpha;
            private double _baselineAlphaCurrent;
            // Current iteration risk statistics.
            private double _idealNextRisk;
            private double _currentRisk;
            private double _countRisk;

            // These reusable buffers are used for
            // 1. preprocessing the scores for continuous cost function
            // 2. shifted NDCG
            // 3. max DCG per query
            private double[] _scoresCopy;
            private short[] _labelsCopy;
            private short[] _groupIdToTopLabel;

            // parameters
            private double _sigmoidParam;
            private char _costFunctionParam;
            private bool _filterZeroLambdas;

            private bool _distanceWeight2;
            private bool _normalizeQueryLambdas;
            private bool _useShiftedNdcg;
            private IParallelTraining _parallelTraining;

            // Used for training NDCG calculation
            // Keeps track of labels of top 3 documents per query
            public short[][] TrainQueriesTopLabels;

            public LambdaRankObjectiveFunction(Dataset trainset, short[] labels, Arguments args, IParallelTraining parallelTraining)
                : base(trainset,
                    args.LearningRates,
                    args.Shrinkage,
                    args.MaxTreeOutput,
                    args.GetDerivativesSampleRate,
                    args.BestStepRankingRegressionTrees,
                    args.RngSeed)
            {

                _labels = labels;
                TrainQueriesTopLabels = new short[Dataset.NumQueries][];
                for (int q = 0; q < Dataset.NumQueries; ++q)
                    TrainQueriesTopLabels[q] = new short[3];

                _labelCounts = new int[Dataset.NumQueries][];
                int relevancyLevel = DcgCalculator.LabelGainMap.Length;
                for (int q = 0; q < Dataset.NumQueries; ++q)
                    _labelCounts[q] = new int[relevancyLevel];

                // precomputed arrays
                _maxDcgTruncationLevel = args.LambdaMartMaxTruncation;
                _trainDcg = args.TrainDcg;
                if (_trainDcg)
                {
                    _inverseMaxDcgt = new double[Dataset.NumQueries];
                    for (int q = 0; q < Dataset.NumQueries; ++q)
                        _inverseMaxDcgt[q] = 1.0;
                }
                else
                {
                    _inverseMaxDcgt = DcgCalculator.MaxDcg(_labels, Dataset.Boundaries, _maxDcgTruncationLevel, _labelCounts);
                    for (int q = 0; q < Dataset.NumQueries; ++q)
                        _inverseMaxDcgt[q] = 1.0 / _inverseMaxDcgt[q];
                }

                _discount = new double[Dataset.MaxDocsPerQuery];
                FillDiscounts(args.PositionDiscountFreeform);

                _oneTwoThree = new int[Dataset.MaxDocsPerQuery];
                for (int d = 0; d < Dataset.MaxDocsPerQuery; ++d)
                    _oneTwoThree[d] = d;

                // reusable resources
                int numThreads = BlockingThreadPool.NumThreads;
                _comparers = new DcgPermutationComparer[numThreads];
                for (int i = 0; i < numThreads; ++i)
                    _comparers[i] = DcgPermutationComparerFactory.GetDcgPermutationFactory(args.SortingAlgorithm);

                _permutationBuffers = new int[numThreads][];
                for (int i = 0; i < numThreads; ++i)
                    _permutationBuffers[i] = new int[Dataset.MaxDocsPerQuery];

                _gain = Dataset.DatasetSkeleton.LabelGainMap;
                FillGainLabels();

                #region parameters
                _sigmoidParam = args.LearningRates;
                _costFunctionParam = args.CostFunctionParam;
                _distanceWeight2 = args.DistanceWeight2;
                _normalizeQueryLambdas = args.NormalizeQueryLambdas;

                _useShiftedNdcg = args.ShiftedNdcg;
                _filterZeroLambdas = args.FilterZeroLambdas;
                #endregion

                _scoresCopy = new double[Dataset.NumDocs];
                _labelsCopy = new short[Dataset.NumDocs];
                _groupIdToTopLabel = new short[Dataset.NumDocs];

                FillSigmoidTable(_sigmoidParam);
#if OLD_DATALOAD
            SetupSecondaryGains(cmd);
#endif
                SetupBaselineRisk(args);
                _parallelTraining = parallelTraining;
            }

#if OLD_DATALOAD
        private void SetupSecondaryGains(Arguments args)
        {
            _secondaryGains = null;
            _secondaryMetricShare = args.secondaryMetricShare;
            _secondaryIsolabelExclusive = args.secondaryIsolabelExclusive;
            if (_secondaryMetricShare != 0.0)
            {
                _secondaryGains = Dataset.Skeleton.GetData<double>("SecondaryGains");
                if (_secondaryGains == null)
                {
                    _secondaryMetricShare = 0.0;
                    return;
                }
                _secondaryInverseMaxDCGT = DCGCalculator.MaxDCG(_secondaryGains, Dataset.Boundaries,
                    new int[] { args.lambdaMartMaxTruncation })[0].Select(d => 1.0 / d).ToArray();
            }
        }
#endif

            private void SetupBaselineRisk(Arguments args)
            {
                double[] scores = Dataset.Skeleton.GetData<double>("BaselineScores");
                if (scores == null)
                    return;

                // Calculate the DCG with the discounts as they exist in the objective function (this
                // can differ versus the actual DCG discount)
                DcgCalculator calc = new DcgCalculator(Dataset.MaxDocsPerQuery, args.SortingAlgorithm);
                _baselineDcg = calc.DcgFromScores(Dataset, scores, _discount);

                IniFileParserInterface ffi = IniFileParserInterface.CreateFromFreeform(string.IsNullOrEmpty(args.BaselineAlphaRisk) ? "0" : args.BaselineAlphaRisk);
                IniFileParserInterface.FeatureEvaluator ffe = ffi.GetFeatureEvaluators()[0];
                IniFileParserInterface.FeatureMap ffmap = ffi.GetFeatureMap();
                string[] ffnames = Enumerable.Range(0, ffmap.RawFeatureCount)
                    .Select(x => ffmap.GetRawFeatureName(x)).ToArray();
                string[] badffnames = ffnames.Where(x => x != "I" && x != "T").ToArray();
                if (badffnames.Length > 0)
                {
                    // The freeform should contain only I and T, that is, the iteration and total iterations.
                    throw Contracts.Except(
                        "alpha freeform must use only I (iterations) and T (total iterations), contains {0} unrecognized names {1}",
                        badffnames.Length, string.Join(", ", badffnames));
                }

                uint[] vals = new uint[ffmap.RawFeatureCount];
                int iInd = Array.IndexOf(ffnames, "I");
                int tInd = Array.IndexOf(ffnames, "T");
                int totalTrees = args.NumTrees;
                if (tInd >= 0)
                    vals[tInd] = (uint)totalTrees;
                _baselineAlpha = Enumerable.Range(0, totalTrees).Select(i =>
                {
                    if (iInd >= 0)
                        vals[iInd] = (uint)i;
                    return ffe.Evaluate(vals);
                }).ToArray();
            }

            private void FillSigmoidTable(double sigmoidParam)
            {
                // minScore is such that 2*sigmoidParam*score is < expAsymptote if score < minScore
                _minScore = _expAsymptote / sigmoidParam / 2;
                _maxScore = -_minScore;

                _sigmoidTable = new double[_sigmoidBins];
                for (int i = 0; i < _sigmoidBins; i++)
                {
                    double score = (_maxScore - _minScore) / _sigmoidBins * i + _minScore;
                    if (score > 0.0)
                        _sigmoidTable[i] = 2.0 - 2.0 / (1.0 + Math.Exp(-2.0 * sigmoidParam * score));
                    else
                        _sigmoidTable[i] = 2.0 / (1.0 + Math.Exp(2.0 * sigmoidParam * score));
                }
                _scoreToSigmoidTableFactor = _sigmoidBins / (_maxScore - _minScore);
                _minSigmoid = _sigmoidTable[0];
                _maxSigmoid = _sigmoidTable.Last();
            }

            private void IgnoreNonBestDuplicates(short[] labels, double[] scores, int[] order, UInt32[] dupeIds, int begin, int numDocuments)
            {
                if (dupeIds == null || dupeIds.Length == 0)
                {
                    return;
                }

                // Reset top label for all groups
                for (int i = begin; i < begin + numDocuments; ++i)
                {
                    _groupIdToTopLabel[i] = -1;
                }

                for (int i = 0; i < numDocuments; ++i)
                {
                    Contracts.Check(0 <= order[i] && order[i] < numDocuments, "the index to document exceeds range");

                    int index = begin + order[i];

                    UInt32 group = dupeIds[index];
                    if (group == (UInt32)DupeIdInfo.Code404 || group == (UInt32)DupeIdInfo.FormatNotSupported ||
                        group == (UInt32)DupeIdInfo.Unique || group == (UInt32)DupeIdInfo.NoInformation)
                    {
                        continue;
                    }

                    // group starts from 2 (since 0 is unknown and 1 is unique)
                    Contracts.Check(2 <= group && group < numDocuments + 2, "dupeId group exceeds range");

                    UInt32 groupIndex = (UInt32)begin + group - 2;

                    if (_groupIdToTopLabel[groupIndex] != -1)
                    {
                        // this is the second+ occurance of a result
                        // of the same duplicate group, so:
                        // - disconsider when applying the cost function
                        //
                        // Only do this if the rating of this dupe is worse or equal,
                        // otherwise we want this dupe to be pushed to the top
                        // so we keep it
                        if (labels[index] <= _groupIdToTopLabel[groupIndex])
                        {
                            labels[index] = 0;
                            scores[index] = double.MinValue;
                        }
                    }
                    else
                    {
                        _groupIdToTopLabel[groupIndex] = labels[index];
                    }
                }
            }

            public override double[] GetGradient(IChannel ch, double[] scores)
            {
                // Set the risk and alpha accumulators appropriately.
                _countRisk = _currentRisk = _idealNextRisk = 0.0;
                _baselineAlphaCurrent = _baselineAlpha == null ? 0.0 : _baselineAlpha[_iteration];
                double[] grads = base.GetGradient(ch, scores);
                if (_baselineDcg != null)
                {
                    ch.Info(
                        "Risk alpha {0:0.000}, total {1:0.000}, avg {2:0.000}, count {3}, next ideal {4:0.000}",
                        _baselineAlphaCurrent, _currentRisk, _currentRisk / Math.Max(1.0, _countRisk),
                        _countRisk, _idealNextRisk);
                }
                _iteration++;
                return grads;
            }

            protected override void GetGradientInOneQuery(int query, int threadIndex)
            {
                int begin = Dataset.Boundaries[query];
                int numDocuments = Dataset.Boundaries[query + 1] - Dataset.Boundaries[query];

                Array.Clear(Gradient, begin, numDocuments);
                Array.Clear(Weights, begin, numDocuments);

                double inverseMaxDcg = _inverseMaxDcgt[query];
                double secondaryInverseMaxDcg = _secondaryMetricShare == 0 ? 0.0 : _secondaryInverseMaxDcgt[query];

                int[] permutation = _permutationBuffers[threadIndex];

                short[] labels = _labels;
                double[] scoresToUse = Scores;

                if (_useShiftedNdcg)
                {
                    // Copy the labels for this query
                    Array.Copy(_labels, begin, _labelsCopy, begin, numDocuments);
                    labels = _labelsCopy;
                }

                if (_costFunctionParam == 'c' || _useShiftedNdcg)
                {
                    // Copy the scores for this query
                    Array.Copy(Scores, begin, _scoresCopy, begin, numDocuments);
                    scoresToUse = _scoresCopy;
                }

                // Keep track of top 3 labels for later use
                //GetTopQueryLabels(query, permutation, false);

                double lambdaSum = 0;

                unsafe
                {
                    fixed (int* pPermutation = permutation)
                    fixed (short* pLabels = labels)
                    fixed (double* pScores = scoresToUse)
                    fixed (double* pLambdas = Gradient)
                    fixed (double* pWeights = Weights)
                    fixed (double* pDiscount = _discount)
                    fixed (double* pGain = _gain)
                    fixed (double* pGainLabels = _gainLabels)
                    fixed (double* pSigmoidTable = _sigmoidTable)
                    fixed (double* pSecondaryGains = _secondaryGains)
                    fixed (int* pOneTwoThree = _oneTwoThree)
                    {
                        // calculates the permutation that orders "scores" in descending order, without modifying "scores"
                        Array.Copy(_oneTwoThree, permutation, numDocuments);
#if USE_FASTTREENATIVE

                        PermutationSort(permutation, scoresToUse, labels, numDocuments, begin);
                        // Get how far about baseline our current
                        double baselineDcgGap = 0.0;
                        if (_baselineDcg != null)
                        {
                            baselineDcgGap = _baselineDcg[query];
                            for (int d = 0; d < numDocuments; ++d)
                            {
                                baselineDcgGap -= _gainLabels[pPermutation[d] + begin] * _discount[d];
                            }
                            if (baselineDcgGap > 1e-7)
                            {
                                Utils.InterlockedAdd(ref _currentRisk, baselineDcgGap);
                                Utils.InterlockedAdd(ref _countRisk, 1.0);
                            }
                        }
                        //baselineDCGGap = ((new Random(query)).NextDouble() * 2 - 1)/inverseMaxDCG; // THIS IS EVIL CODE REMOVE LATER
                        // Keep track of top 3 labels for later use
                        GetTopQueryLabels(query, permutation, true);

                        if (_useShiftedNdcg)
                        {
                            // Set non-best (rank-wise) duplicates to be ignored. Set Score to MinValue, Label to 0
                            IgnoreNonBestDuplicates(labels, scoresToUse, permutation, Dataset.DupeIds, begin, numDocuments);
                        }

                        int numActualResults = numDocuments;

                        // If the const function is ContinuousWeightedRanknet, update output scores
                        if (_costFunctionParam == 'c')
                        {
                            for (int i = begin; i < begin + numDocuments; ++i)
                            {
                                if (pScores[i] == double.MinValue)
                                {
                                    numActualResults--;
                                }
                                else
                                {
                                    pScores[i] = pScores[i] * (1.0 - pLabels[i] * 1.0 / (20.0 * Dataset.DatasetSkeleton.LabelGainMap.Length));
                                }
                            }
                        }

                        // Continous cost function and shifted NDCG require a re-sort and recomputation of maxDCG
                        // (Change of scores in the former and scores and labels in the latter)
                        if (!_trainDcg && (_costFunctionParam == 'c' || _useShiftedNdcg))
                        {
                            PermutationSort(permutation, scoresToUse, labels, numDocuments, begin);
                            inverseMaxDcg = 1.0 / DcgCalculator.MaxDcgQuery(labels, begin, numDocuments, numDocuments, _labelCounts[query]);
                        }
                        // A constant related to secondary labels, which does not exist in the current codebase.
                        const bool secondaryIsolabelExclusive = false;
                        GetDerivatives(numDocuments, begin, pPermutation, pLabels,
                                pScores, pLambdas, pWeights, pDiscount,
                                inverseMaxDcg, pGainLabels,
                                _secondaryMetricShare, secondaryIsolabelExclusive, secondaryInverseMaxDcg, pSecondaryGains,
                                pSigmoidTable, _minScore, _maxScore, _sigmoidTable.Length, _scoreToSigmoidTableFactor,
                                _costFunctionParam, _distanceWeight2, numActualResults, &lambdaSum, double.MinValue,
                                _baselineAlphaCurrent, baselineDcgGap);

                        // For computing the "ideal" case of the DCGs.
                        if (_baselineDcg != null)
                        {
                            if (scoresToUse == Scores)
                                Array.Copy(Scores, begin, _scoresCopy, begin, numDocuments);
                            for (int i = begin; i < begin + numDocuments; ++i)
                            {
                                _scoresCopy[i] += Gradient[i] / Weights[i];
                            }
                            Array.Copy(_oneTwoThree, permutation, numDocuments);
                            PermutationSort(permutation, _scoresCopy, labels, numDocuments, begin);
                            double idealNextRisk = _baselineDcg[query];
                            for (int d = 0; d < numDocuments; ++d)
                            {
                                idealNextRisk -= _gainLabels[pPermutation[d] + begin] * _discount[d];
                            }
                            if (idealNextRisk > 1e-7)
                            {
                                Utils.InterlockedAdd(ref _idealNextRisk, idealNextRisk);
                            }
                        }

#else
                        if (_useShiftedNdcg || _costFunctionParam == 'c' || _distanceWeight2 || _normalizeQueryLambdas)
                        {
                            throw new Exception("Shifted NDCG / ContinuousWeightedRanknet / distanceWeight2 / normalized lambdas are only supported by unmanaged code");
                        }

                        var comparer = _comparers[threadIndex];
                        comparer.Scores = scoresToUse;
                        comparer.Labels = labels;
                        comparer.ScoresOffset = begin;
                        comparer.LabelsOffset = begin;
                        Array.Sort(permutation, 0, numDocuments, comparer);

                        // go over all pairs
                        double scoreHighMinusLow;
                        double lambdaP;
                        double weightP;
                        double deltaNdcgP;
                        for (int i = 0; i < numDocuments; ++i)
                        {
                            int high = begin + pPermutation[i];
                            if (pLabels[high] == 0)
                                continue;
                            double deltaLambdasHigh = 0;
                            double deltaWeightsHigh = 0;

                            for (int j = 0; j < numDocuments; ++j)
                            {
                                // only consider pairs with different labels, where "high" has a higher label than "low"
                                if (i == j)
                                    continue;
                                int low = begin + pPermutation[j];
                                if (pLabels[high] <= pLabels[low])
                                    continue;

                                // calculate the lambdaP for this pair
                                scoreHighMinusLow = pScores[high] - pScores[low];

                                if (scoreHighMinusLow <= _minScore)
                                    lambdaP = _minSigmoid;
                                else if (scoreHighMinusLow >= _maxScore)
                                    lambdaP = _maxSigmoid;
                                else
                                    lambdaP = _sigmoidTable[(int)((scoreHighMinusLow - _minScore) * _scoreToSigmoidTableFactor)];

                                weightP = lambdaP * (2.0 - lambdaP);

                                // calculate the deltaNDCGP for this pair
                                deltaNdcgP =
                                    (pGain[pLabels[high]] - pGain[pLabels[low]]) *
                                    Math.Abs((pDiscount[i] - pDiscount[j])) *
                                    inverseMaxDcg;

                                // update lambdas and weights
                                deltaLambdasHigh += lambdaP * deltaNdcgP;
                                pLambdas[low] -= lambdaP * deltaNdcgP;
                                deltaWeightsHigh += weightP * deltaNdcgP;
                                pWeights[low] += weightP * deltaNdcgP;
                            }
                            pLambdas[high] += deltaLambdasHigh;
                            pWeights[high] += deltaWeightsHigh;
                        }
#endif
                        if (_normalizeQueryLambdas)
                        {
                            if (lambdaSum > 0)
                            {
                                double normFactor = (10 * Math.Log(1 + lambdaSum)) / lambdaSum;

                                for (int i = begin; i < begin + numDocuments; ++i)
                                {
                                    pLambdas[i] = pLambdas[i] * normFactor;
                                    pWeights[i] = pWeights[i] * normFactor;
                                }
                            }
                        }
                    }
                }
            }

            public void AdjustTreeOutputs(IChannel ch, RegressionTree tree, DocumentPartitioning partitioning,
                                            ScoreTracker trainingScores)
            {
                const double epsilon = 1.4e-45;
                double[] means = null;
                if (!BestStepRankingRegressionTrees)
                    means = _parallelTraining.GlobalMean(Dataset, tree, partitioning, Weights, _filterZeroLambdas);
                for (int l = 0; l < tree.NumLeaves; ++l)
                {
                    double output = tree.LeafValue(l);
                    if (!BestStepRankingRegressionTrees)
                        output = (output + epsilon) / (2.0 * means[l] + epsilon);

                    if (output > MaxTreeOutput)
                        output = MaxTreeOutput;
                    else if (output < -MaxTreeOutput)
                        output = -MaxTreeOutput;

                    tree.SetLeafValue(l, output);
                }
            }

            private void FillDiscounts(string positionDiscountFreeform)
            {
                if (positionDiscountFreeform == null)
                {
                    for (int d = 0; d < Dataset.MaxDocsPerQuery; ++d)
                        _discount[d] = 1.0 / Math.Log(2.0 + d);
                }
                else
                {
                    IniFileParserInterface inip = IniFileParserInterface.CreateFromFreeform(positionDiscountFreeform);
                    if (inip.GetFeatureMap().RawFeatureCount != 1)
                    {
                        throw Contracts.Except(
                            "The position discount freeform requires exactly 1 variable, {0} encountered",
                            inip.GetFeatureMap().RawFeatureCount);
                    }
                    var freeformEval = inip.GetFeatureEvaluators()[0];
                    uint[] p = new uint[1];
                    for (int d = 0; d < Dataset.MaxDocsPerQuery; ++d)
                    {
                        p[0] = (uint)d;
                        _discount[d] = freeformEval.Evaluate(p);
                    }
                }
            }

            private void FillGainLabels()
            {
                _gainLabels = new double[Dataset.NumDocs];
                for (int i = 0; i < Dataset.NumDocs; i++)
                {
                    _gainLabels[i] = _gain[_labels[i]];
                }
            }

            // Keep track of top 3 labels for later use.
            private void GetTopQueryLabels(int query, int[] permutation, bool bAlreadySorted)
            {
                int numDocuments = Dataset.Boundaries[query + 1] - Dataset.Boundaries[query];
                int begin = Dataset.Boundaries[query];

                if (!bAlreadySorted)
                {
                    // calculates the permutation that orders "scores" in descending order, without modifying "scores"
                    Array.Copy(_oneTwoThree, permutation, numDocuments);
                    PermutationSort(permutation, Scores, _labels, numDocuments, begin);
                }

                for (int i = 0; i < 3 && i < numDocuments; ++i)
                    TrainQueriesTopLabels[query][i] = _labels[begin + permutation[i]];
            }

            private static void PermutationSort(int[] permutation, double[] scores, short[] labels, int numDocs, int shift)
            {
                Contracts.AssertValue(permutation);
                Contracts.AssertValue(scores);
                Contracts.AssertValue(labels);
                Contracts.Assert(numDocs > 0);
                Contracts.Assert(shift >= 0);
                Contracts.Assert(scores.Length - numDocs >= shift);
                Contracts.Assert(labels.Length - numDocs >= shift);

                Array.Sort(permutation, 0, numDocs,
                    Comparer<int>.Create((x, y) =>
                    {
                        if (scores[shift + x] > scores[shift + y])
                            return -1;
                        if (scores[shift + x] < scores[shift + y])
                            return 1;
                        if (labels[shift + x] < labels[shift + y])
                            return -1;
                        if (labels[shift + x] > labels[shift + y])
                            return 1;
                        return x - y;
                    }));
            }

            [DllImport("FastTreeNative", EntryPoint = "C_GetDerivatives", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
            private static extern unsafe void GetDerivatives(
                int numDocuments, int begin, int* pPermutation, short* pLabels,
                double* pScores, double* pLambdas, double* pWeights, double* pDiscount,
                double inverseMaxDcg, double* pGainLabels,
                double secondaryMetricShare, [MarshalAs(UnmanagedType.U1)] bool secondaryExclusive, double secondaryInverseMaxDcg, double* pSecondaryGains,
                double* lambdaTable, double minScore, double maxScore,
                int lambdaTableLength, double scoreToLambdaTableFactor,
                char costFunctionParam, [MarshalAs(UnmanagedType.U1)] bool distanceWeight2, int numActualDocuments,
                double* pLambdaSum, double doubleMinValue, double alphaRisk, double baselineVersusCurrentDcg);

        }
    }

    public sealed class FastTreeRankingPredictor : FastTreePredictionWrapper
    {
        public const string LoaderSignature = "FastTreeRankerExec";
        public const string RegistrationName = "FastTreeRankingPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FTREE RA",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010002;

        protected override uint VerDefaultValueSerialized => 0x00010004;

        protected override uint VerCategoricalSplitSerialized => 0x00010005;

        internal FastTreeRankingPredictor(IHostEnvironment env, Ensemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private FastTreeRankingPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public static FastTreeRankingPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            return new FastTreeRankingPredictor(env, ctx);
        }

        public override PredictionKind PredictionKind => PredictionKind.Ranking;
    }

    public static partial class FastTree
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastTreeRanker",
            Desc = FastTreeRankingTrainer.Summary,
            UserName = FastTreeRankingTrainer.UserNameValue,
            ShortName = FastTreeRankingTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/member[@name=""FastTree""]/*' />",
                                 @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/example[@name=""FastTreeRanker""]/*' />"})]
        public static CommonOutputs.RankingOutput TrainRanking(IHostEnvironment env, FastTreeRankingTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainFastTree");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<FastTreeRankingTrainer.Arguments, CommonOutputs.RankingOutput>(host, input,
                () => new FastTreeRankingTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.GroupIdColumn));
        }
    }
}
