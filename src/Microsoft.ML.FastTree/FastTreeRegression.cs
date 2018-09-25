// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;
using System.Linq;
using System.Text;

[assembly: LoadableClass(FastTreeRegressionTrainer.Summary, typeof(FastTreeRegressionTrainer), typeof(FastTreeRegressionTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastTreeRegressionTrainer.UserNameValue,
    FastTreeRegressionTrainer.LoadNameValue,
    FastTreeRegressionTrainer.ShortName,

    // FastRank names
    "FastRankRegression",
    "FastRankRegressionWrapper",
    "frr",
    "btr")]

[assembly: LoadableClass(typeof(FastTreeRegressionPredictor), null, typeof(SignatureLoadModel),
    "FastTree Regression Executor",
    FastTreeRegressionPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.FastTree
{
    /// <include file='doc.xml' path='doc/members/member[@name="FastTree"]/*' />
    public sealed partial class FastTreeRegressionTrainer
        : BoostingFastTreeTrainerBase<FastTreeRegressionTrainer.Arguments, RegressionPredictionTransformer<FastTreeRegressionPredictor>, FastTreeRegressionPredictor>
    {
        public const string LoadNameValue = "FastTreeRegression";
        internal const string UserNameValue = "FastTree (Boosted Trees) Regression";
        internal const string Summary = "Trains gradient boosted decision trees to fit target values using least-squares.";
        internal const string ShortName = "ftr";

        private TestHistory _firstTestSetHistory;
        private Test _trainRegressionTest;
        private Test _testRegressionTest;

        /// <summary>
        /// The type of prediction for the trainer.
        /// </summary>
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        /// <param name="learningRates">The learning rate.</param>
        /// <param name="minDocumentsInLeafs">The minimal number of documents allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="numLeaves">The max number of leaves in each regression tree.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        public FastTreeRegressionTrainer(IHostEnvironment env,
            string labelColumn,
            string featureColumn,
            string weightColumn = null,
            int numLeaves = 20,
            int numTrees = 100,
            int minDocumentsInLeafs = 10,
            double learningRates = 0.2,
            Action<Arguments> advancedSettings = null)
            : base(env, TrainerUtils.MakeR4ScalarLabel(labelColumn), featureColumn, weightColumn, null, advancedSettings)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));

            if (advancedSettings != null)
            {
                using (var ch = Host.Start("Validating advanced settings."))
                {
                    //take a quick snapshot at the defaults, for comparison with the current args values
                    var snapshot = new Arguments();

                    // Check that the user didn't supply different parameters in the args, from what it specified directly.
                    TrainerUtils.CheckArgsAndAdvancedSettingMismatch(ch, numLeaves, snapshot.NumLeaves, Args.NumLeaves, nameof(numLeaves));
                    TrainerUtils.CheckArgsAndAdvancedSettingMismatch(ch, numTrees, snapshot.NumTrees, Args.NumTrees, nameof(numTrees));
                    TrainerUtils.CheckArgsAndAdvancedSettingMismatch(ch, minDocumentsInLeafs, snapshot.MinDocumentsInLeafs, Args.MinDocumentsInLeafs, nameof(minDocumentsInLeafs));
                    TrainerUtils.CheckArgsAndAdvancedSettingMismatch(ch, numLeaves, snapshot.NumLeaves, Args.NumLeaves, nameof(numLeaves));
                    ch.Done();
                }
            }
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeRegressionTrainer"/> by using the legacy <see cref="Arguments"/> class.
        /// </summary>
        internal FastTreeRegressionTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, TrainerUtils.MakeR4ScalarLabel(args.LabelColumn))
        {
        }

        protected override FastTreeRegressionPredictor TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;
            ValidData = context.ValidationSet;

            using (var ch = Host.Start("Training"))
            {
                trainData.CheckRegressionLabel();
                trainData.CheckFeatureFloatVector();
                trainData.CheckOptFloatWeight();
                FeatureCount = trainData.Schema.Feature.Type.ValueCount;
                ConvertData(trainData);
                TrainCore(ch);
                ch.Done();
            }
            return new FastTreeRegressionPredictor(Host, TrainedEnsemble, FeatureCount, InnerArgs);
        }

        protected override void CheckArgs(IChannel ch)
        {
            Contracts.AssertValue(ch);

            base.CheckArgs(ch);

            ch.CheckUserArg((Args.EarlyStoppingRule == null && !Args.EnablePruning) || (Args.EarlyStoppingMetrics >= 1 && Args.EarlyStoppingMetrics <= 2), nameof(Args.EarlyStoppingMetrics),
                    "earlyStoppingMetrics should be 1 or 2. (1: L1, 2: L2)");
        }

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
        {
            return new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);
        }

        protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch)
        {
            return new ObjectiveImpl(TrainSet, Args);
        }

        protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            if (Args.UseLineSearch)
            {
                var lossCalculator = new RegressionTest(optimizationAlgorithm.TrainingScores);
                // REVIEW: We should make loss indices an enum in BinaryClassificationTest.
                optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, 1 /*L2 error*/, Args.NumPostBracketSteps, Args.MinStepSize);
            }

            return optimizationAlgorithm;
        }

        /// <summary>
        /// Gets the regression labels that were stored in the dataset skeleton, or
        /// constructs them from the ratings if absent. This returns null if the
        /// dataset itself is null.
        /// </summary>
        /// <param name="set">The dataset</param>
        /// <returns>The list of regression targets, or null if <paramref name="set"/> was null</returns>
        public static float[] GetDatasetRegressionLabels(Dataset set)
        {
            if (set == null)
                return null;
            double[] dlabels = set.Targets;
            Contracts.AssertValue(dlabels);
            Contracts.Assert(dlabels.Length == set.NumDocs);
            // REVIEW: Seems wasteful??
            return dlabels.Select(x => (float)x).ToArray(dlabels.Length);
        }

        protected override void PrepareLabels(IChannel ch)
        {
        }

        protected override Test ConstructTestForTrainingData()
        {
            return new RegressionTest(ConstructScoreTracker(TrainSet));
        }

        protected override RegressionPredictionTransformer<FastTreeRegressionPredictor> MakeTransformer(FastTreeRegressionPredictor model, ISchema trainSchema)
            => new RegressionPredictionTransformer<FastTreeRegressionPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        private void AddFullRegressionTests()
        {
            // Always compute training L1/L2 errors.
            Tests.Add(new RegressionTest(ConstructScoreTracker(TrainSet)));
            RegressionTest validTest = null;
            if (ValidSet != null)
            {
                validTest = new RegressionTest(ConstructScoreTracker(ValidSet));
                Tests.Add(validTest);
            }

            // If external label is missing use Rating column for L1/L2 error.
            // The values may not make much sense if regression value is not an actual label value.
            if (TestSets != null)
            {
                for (int t = 0; t < TestSets.Length; ++t)
                    Tests.Add(new RegressionTest(ConstructScoreTracker(TestSets[t])));
            }
        }

#if OLD_TRACING
        protected virtual void AddFullNDCGTests()
        {
            Tests.Add(new NDCGTest(ConstructScoreTracker(TrainSet), TrainSet.Ratings, _args.sortingAlgorithm));
            if (ValidSet != null)
            {
                Test test = new NDCGTest(ConstructScoreTracker(ValidSet), ValidSet.Ratings, _args.sortingAlgorithm);
                Tests.Add(test);
            }

            if (TestSets != null)
            {
                for (int t = 0; t < TestSets.Length; ++t)
                {
                    Test test = new NDCGTest(ConstructScoreTracker(TestSets[t]), TestSets[t].Ratings, _args.sortingAlgorithm);

                    if (t == 0)
                    {
                        _firstTestSetHistory = new TestHistory(test, 0);
                    }

                    Tests.Add(test);
                }
            }
        }
#endif

        protected override void InitializeTests()
        {
            // Initialize regression tests.
            if (Args.TestFrequency != int.MaxValue)
                AddFullRegressionTests();

            if (Args.PrintTestGraph)
            {
                // If FirstTestHistory is null (which means the tests were not intialized due to /tf==infinity),
                // we need initialize first set for graph printing.
                // Adding to a tests would result in printing the results after final iteration.
                if (_firstTestSetHistory == null)
                {
                    var firstTestSetTest = new RegressionTest(ConstructScoreTracker(TestSets[0]));
                    _firstTestSetHistory = new TestHistory(firstTestSetTest, 0);
                }
            }

            if (Args.PrintTrainValidGraph && _trainRegressionTest == null)
            {
                Test trainRegressionTest = new RegressionTest(ConstructScoreTracker(TrainSet));
                _trainRegressionTest = trainRegressionTest;
            }

            if (Args.PrintTrainValidGraph && _testRegressionTest == null && TestSets != null && TestSets.Length > 0)
                _testRegressionTest = new RegressionTest(ConstructScoreTracker(TestSets[0]));

            // Add early stopping if appropriate.
            TrainTest = new RegressionTest(ConstructScoreTracker(TrainSet), Args.EarlyStoppingMetrics);
            if (ValidSet != null)
                ValidTest = new RegressionTest(ConstructScoreTracker(ValidSet), Args.EarlyStoppingMetrics);

            if (Args.EnablePruning && ValidTest != null)
            {
                if (Args.UseTolerantPruning) // Use simple early stopping condition.
                    PruningTest = new TestWindowWithTolerance(ValidTest, 0, Args.PruningWindowSize, Args.PruningThreshold);
                else
                    PruningTest = new TestHistory(ValidTest, 0);
            }
        }

        protected override void PrintIterationMessage(IChannel ch, IProgressChannel pch)
        {
            // REVIEW: Shift this to use progress channels.
#if OLD_TRACING
            ch.Info("Finished iteration {0}", Ensemble.NumTrees);

            //This needs to be executed every iteration
            if (PruningTest != null)
            {
                if (PruningTest is TestWindowWithTolerance)
                {
                    if (PruningTest.BestIteration != -1)
                    {
                        ch.Info("Iteration {0} \t(Best tolerated validation moving average iter {1}:{2}~{3})",
                                Ensemble.NumTrees,
                                PruningTest.BestIteration,
                                (PruningTest as TestWindowWithTolerance).BestAverageValue,
                                (PruningTest as TestWindowWithTolerance).CurrentAverageValue);
                    }
                    else
                    {
                        ch.Info("Iteration {0}", Ensemble.NumTrees);
                    }
                }
                else
                {
                    ch.Info("Iteration {0} \t(best validation iter {1}:{2}>{3})",
                            Ensemble.NumTrees,
                            PruningTest.BestIteration,
                            PruningTest.BestResult.FinalValue,
                            PruningTest.ComputeTests().First().FinalValue);
                }
            }
            else
                base.PrintIterationMessage(ch, pch);
#else
            base.PrintIterationMessage(ch, pch);
#endif
        }

        protected override string GetTestGraphHeader()
        {
            StringBuilder headerBuilder = new StringBuilder("Eval:\tFileName\tNDCG@1\tNDCG@2\tNDCG@3\tNDCG@4\tNDCG@5\tNDCG@6\tNDCG@7\tNDCG@8\tNDCG@9\tNDCG@10");

            if (Args.PrintTrainValidGraph)
            {
                headerBuilder.Append("\tNDCG@20\tNDCG@40");
                headerBuilder.Append("\nNote: Printing train L2 error as NDCG@20 and test L2 error as NDCG@40..\n");
            }

            return headerBuilder.ToString();
        }

        protected override void ComputeTests()
        {
            if (_firstTestSetHistory != null)
            {
                _firstTestSetHistory.ComputeTests();
            }

            if (_trainRegressionTest != null)
            {
                _trainRegressionTest.ComputeTests();
            }

            if (_testRegressionTest != null)
            {
                _testRegressionTest.ComputeTests();
            }

            if (PruningTest != null)
            {
                PruningTest.ComputeTests();
            }
        }

        protected override string GetTestGraphLine()
        {
            StringBuilder lineBuilder = new StringBuilder();

            lineBuilder.AppendFormat("Eval:\tnet.{0:D8}.ini", Ensemble.NumTrees - 1);

            foreach (var r in _firstTestSetHistory.ComputeTests())
            {
                lineBuilder.AppendFormat("\t{0:0.0000}", r.FinalValue);
            }

            double trainRegression = 0.0;
            double validRegression = 0.0;

            // We only print non-zero train&valid graph if earlyStoppingTruncation!=0.
            // In case /es is not set, we print 0 for train and valid graph NDCG.
            // Let's keeping this behaviour for backward compatibility with previous FR version.
            // Ideally /graphtv should enforce non-zero /es in the commandline validation.
            if (_trainRegressionTest != null)
                trainRegression = _trainRegressionTest.ComputeTests().Last().FinalValue;
            if (_testRegressionTest != null)
                validRegression = _testRegressionTest.ComputeTests().Last().FinalValue;

            lineBuilder.AppendFormat("\t{0:0.0000}\t{1:0.0000}", trainRegression, validRegression);

            return lineBuilder.ToString();
        }

        protected override void Train(IChannel ch)
        {
            base.Train(ch);
            // Print final last iteration.
            // Note that trainNDCG printed in graph will be from copy of a value from previous iteration
            // and will differ slightly from the proper final value computed by FullTest.
            // We cannot compute the final NDCG here due to the fact we use FastNDCGTestForTrainSet
            // computing NDCG based on label sort saved during gradient computation (and we don't have
            // gradients for n+1 iteration).
            // Keeping it in sync with original FR code
            PrintTestGraph(ch);
        }

        internal sealed class ObjectiveImpl : ObjectiveFunctionBase, IStepSearch
        {
            private readonly float[] _labels;

            public ObjectiveImpl(Dataset trainData, RegressionGamTrainer.Arguments args) :
                base(
                    trainData,
                    args.LearningRates,
                    0,
                    args.MaxOutput,
                    args.GetDerivativesSampleRate,
                    false,
                    args.RngSeed)
            {
                _labels = GetDatasetRegressionLabels(trainData);
            }

            public ObjectiveImpl(Dataset trainData, Arguments args)
                : base(
                    trainData,
                    args.LearningRates,
                    args.Shrinkage,
                    args.MaxTreeOutput,
                    args.GetDerivativesSampleRate,
                    args.BestStepRankingRegressionTrees,
                    args.RngSeed)
            {
                if (args.DropoutRate > 0 && LearningRate > 0) // Don't do shrinkage if dropouts are used.
                    Shrinkage = 1.0 / LearningRate;

                _labels = GetDatasetRegressionLabels(trainData);
            }

            public void AdjustTreeOutputs(IChannel ch, RegressionTree tree, DocumentPartitioning partitioning, ScoreTracker trainingScores)
            {
                double shrinkage = LearningRate * Shrinkage;
                for (int l = 0; l < tree.NumLeaves; ++l)
                {
                    double output = tree.GetOutput(l) * shrinkage;
                    tree.SetOutput(l, output);
                }
            }

            protected override void GetGradientInOneQuery(int query, int threadIndex)
            {
                int begin = Dataset.Boundaries[query];
                int end = Dataset.Boundaries[query + 1];

                // Gradient.
                unchecked
                {
                    for (int i = begin; i < end; ++i)
                        Gradient[i] = _labels[i] - Scores[i];
                }
            }
        }
    }

    public sealed class FastTreeRegressionPredictor : FastTreePredictionWrapper
    {
        public const string LoaderSignature = "FastTreeRegressionExec";
        public const string RegistrationName = "FastTreeRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FTREE RE",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                //verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FastTreeRegressionPredictor).Assembly.FullName);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010002;

        protected override uint VerDefaultValueSerialized => 0x00010004;

        protected override uint VerCategoricalSplitSerialized => 0x00010005;

        internal FastTreeRegressionPredictor(IHostEnvironment env, Ensemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private FastTreeRegressionPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public static FastTreeRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new FastTreeRegressionPredictor(env, ctx);
        }

        public override PredictionKind PredictionKind => PredictionKind.Regression;
    }

    public static partial class FastTree
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastTreeRegressor",
            Desc = FastTreeRegressionTrainer.Summary,
            UserName = FastTreeRegressionTrainer.UserNameValue,
            ShortName = FastTreeRegressionTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/member[@name=""FastTree""]/*' />",
                                 @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/example[@name=""FastTreeRegressor""]/*' />"})]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, FastTreeRegressionTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainFastTree");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<FastTreeRegressionTrainer.Arguments, CommonOutputs.RegressionOutput>(host, input,
                () => new FastTreeRegressionTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.GroupIdColumn));
        }
    }
}
