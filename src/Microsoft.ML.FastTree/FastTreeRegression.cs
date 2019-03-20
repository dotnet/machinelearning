// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(FastTreeRegressionTrainer.Summary, typeof(FastTreeRegressionTrainer), typeof(FastTreeRegressionTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastTreeRegressionTrainer.UserNameValue,
    FastTreeRegressionTrainer.LoadNameValue,
    FastTreeRegressionTrainer.ShortName,

    // FastRank names
    "FastRankRegression",
    "FastRankRegressionWrapper",
    "frr",
    "btr")]

[assembly: LoadableClass(typeof(FastTreeRegressionModelParameters), null, typeof(SignatureLoadModel),
    "FastTree Regression Executor",
    FastTreeRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a decision tree regression model using FastTree.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="FastTree_remarks"]/*' />
    public sealed partial class FastTreeRegressionTrainer
        : BoostingFastTreeTrainerBase<FastTreeRegressionTrainer.Options, RegressionPredictionTransformer<FastTreeRegressionModelParameters>, FastTreeRegressionModelParameters>
    {
        internal const string LoadNameValue = "FastTreeRegression";
        internal const string UserNameValue = "FastTree (Boosted Trees) Regression";
        internal const string Summary = "Trains gradient boosted decision trees to fit target values using least-squares.";
        internal const string ShortName = "ftr";

        private TestHistory _firstTestSetHistory;
        private Test _trainRegressionTest;
        private Test _testRegressionTest;

        /// <summary>
        /// The type of prediction for the trainer.
        /// </summary>
        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name for the column containing the example weight.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of examples allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="numberOfLeaves">The max number of leaves in each regression tree.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        internal FastTreeRegressionTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate)
            : base(env, TrainerUtils.MakeR4ScalarColumn(labelColumnName), featureColumnName, exampleWeightColumnName, null, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate)
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeRegressionTrainer"/> by using the <see cref="Options"/> class.
        /// </summary>
        /// <param name="env">The instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        internal FastTreeRegressionTrainer(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
        }

        private protected override FastTreeRegressionModelParameters TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;
            ValidData = context.ValidationSet;
            TestData = context.TestSet;

            using (var ch = Host.Start("Training"))
            {
                trainData.CheckRegressionLabel();
                trainData.CheckFeatureFloatVector();
                trainData.CheckOptFloatWeight();
                FeatureCount = trainData.Schema.Feature.Value.Type.GetValueCount();
                ConvertData(trainData);
                TrainCore(ch);
            }
            return new FastTreeRegressionModelParameters(Host, TrainedEnsemble, FeatureCount, InnerOptions);
        }

        private protected override void CheckOptions(IChannel ch)
        {
            Contracts.AssertValue(ch);

            base.CheckOptions(ch);

            bool doEarlyStop = FastTreeTrainerOptions.EarlyStoppingRuleFactory != null ||
                FastTreeTrainerOptions.EnablePruning;

            if (doEarlyStop)
                ch.CheckUserArg(FastTreeTrainerOptions.EarlyStoppingMetrics >= 1 && FastTreeTrainerOptions.EarlyStoppingMetrics <= 2,
                    nameof(FastTreeTrainerOptions.EarlyStoppingMetrics), "earlyStoppingMetrics should be 1 or 2. (1: L1, 2: L2)");
        }

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
        {
            return new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false);
        }

        private protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch)
        {
            return new ObjectiveImpl(TrainSet, FastTreeTrainerOptions);
        }

        private protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            if (FastTreeTrainerOptions.UseLineSearch)
            {
                var lossCalculator = new RegressionTest(optimizationAlgorithm.TrainingScores);
                // REVIEW: We should make loss indices an enum in BinaryClassificationTest.
                optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, 1 /*L2 error*/, FastTreeTrainerOptions.MaximumNumberOfLineSearchSteps, FastTreeTrainerOptions.MinimumStepSize);
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
        internal static float[] GetDatasetRegressionLabels(Dataset set)
        {
            if (set == null)
                return null;
            double[] dlabels = set.Targets;
            Contracts.AssertValue(dlabels);
            Contracts.Assert(dlabels.Length == set.NumDocs);
            // REVIEW: Seems wasteful??
            return dlabels.Select(x => (float)x).ToArray(dlabels.Length);
        }

        private protected override void PrepareLabels(IChannel ch)
        {
        }

        private protected override Test ConstructTestForTrainingData()
        {
            return new RegressionTest(ConstructScoreTracker(TrainSet));
        }

        private protected override RegressionPredictionTransformer<FastTreeRegressionModelParameters> MakeTransformer(FastTreeRegressionModelParameters model, DataViewSchema trainSchema)
            => new RegressionPredictionTransformer<FastTreeRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="FastTreeRegressionTrainer"/> using both training and validation data, returns
        /// a <see cref="RegressionPredictionTransformer{FastTreeRegressionModelParameters}"/>.
        /// </summary>
        public RegressionPredictionTransformer<FastTreeRegressionModelParameters> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
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

        private protected override void InitializeTests()
        {
            // Initialize regression tests.
            if (FastTreeTrainerOptions.TestFrequency != int.MaxValue)
                AddFullRegressionTests();

            if (FastTreeTrainerOptions.PrintTestGraph)
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

            if (FastTreeTrainerOptions.PrintTrainValidGraph && _trainRegressionTest == null)
            {
                Test trainRegressionTest = new RegressionTest(ConstructScoreTracker(TrainSet));
                _trainRegressionTest = trainRegressionTest;
            }

            if (FastTreeTrainerOptions.PrintTrainValidGraph && _testRegressionTest == null && TestSets != null && TestSets.Length > 0)
                _testRegressionTest = new RegressionTest(ConstructScoreTracker(TestSets[0]));

            // Add early stopping if appropriate.
            TrainTest = new RegressionTest(ConstructScoreTracker(TrainSet), FastTreeTrainerOptions.EarlyStoppingMetrics);
            if (ValidSet != null)
                ValidTest = new RegressionTest(ConstructScoreTracker(ValidSet), FastTreeTrainerOptions.EarlyStoppingMetrics);

            if (FastTreeTrainerOptions.EnablePruning && ValidTest != null)
            {
                if (FastTreeTrainerOptions.UseTolerantPruning) // Use simple early stopping condition.
                    PruningTest = new TestWindowWithTolerance(ValidTest, 0, FastTreeTrainerOptions.PruningWindowSize, FastTreeTrainerOptions.PruningThreshold);
                else
                    PruningTest = new TestHistory(ValidTest, 0);
            }
        }

        private protected override void PrintIterationMessage(IChannel ch, IProgressChannel pch)
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

        private protected override string GetTestGraphHeader()
        {
            StringBuilder headerBuilder = new StringBuilder("Eval:\tFileName\tNDCG@1\tNDCG@2\tNDCG@3\tNDCG@4\tNDCG@5\tNDCG@6\tNDCG@7\tNDCG@8\tNDCG@9\tNDCG@10");

            if (FastTreeTrainerOptions.PrintTrainValidGraph)
            {
                headerBuilder.Append("\tNDCG@20\tNDCG@40");
                headerBuilder.Append("\nNote: Printing train L2 error as NDCG@20 and test L2 error as NDCG@40..\n");
            }

            return headerBuilder.ToString();
        }

        private protected override void ComputeTests()
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

        private protected override string GetTestGraphLine()
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

        private protected override void Train(IChannel ch)
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

            public ObjectiveImpl(Dataset trainData, GamRegressionTrainer.Options options) :
                base(
                    trainData,
                    options.LearningRate,
                    0,
                    options.MaximumTreeOutput,
                    options.GetDerivativesSampleRate,
                    false,
                    options.Seed)
            {
                _labels = GetDatasetRegressionLabels(trainData);
            }

            public ObjectiveImpl(Dataset trainData, Options options)
                : base(
                    trainData,
                    options.LearningRate,
                    options.Shrinkage,
                    options.MaximumTreeOutput,
                    options.GetDerivativesSampleRate,
                    options.BestStepRankingRegressionTrees,
                    options.Seed)
            {
                if (options.DropoutRate > 0 && LearningRate > 0) // Don't do shrinkage if dropouts are used.
                    Shrinkage = 1.0 / LearningRate;

                _labels = GetDatasetRegressionLabels(trainData);
            }

            public void AdjustTreeOutputs(IChannel ch, InternalRegressionTree tree, DocumentPartitioning partitioning, ScoreTracker trainingScores)
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

    public sealed class FastTreeRegressionModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "FastTreeRegressionExec";
        internal const string RegistrationName = "FastTreeRegressionPredictor";

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
                loaderAssemblyName: typeof(FastTreeRegressionModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010002;

        private protected override uint VerDefaultValueSerialized => 0x00010004;

        private protected override uint VerCategoricalSplitSerialized => 0x00010005;

        internal FastTreeRegressionModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private FastTreeRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static FastTreeRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new FastTreeRegressionModelParameters(env, ctx);
        }

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;
    }

    internal static partial class FastTree
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastTreeRegressor",
            Desc = FastTreeRegressionTrainer.Summary,
            UserName = FastTreeRegressionTrainer.UserNameValue,
            ShortName = FastTreeRegressionTrainer.ShortName)]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, FastTreeRegressionTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainFastTree");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<FastTreeRegressionTrainer.Options, CommonOutputs.RegressionOutput>(host, input,
                () => new FastTreeRegressionTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.RowGroupColumnName));
        }
    }
}
