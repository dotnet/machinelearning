// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(FastTreeBinaryTrainer.Summary, typeof(FastTreeBinaryTrainer), typeof(FastTreeBinaryTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastTreeBinaryTrainer.UserNameValue,
    FastTreeBinaryTrainer.LoadNameValue,
    "FastTreeClassification",
    "FastTree",
    "ft",
    FastTreeBinaryTrainer.ShortName,

    // FastRank names
    "FastRankBinaryClassification",
    "FastRankBinaryClassificationWrapper",
    "FastRankClassification",
    "fr",
    "btc",
    "frc",
    "fastrank",
    "fastrankwrapper")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(FastTreeBinaryModelParameters), null, typeof(SignatureLoadModel),
    "FastTree Binary Executor",
    FastTreeBinaryModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    public sealed class FastTreeBinaryModelParameters :
        TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "FastTreeBinaryExec";
        internal const string RegistrationName = "FastTreeBinaryPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FTREE BC",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, //Categorical splits.
                verReadableCur: 0x00010005,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FastTreeBinaryModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010002;

        private protected override uint VerDefaultValueSerialized => 0x00010004;

        private protected override uint VerCategoricalSplitSerialized => 0x00010005;

        internal FastTreeBinaryModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private FastTreeBinaryModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new FastTreeBinaryModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new SchemaBindableCalibratedModelParameters<FastTreeBinaryModelParameters, ICalibrator>(env, predictor, calibrator);
        }

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a decision tree binary classification model using FastTree.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="FastTree_remarks"]/*' />
    public sealed partial class FastTreeBinaryTrainer :
        BoostingFastTreeTrainerBase<FastTreeBinaryTrainer.Options,
        BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>,
        CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>
    {
        /// <summary>
        /// The LoadName for the assembly containing the trainer.
        /// </summary>
        internal const string LoadNameValue = "FastTreeBinaryClassification";
        internal const string UserNameValue = "FastTree (Boosted Trees) Classification";
        internal const string Summary = "Uses a logit-boost boosted tree learner to perform binary classification.";
        internal const string ShortName = "ftc";

        private bool[] _trainSetLabels;
        private double _sigmoidParameter;

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name for the column containing the example weight.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of documents allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="numberOfLeaves">The max number of leaves in each regression tree.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        internal FastTreeBinaryTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate)
            : base(env, TrainerUtils.MakeBoolScalarLabel(labelColumnName), featureColumnName, exampleWeightColumnName, null, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate)
        {
            // Set the sigmoid parameter to the 2 * learning rate, for traditional FastTreeClassification loss
            _sigmoidParameter = 2.0 * FastTreeTrainerOptions.LearningRate;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeBinaryTrainer"/> by using the <see cref="Options"/> class.
        /// </summary>
        /// <param name="env">The instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        internal FastTreeBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            // Set the sigmoid parameter to the 2 * learning rate, for traditional FastTreeClassification loss
            _sigmoidParameter = 2.0 * FastTreeTrainerOptions.LearningRate;
        }

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator> TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;
            ValidData = context.ValidationSet;
            TestData = context.TestSet;

            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(trainData, nameof(trainData));
                trainData.CheckBinaryLabel();
                trainData.CheckFeatureFloatVector();
                trainData.CheckOptFloatWeight();
                FeatureCount = trainData.Schema.Feature.Value.Type.GetValueCount();
                ConvertData(trainData);
                TrainCore(ch);
            }

            // The FastTree binary classification boosting is naturally calibrated to
            // output probabilities when transformed using a scaled logistic function,
            // so transform the scores using that.

            var pred = new FastTreeBinaryModelParameters(Host, TrainedEnsemble, FeatureCount, InnerOptions);
            // FastTree's binary classification boosting framework's natural probabilistic interpretation
            // is explained in "From RankNet to LambdaRank to LambdaMART: An Overview" by Chris Burges.
            // The correctness of this scaling depends upon the gradient calculation in
            // BinaryClassificationObjectiveFunction.GetGradientInOneQuery being consistent with the
            // description in section 6 of the paper.
            var cali = new PlattCalibrator(Host, -1 * _sigmoidParameter, 0);
            return new FeatureWeightsCalibratedModelParameters<FastTreeBinaryModelParameters, PlattCalibrator>(Host, pred, cali);
        }

        private protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch)
        {
            return new ObjectiveImpl(
                TrainSet,
                _trainSetLabels,
                FastTreeTrainerOptions.LearningRate,
                FastTreeTrainerOptions.Shrinkage,
                _sigmoidParameter,
                FastTreeTrainerOptions.UnbalancedSets,
                FastTreeTrainerOptions.MaximumTreeOutput,
                FastTreeTrainerOptions.GetDerivativesSampleRate,
                FastTreeTrainerOptions.BestStepRankingRegressionTrees,
                FastTreeTrainerOptions.Seed,
                ParallelTraining);
        }

        private protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            if (FastTreeTrainerOptions.UseLineSearch)
            {
                var lossCalculator = new BinaryClassificationTest(optimizationAlgorithm.TrainingScores, _trainSetLabels, _sigmoidParameter);
                // REVIEW: we should makeloss indices an enum in BinaryClassificationTest
                optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, FastTreeTrainerOptions.UnbalancedSets ? 3 /*Unbalanced  sets  loss*/ : 1 /*normal loss*/, FastTreeTrainerOptions.MaximumNumberOfLineSearchSteps, FastTreeTrainerOptions.MinimumStepSize);
            }
            return optimizationAlgorithm;
        }

        private IEnumerable<bool> GetClassificationLabelsFromRatings(Dataset set)
        {
            // REVIEW: Historically FastTree has this test as >= 1. TLC however
            // generally uses > 0. Consider changing FastTree to be consistent.
            return set.Ratings.Select(x => x >= 1);
        }

        private protected override void PrepareLabels(IChannel ch)
        {
            _trainSetLabels = GetClassificationLabelsFromRatings(TrainSet).ToArray(TrainSet.NumDocs);
            //Here we set regression labels to what is in bin file if the values were not overriden with floats
        }

        private protected override Test ConstructTestForTrainingData()
        {
            return new BinaryClassificationTest(ConstructScoreTracker(TrainSet), _trainSetLabels, _sigmoidParameter);
        }

        private protected override void InitializeTests()
        {
            //Always compute training L1/L2 errors
            TrainTest = new BinaryClassificationTest(ConstructScoreTracker(TrainSet), _trainSetLabels, _sigmoidParameter);
            Tests.Add(TrainTest);

            if (ValidSet != null)
            {
                ValidTest = new BinaryClassificationTest(ConstructScoreTracker(ValidSet),
                    GetClassificationLabelsFromRatings(ValidSet).ToArray(), _sigmoidParameter);
                Tests.Add(ValidTest);
            }

            //If external label is missing use Rating column for L1/L2 error
            //The values may not make much sense if regression value is not an actual label value
            if (TestSets != null)
            {
                for (int t = 0; t < TestSets.Length; ++t)
                {
                    bool[] labels = GetClassificationLabelsFromRatings(TestSets[t]).ToArray();
                    Tests.Add(new BinaryClassificationTest(ConstructScoreTracker(TestSets[t]), labels, _sigmoidParameter));
                }
            }

            if (FastTreeTrainerOptions.EnablePruning && ValidSet != null)
            {
                if (!FastTreeTrainerOptions.UseTolerantPruning)
                {
                    //use simple early stopping condition
                    PruningTest = new TestHistory(ValidTest, 0);
                }
                else
                {
                    //use tollerant stopping condition
                    PruningTest = new TestWindowWithTolerance(ValidTest, 0, FastTreeTrainerOptions.PruningWindowSize, FastTreeTrainerOptions.PruningThreshold);
                }
            }
        }

        private protected override BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>> MakeTransformer(
            CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator> model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="FastTreeBinaryTrainer"/> using both training and validation data, returns
        /// a <see cref="BinaryPredictionTransformer{CalibratedModelParametersBase}"/>.
        /// </summary>
        public BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        internal sealed class ObjectiveImpl : ObjectiveFunctionBase, IStepSearch
        {
            private readonly bool[] _labels;
            private readonly bool _unbalancedSets; //Should we use balanced or unbalanced loss function
            private readonly long _npos;
            private readonly long _nneg;
            private IParallelTraining _parallelTraining;
            private readonly double _sigmoidParameter; // Parameter for scaling the loss

            public ObjectiveImpl(
                Dataset trainSet,
                bool[] trainSetLabels,
                double learningRate,
                double shrinkage,
                double sigmoidParameter,
                bool unbalancedSets,
                double maxTreeOutput,
                int getDerivativesSampleRate,
                bool bestStepRankingRegressionTrees,
                int rngSeed,
                IParallelTraining parallelTraining)
                : base(
                    trainSet,
                    learningRate,
                    shrinkage,
                    maxTreeOutput,
                    getDerivativesSampleRate,
                    bestStepRankingRegressionTrees,
                    rngSeed)
            {
                _sigmoidParameter = sigmoidParameter;
                _labels = trainSetLabels;
                _unbalancedSets = unbalancedSets;
                if (_unbalancedSets)
                {
                    BinaryClassificationTest.ComputeExampleCounts(_labels, out _npos, out _nneg);
                    Contracts.Check(_nneg > 0 && _npos > 0, "Only one class in training set.");
                }
                _parallelTraining = parallelTraining;
            }

            protected override void GetGradientInOneQuery(int query, int threadIndex)
            {
                int begin = Dataset.Boundaries[query];
                int numDocuments = Dataset.Boundaries[query + 1] - Dataset.Boundaries[query];

                double recipNpos = 1.0;
                double recipNneg = 1.0;

                if (_unbalancedSets)
                {
                    recipNpos = 1.0 / _npos;
                    recipNneg = 1.0 / _nneg;
                }
                // See "From RankNet to LambdaRank to LambdaMART: An Overview" section 6 for a
                // description of these gradients.
                unsafe
                {
                    fixed (bool* pLabels = _labels)
                    fixed (double* pScores = Scores)
                    fixed (double* pLambdas = Gradient)
                    fixed (double* pWeights = Weights)
                    {
                        for (int i = begin; i < begin + numDocuments; ++i)
                        {
                            int label = pLabels[i] ? 1 : -1;
                            double recip = pLabels[i] ? recipNpos : recipNneg;
                            double response = label * _sigmoidParameter / (1.0 + Math.Exp(label * _sigmoidParameter * pScores[i]));
                            double absResponse = Math.Abs(response);
                            pLambdas[i] = response * recip;
                            pWeights[i] = absResponse * (_sigmoidParameter - absResponse) * recip;
                        }
                    }
                }
            }

            public void AdjustTreeOutputs(IChannel ch, InternalRegressionTree tree,
                DocumentPartitioning partitioning, ScoreTracker trainingScores)
            {
                const double epsilon = 1.4e-45;
                double multiplier = LearningRate * Shrinkage;
                double[] means = null;
                if (!BestStepRankingRegressionTrees)
                    means = _parallelTraining.GlobalMean(Dataset, tree, partitioning, Weights, false);
                for (int l = 0; l < tree.NumLeaves; ++l)
                {
                    double output = tree.GetOutput(l);

                    if (BestStepRankingRegressionTrees)
                        output *= multiplier;
                    else
                        output = multiplier * (output + epsilon) / (means[l] + epsilon);

                    if (output > MaxTreeOutput)
                        output = MaxTreeOutput;
                    else if (output < -MaxTreeOutput)
                        output = -MaxTreeOutput;
                    tree.SetOutput(l, output);
                }
            }
        }
    }

    /// <summary>
    /// The Entry Point for the FastTree Binary Classifier.
    /// </summary>
    internal static partial class FastTree
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastTreeBinaryClassifier",
            Desc = FastTreeBinaryTrainer.Summary,
            UserName = FastTreeBinaryTrainer.UserNameValue,
            ShortName = FastTreeBinaryTrainer.ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, FastTreeBinaryTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainFastTree");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<FastTreeBinaryTrainer.Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new FastTreeBinaryTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.RowGroupColumnName));
        }
    }
}
