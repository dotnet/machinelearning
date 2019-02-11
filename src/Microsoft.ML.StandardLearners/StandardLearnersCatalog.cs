// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Training;

namespace Microsoft.ML
{
    using LROptions = LogisticRegression.Options;
    using SgdOptions = StochasticGradientDescentClassificationTrainer.Options;
    using TLegacyPredictor = IPredictorProducing<float>;

    /// <summary>
    /// TrainerEstimator extension methods.
    /// </summary>
    public static class StandardLearnersCatalog
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="StochasticGradientDescentClassificationTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="maxIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Weight">The L2 regularization constant.</param>
        /// <param name="loss">The loss function to use.</param>
        public static StochasticGradientDescentClassificationTrainer StochasticGradientDescent(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int maxIterations = SgdOptions.Defaults.MaxIterations,
            double initLearningRate = SgdOptions.Defaults.InitLearningRate,
            float l2Weight = SgdOptions.Defaults.L2Weight,
            ISupportClassificationLossFactory loss = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new StochasticGradientDescentClassificationTrainer(env, labelColumn, featureColumn, weights, maxIterations, initLearningRate, l2Weight, loss);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="StochasticGradientDescentClassificationTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static StochasticGradientDescentClassificationTrainer StochasticGradientDescent(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            SgdOptions options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new StochasticGradientDescentClassificationTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="labelColumn">The label column, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="loss">The custom loss, if unspecified will be <see cref="SquaredLoss"/>.</param>

        public static SdcaRegressionTrainer StochasticDualCoordinateAscent(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            ISupportSdcaRegressionLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaRegressionTrainer(env, labelColumn, featureColumn, weights, loss, l2Const, l1Threshold, maxIterations);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static SdcaRegressionTrainer StochasticDualCoordinateAscent(this RegressionCatalog.RegressionTrainers catalog,
            SdcaRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="loss">The custom loss. Defaults to log-loss if not specified.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/SDCA.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaBinaryTrainer StochasticDualCoordinateAscent(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                string labelColumn = DefaultColumnNames.Label,
                string featureColumn = DefaultColumnNames.Features,
                string weights = null,
                ISupportSdcaClassificationLoss loss = null,
                float? l2Const = null,
                float? l1Threshold = null,
                int? maxIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaBinaryTrainer(env, labelColumn, featureColumn, weights, loss, l2Const, l1Threshold, maxIterations);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static SdcaBinaryTrainer StochasticDualCoordinateAscent(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                SdcaBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="loss">The optional custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        public static SdcaMultiClassTrainer StochasticDualCoordinateAscent(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
                    string labelColumn = DefaultColumnNames.Label,
                    string featureColumn = DefaultColumnNames.Features,
                    string weights = null,
                    ISupportSdcaClassificationLoss loss = null,
                    float? l2Const = null,
                    float? l1Threshold = null,
                    int? maxIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaMultiClassTrainer(env, labelColumn, featureColumn, weights, loss, l2Const, l1Threshold, maxIterations);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static SdcaMultiClassTrainer StochasticDualCoordinateAscent(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
                    SdcaMultiClassTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaMultiClassTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the AveragedPerceptron trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumn">The name of the label column, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="lossFunction">The custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 regularization weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        public static AveragedPerceptronTrainer AveragedPerceptron(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            IClassificationLoss lossFunction = null,
            float learningRate = AveragedLinearArguments.AveragedDefaultArgs.LearningRate,
            bool decreaseLearningRate = AveragedLinearArguments.AveragedDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
            int numIterations = AveragedLinearArguments.AveragedDefaultArgs.NumIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new AveragedPerceptronTrainer(env, labelColumn, featureColumn, weights, lossFunction ?? new LogLoss(), learningRate, decreaseLearningRate, l2RegularizerWeight, numIterations);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the AveragedPerceptron trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static AveragedPerceptronTrainer AveragedPerceptron(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog, AveragedPerceptronTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new AveragedPerceptronTrainer(env, options);
        }

        private sealed class TrivialClassificationLossFactory : ISupportClassificationLossFactory
        {
            private readonly IClassificationLoss _loss;

            public TrivialClassificationLossFactory(IClassificationLoss loss)
            {
                _loss = loss;
            }

            public IClassificationLoss CreateComponent(IHostEnvironment env)
            {
                return _loss;
            }
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OnlineGradientDescentTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="labelColumn">The name of the label, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="lossFunction">The custom loss. Defaults to <see cref="SquaredLoss"/> if not provided.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 regularization weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        public static OnlineGradientDescentTrainer OnlineGradientDescent(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            IRegressionLoss lossFunction = null,
            float learningRate = OnlineGradientDescentTrainer.Options.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = OnlineGradientDescentTrainer.Options.OgdDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
            int numIterations = OnlineLinearArguments.OnlineDefaultArgs.NumIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new OnlineGradientDescentTrainer(env, labelColumn, featureColumn, learningRate, decreaseLearningRate, l2RegularizerWeight,
                numIterations, weights, lossFunction);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OnlineGradientDescentTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static OnlineGradientDescentTrainer OnlineGradientDescent(this RegressionCatalog.RegressionTrainers catalog,
            OnlineGradientDescentTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new OnlineGradientDescentTrainer(env, options);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Trainers.LogisticRegression"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="labelColumn">The label column name, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="memorySize">Memory size for <see cref="Trainers.LogisticRegression"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Logistic Regression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/LogisticRegression.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LogisticRegression LogisticRegression(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = LROptions.Defaults.L1Weight,
            float l2Weight = LROptions.Defaults.L2Weight,
            float optimizationTolerance = LROptions.Defaults.OptTol,
            int memorySize = LROptions.Defaults.MemorySize,
            bool enforceNoNegativity = LROptions.Defaults.EnforceNonNegativity)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LogisticRegression(env, labelColumn, featureColumn, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Trainers.LogisticRegression"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static LogisticRegression LogisticRegression(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog, LROptions options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new LogisticRegression(env, options);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.PoissonRegression"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Trainers.PoissonRegression"/>. Low=faster, less accurate.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        public static PoissonRegression PoissonRegression(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = LROptions.Defaults.L1Weight,
            float l2Weight = LROptions.Defaults.L2Weight,
            float optimizationTolerance = LROptions.Defaults.OptTol,
            int memorySize = LROptions.Defaults.MemorySize,
            bool enforceNoNegativity = LROptions.Defaults.EnforceNonNegativity)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new PoissonRegression(env, labelColumn, featureColumn, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.PoissonRegression"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static PoissonRegression PoissonRegression(this RegressionCatalog.RegressionTrainers catalog, PoissonRegression.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new PoissonRegression(env, options);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the <see cref="MulticlassLogisticRegression"/> trainer.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Trainers.MulticlassLogisticRegression"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        public static MulticlassLogisticRegression LogisticRegression(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = LROptions.Defaults.L1Weight,
            float l2Weight = LROptions.Defaults.L2Weight,
            float optimizationTolerance = LROptions.Defaults.OptTol,
            int memorySize = LROptions.Defaults.MemorySize,
            bool enforceNoNegativity = LROptions.Defaults.EnforceNonNegativity)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new MulticlassLogisticRegression(env, labelColumn, featureColumn, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the <see cref="MulticlassLogisticRegression"/> trainer.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static MulticlassLogisticRegression LogisticRegression(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            MulticlassLogisticRegression.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new MulticlassLogisticRegression(env, options);
        }

        /// <summary>
        /// Predicts a target using a linear multiclass classification model trained with the <see cref="MultiClassNaiveBayesTrainer"/>.
        /// The <see cref="MultiClassNaiveBayesTrainer"/> trains a multiclass Naive Bayes predictor that supports binary feature values.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        public static MultiClassNaiveBayesTrainer NaiveBayes(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            return new MultiClassNaiveBayesTrainer(CatalogUtils.GetEnvironment(catalog), labelColumn, featureColumn);
        }

        /// <summary>
        /// Predicts a target using a linear multiclass classification model trained with the <see cref="Ova"/>.
        /// </summary>
        /// <remarks>
        /// <para>
        /// In <see cref="Ova"/> In this strategy, a binary classification algorithm is used to train one classifier for each class,
        /// which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers,
        /// and choosing the prediction with the highest confidence score.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="binaryEstimator">An instance of a binary <see cref="ITrainerEstimator{TTransformer, TPredictor}"/> used as the base trainer.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitely provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        /// <param name="labelColumn">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="maxCalibrationExamples">Number of instances to train the calibrator.</param>
        /// <param name="useProbabilities">Use probabilities (vs. raw outputs) to identify top-score category.</param>
        /// <typeparam name="TModel">The type of the model. This type parameter will usually be inferred automatically from <paramref name="binaryEstimator"/>.</typeparam>
        public static Ova OneVersusAll<TModel>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            ITrainerEstimator<ISingleFeaturePredictionTransformer<TModel>, TModel> binaryEstimator,
            string labelColumn = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            ICalibratorTrainer calibrator = null,
            int maxCalibrationExamples = 1000000000,
            bool useProbabilities = true)
            where TModel : class
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            if (!(binaryEstimator is ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>> est))
                throw env.ExceptParam(nameof(binaryEstimator), "Trainer estimator does not appear to produce the right kind of model.");
            return new Ova(env, est, labelColumn, imputeMissingLabelsAsNegative, calibrator, maxCalibrationExamples, useProbabilities);
        }

        /// <summary>
        /// Predicts a target using a linear multiclass classification model trained with the <see cref="Pkpd"/>.
        /// </summary>
        /// <remarks>
        /// <para>
        /// In the Pairwise coupling (PKPD) strategy, a binary classification algorithm is used to train one classifier for each pair of classes.
        /// Prediction is then performed by running these binary classifiers, and computing a score for each class by counting how many of the binary
        /// classifiers predicted it. The prediction is the class with the highest score.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="binaryEstimator">An instance of a binary <see cref="ITrainerEstimator{TTransformer, TPredictor}"/> used as the base trainer.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitely provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        /// <param name="labelColumn">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="maxCalibrationExamples">Number of instances to train the calibrator.</param>
        /// <typeparam name="TModel">The type of the model. This type parameter will usually be inferred automatically from <paramref name="binaryEstimator"/>.</typeparam>
        public static Pkpd PairwiseCoupling<TModel>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            ITrainerEstimator<ISingleFeaturePredictionTransformer<TModel>, TModel> binaryEstimator,
            string labelColumn = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            ICalibratorTrainer calibrator = null,
            int maxCalibrationExamples = 1_000_000_000)
            where TModel : class
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            if (!(binaryEstimator is ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>> est))
                throw env.ExceptParam(nameof(binaryEstimator), "Trainer estimator does not appear to produce the right kind of model.");
            return new Pkpd(env, est, labelColumn, imputeMissingLabelsAsNegative, calibrator, maxCalibrationExamples);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the <see cref="LinearSvmTrainer"/> trainer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The idea behind support vector machines, is to map instances into a high dimensional space
        /// in which the two classes are linearly separable, i.e., there exists a hyperplane such that all the positive examples are on one side of it,
        /// and all the negative examples are on the other.
        /// </para>
        /// <para>
        /// After this mapping, quadratic programming is used to find the separating hyperplane that maximizes the
        /// margin, i.e., the minimal distance between it and the instances.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumn">The name of the label column. </param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightsColumn">The optional name of the weights column.</param>
        /// <param name="numIterations">The number of training iteraitons.</param>
        public static LinearSvmTrainer LinearSupportVectorMachines(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightsColumn = null,
            int numIterations = OnlineLinearArguments.OnlineDefaultArgs.NumIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            return new LinearSvmTrainer(CatalogUtils.GetEnvironment(catalog), labelColumn, featureColumn, weightsColumn, numIterations);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the <see cref="LinearSvmTrainer"/> trainer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The idea behind support vector machines, is to map instances into a high dimensional space
        /// in which the two classes are linearly separable, i.e., there exists a hyperplane such that all the positive examples are on one side of it,
        /// and all the negative examples are on the other.
        /// </para>
        /// <para>
        /// After this mapping, quadratic programming is used to find the separating hyperplane that maximizes the
        /// margin, i.e., the minimal distance between it and the instances.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static LinearSvmTrainer LinearSupportVectorMachines(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            LinearSvmTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            return new LinearSvmTrainer(CatalogUtils.GetEnvironment(catalog), options);
        }
    }
}
