// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    using LROptions = LogisticRegressionBinaryTrainer.Options;

    /// <summary>
    /// TrainerEstimator extension methods.
    /// </summary>
    public static class StandardTrainersCatalog
    {
        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SgdCalibratedTrainer"/>.
        /// Stochastic gradient descent (SGD) is an iterative algorithm that optimizes a differentiable objective function.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column, or dependent variable.</param>
        /// <param name="featureColumnName">The features, or independent variables.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfIterations">The maximum number of passes through the training dataset; set to 1 to simulate online learning.</param>
        /// <param name="initialLearningRate">The initial <a href="tmpurl_lr">learning rate</a> used by SGD.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='tmpurl_regularization'>regularization</a>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[StochasticGradientDescent](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/StochasticGradientDescent.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SgdCalibratedTrainer SgdCalibrated(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = SgdCalibratedTrainer.Options.Defaults.NumberOfIterations,
            double initialLearningRate = SgdCalibratedTrainer.Options.Defaults.InitialLearningRate,
            float l2Regularization = SgdCalibratedTrainer.Options.Defaults.L2Regularization)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SgdCalibratedTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName,
                numberOfIterations, initialLearningRate, l2Regularization);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SgdCalibratedTrainer"/> and advanced options.
        /// Stochastic gradient descent (SGD) is an iterative algorithm that optimizes a differentiable objective function.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[StochasticGradientDescentWithOptions](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/StochasticGradientDescentWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SgdCalibratedTrainer SgdCalibrated(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            SgdCalibratedTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SgdCalibratedTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SgdNonCalibratedTrainer"/>.
        /// Stochastic gradient descent (SGD) is an iterative algorithm that optimizes a differentiable objective function.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column, or dependent variable.</param>
        /// <param name="featureColumnName">The features, or independent variables.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="loss">The loss function minimized in the training process. Using, for example, <see cref="HingeLoss"/> leads to a support vector machine trainer.</param>
        /// <param name="numberOfIterations">The maximum number of passes through the training dataset; set to 1 to simulate online learning.</param>
        /// <param name="initialLearningRate">The initial <a href="tmpurl_lr">learning rate</a> used by SGD.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='tmpurl_regularization'>regularization</a>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[StochasticGradientDescentNonCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/StochasticGradientDescentNonCalibrated.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SgdNonCalibratedTrainer SgdNonCalibrated(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            IClassificationLoss loss = null,
            int numberOfIterations = SgdNonCalibratedTrainer.Options.Defaults.NumberOfIterations,
            double initialLearningRate = SgdNonCalibratedTrainer.Options.Defaults.InitialLearningRate,
            float l2Regularization = SgdNonCalibratedTrainer.Options.Defaults.L2Regularization)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SgdNonCalibratedTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName,
                numberOfIterations, initialLearningRate, l2Regularization, loss);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SgdNonCalibratedTrainer"/> and advanced options.
        /// Stochastic gradient descent (SGD) is an iterative algorithm that optimizes a differentiable objective function.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[StochasticGradientDescentNonCalibratedWithOptions](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/StochasticGradientDescentNonCalibratedWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SgdNonCalibratedTrainer SgdNonCalibrated(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            SgdNonCalibratedTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SgdNonCalibratedTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with <see cref="SdcaRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="l2Regularization">The L2 <a href='tmpurl_regularization'>regularization</a> hyperparameter.</param>
        /// <param name="l1Threshold">The L1 <a href='tmpurl_regularization'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="loss">The custom <a href="tmpurl_loss">loss</a>, if unspecified will be <see cref="SquaredLoss"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/StochasticDualCoordinateAscent.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaRegressionTrainer Sdca(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            ISupportSdcaRegressionLoss loss = null,
            float? l2Regularization = null,
            float? l1Threshold = null,
            int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, loss, l2Regularization, l1Threshold, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with <see cref="SdcaRegressionTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/StochasticDualCoordinateAscentWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaRegressionTrainer Sdca(this RegressionCatalog.RegressionTrainers catalog,
        SdcaRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SdcaCalibratedBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="l2Regularization">The L2 <a href='tmpurl_regularization'>regularization</a> hyperparameter.</param>
        /// <param name="l1Threshold">The L1 <a href='tmpurl_regularization'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/StochasticDualCoordinateAscent.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaCalibratedBinaryTrainer SdcaCalibrated(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                string labelColumnName = DefaultColumnNames.Label,
                string featureColumnName = DefaultColumnNames.Features,
                string exampleWeightColumnName = null,
                float? l2Regularization = null,
                float? l1Threshold = null,
                int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaCalibratedBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l2Regularization, l1Threshold, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SdcaCalibratedBinaryTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/StochasticDualCoordinateAscentWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaCalibratedBinaryTrainer SdcaCalibrated(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                SdcaCalibratedBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaCalibratedBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SdcaNonCalibratedBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="loss">The custom <a href="tmpurl_loss">loss</a>. Defaults to <see cref="LogLoss"/> if not specified.</param>
        /// <param name="l2Regularization">The L2 <a href='tmpurl_regularization'>regularization</a> hyperparameter.</param>
        /// <param name="l1Threshold">The L1 <a href='tmpurl_regularization'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/StochasticDualCoordinateAscentNonCalibrated.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaNonCalibratedBinaryTrainer SdcaNonCalibrated(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                string labelColumnName = DefaultColumnNames.Label,
                string featureColumnName = DefaultColumnNames.Features,
                string exampleWeightColumnName = null,
                ISupportSdcaClassificationLoss loss = null,
                float? l2Regularization = null,
                float? l1Threshold = null,
                int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaNonCalibratedBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, loss, l2Regularization, l1Threshold, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SdcaNonCalibratedBinaryTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        public static SdcaNonCalibratedBinaryTrainer SdcaNonCalibrated(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                SdcaNonCalibratedBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaNonCalibratedBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with <see cref="SdcaCalibratedMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="l2Regularization">The L2 <a href='tmpurl_regularization'>regularization</a> hyperparameter.</param>
        /// <param name="l1Threshold">The L1 <a href='tmpurl_regularization'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/StochasticDualCoordinateAscent.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaCalibratedMulticlassTrainer SdcaCalibrated(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
        string labelColumnName = DefaultColumnNames.Label,
                    string featureColumnName = DefaultColumnNames.Features,
                    string exampleWeightColumnName = null,
                    float? l2Regularization = null,
                    float? l1Threshold = null,
                    int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaCalibratedMulticlassTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l2Regularization, l1Threshold, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with <see cref="SdcaCalibratedMulticlassTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/StochasticDualCoordinateAscentWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaCalibratedMulticlassTrainer SdcaCalibrated(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
        SdcaCalibratedMulticlassTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaCalibratedMulticlassTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with <see cref="SdcaNonCalibratedMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="loss">Loss function to be minimized. Defaults to <see cref="LogLoss"/> if not specified.</param>
        /// <param name="l2Regularization">The L2 <a href='tmpurl_regularization'>regularization</a> hyperparameter.</param>
        /// <param name="l1Threshold">The L1 <a href='tmpurl_regularization'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/StochasticDualCoordinateAscent.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaNonCalibratedMulticlassTrainer SdcaNonCalibrated(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
        string labelColumnName = DefaultColumnNames.Label,
                    string featureColumnName = DefaultColumnNames.Features,
                    string exampleWeightColumnName = null,
                    ISupportSdcaClassificationLoss loss = null,
                    float? l2Regularization = null,
                    float? l1Threshold = null,
                    int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaNonCalibratedMulticlassTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, loss, l2Regularization, l1Threshold, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using linear multiclass classification model trained with <see cref="SdcaNonCalibratedMulticlassTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/StochasticDualCoordinateAscentWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaNonCalibratedMulticlassTrainer SdcaNonCalibrated(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
        SdcaNonCalibratedMulticlassTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaNonCalibratedMulticlassTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with <see cref="AveragedPerceptronTrainer"/>.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="lossFunction">A custom <a href="tmpurl_loss">loss</a>. If <see langword="null"/>, hinge loss will be used resulting in max-margin averaged perceptron.</param>
        /// <param name="learningRate"><a href="tmpurl_lr">Learning rate</a>.</param>
        /// <param name="decreaseLearningRate">
        /// <see langword="true" /> to decrease the <paramref name="learningRate"/> as iterations progress; otherwise, <see langword="false" />.
        /// Default is <see langword="false" />.
        /// </param>
        /// <param name="l2Regularization">The L2 weight for <a href='tmpurl_regularization'>regularization</a>.</param>
        /// <param name="numberOfIterations">Number of passes through the training dataset.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[AveragedPerceptron](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/AveragedPerceptron.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static AveragedPerceptronTrainer AveragedPerceptron(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            IClassificationLoss lossFunction = null,
            float learningRate = AveragedLinearOptions.AveragedDefault.LearningRate,
            bool decreaseLearningRate = AveragedLinearOptions.AveragedDefault.DecreaseLearningRate,
            float l2Regularization = AveragedLinearOptions.AveragedDefault.L2Regularization,
            int numberOfIterations = AveragedLinearOptions.AveragedDefault.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new AveragedPerceptronTrainer(env, labelColumnName, featureColumnName, lossFunction ?? new LogLoss(), learningRate, decreaseLearningRate, l2Regularization, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with <see cref="AveragedPerceptronTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[AveragedPerceptron](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/AveragedPerceptronWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="lossFunction">The custom loss. Defaults to <see cref="SquaredLoss"/> if not provided.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='tmpurl_regularization'>regularization</a>.</param>
        /// <param name="numberOfIterations">Number of training iterations through the data.</param>
        public static OnlineGradientDescentTrainer OnlineGradientDescent(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            IRegressionLoss lossFunction = null,
            float learningRate = OnlineGradientDescentTrainer.Options.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = OnlineGradientDescentTrainer.Options.OgdDefaultArgs.DecreaseLearningRate,
            float l2Regularization = AveragedLinearOptions.AveragedDefault.L2Regularization,
            int numberOfIterations = OnlineLinearOptions.OnlineDefault.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new OnlineGradientDescentTrainer(env, labelColumnName, featureColumnName, learningRate, decreaseLearningRate, l2Regularization,
                numberOfIterations, lossFunction);
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
        ///  Predict a target using a linear binary classification model trained with the <see cref="Trainers.LogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Regularization">Weight of L1 regularization term.</param>
        /// <param name="l2Regularization">Weight of L2 regularization term.</param>
        /// <param name="historySize">Memory size for <see cref="Trainers.LogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Logistic Regression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/LogisticRegression.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LogisticRegressionBinaryTrainer LogisticRegression(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            float l1Regularization = LROptions.Defaults.L1Regularization,
            float l2Regularization = LROptions.Defaults.L2Regularization,
            float optimizationTolerance = LROptions.Defaults.OptimizationTolerance,
            int historySize = LROptions.Defaults.HistorySize,
            bool enforceNonNegativity = LROptions.Defaults.EnforceNonNegativity)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LogisticRegressionBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Trainers.LogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static LogisticRegressionBinaryTrainer LogisticRegression(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog, LROptions options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new LogisticRegressionBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.PoissonRegressionTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="l1Regularization">Weight of L1 regularization term.</param>
        /// <param name="l2Regularization">Weight of L2 regularization term.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="historySize">Memory size for <see cref="Microsoft.ML.Trainers.PoissonRegressionTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        public static PoissonRegressionTrainer PoissonRegression(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            float l1Regularization = LROptions.Defaults.L1Regularization,
            float l2Regularization = LROptions.Defaults.L2Regularization,
            float optimizationTolerance = LROptions.Defaults.OptimizationTolerance,
            int historySize = LROptions.Defaults.HistorySize,
            bool enforceNonNegativity = LROptions.Defaults.EnforceNonNegativity)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new PoissonRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.PoissonRegressionTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static PoissonRegressionTrainer PoissonRegression(this RegressionCatalog.RegressionTrainers catalog, PoissonRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new PoissonRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the L-BFGS method implemented in <see cref="LbfgsMaximumEntropyTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Regularization">Weight of L1 regularization term.</param>
        /// <param name="l2Regularization">Weight of L2 regularization term.</param>
        /// <param name="historySize">Memory size for <see cref="Microsoft.ML.Trainers.LbfgsMaximumEntropyTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        public static LbfgsMaximumEntropyTrainer LbfgsMaximumEntropy(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            float l1Regularization = LROptions.Defaults.L1Regularization,
            float l2Regularization = LROptions.Defaults.L2Regularization,
            float optimizationTolerance = LROptions.Defaults.OptimizationTolerance,
            int historySize = LROptions.Defaults.HistorySize,
            bool enforceNonNegativity = LROptions.Defaults.EnforceNonNegativity)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LbfgsMaximumEntropyTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the L-BFGS method implemented in <see cref="LbfgsMaximumEntropyTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        public static LbfgsMaximumEntropyTrainer LbfgsMaximumEntropy(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            LbfgsMaximumEntropyTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new LbfgsMaximumEntropyTrainer(env, options);
        }

        /// <summary>
        /// Predicts a target using a linear multiclass classification model trained with the <see cref="NaiveBayesMulticlassTrainer"/>.
        /// The <see cref="NaiveBayesMulticlassTrainer"/> trains a multiclass Naive Bayes predictor that supports binary feature values.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        public static NaiveBayesMulticlassTrainer NaiveBayes(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            return new NaiveBayesMulticlassTrainer(CatalogUtils.GetEnvironment(catalog), labelColumnName, featureColumnName);
        }

        /// <summary>
        /// Works via the <see cref="IHaveCalibratorTrainer"/> shim interface to extract from the calibrating training
        /// estimator the internal <see cref="ICalibratorTrainer"/> object. Note that this should be a temporary measure,
        /// since the trainers should really be changed to actually work over estimators.
        /// </summary>
        /// <param name="ectx">The exception context.</param>
        /// <param name="calibratorEstimator">The estimator out of which we should try to extract the calibrator trainer.</param>
        /// <returns>The calibrator trainer.</returns>
        private static ICalibratorTrainer GetCalibratorTrainerOrThrow(IExceptionContext ectx, IEstimator<ISingleFeaturePredictionTransformer<ICalibrator>> calibratorEstimator)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValueOrNull(calibratorEstimator);
            if (calibratorEstimator == null)
                return null;
            if (calibratorEstimator is IHaveCalibratorTrainer haveCalibratorTrainer)
                return haveCalibratorTrainer.CalibratorTrainer;
            throw ectx.ExceptParam(nameof(calibratorEstimator),
                "Calibrator estimator was not of a type usable in this context.");
        }

        /// <summary>
        /// Predicts a target using a linear multiclass classification model trained with the <see cref="OneVersusAllTrainer"/>.
        /// </summary>
        /// <remarks>
        /// <para>
        /// In <see cref="OneVersusAllTrainer"/> In this strategy, a binary classification algorithm is used to train one classifier for each class,
        /// which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers,
        /// and choosing the prediction with the highest confidence score.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="binaryEstimator">An instance of a binary <see cref="ITrainerEstimator{TTransformer, TPredictor}"/> used as the base trainer.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitely provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        /// <param name="labelColumnName">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="maxCalibrationExamples">Number of instances to train the calibrator.</param>
        /// <param name="useProbabilities">Use probabilities (vs. raw outputs) to identify top-score category.</param>
        /// <typeparam name="TModel">The type of the model. This type parameter will usually be inferred automatically from <paramref name="binaryEstimator"/>.</typeparam>
        public static OneVersusAllTrainer OneVersusAll<TModel>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            ITrainerEstimator<BinaryPredictionTransformer<TModel>, TModel> binaryEstimator,
            string labelColumnName = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            IEstimator<ISingleFeaturePredictionTransformer<ICalibrator>> calibrator = null,
            int maxCalibrationExamples = 1000000000,
            bool useProbabilities = true)
            where TModel : class
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            if (!(binaryEstimator is ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>> est))
                throw env.ExceptParam(nameof(binaryEstimator), "Trainer estimator does not appear to produce the right kind of model.");
            return new OneVersusAllTrainer(env, est, labelColumnName, imputeMissingLabelsAsNegative, GetCalibratorTrainerOrThrow(env, calibrator), maxCalibrationExamples, useProbabilities);
        }

        /// <summary>
        /// Predicts a target using a linear multiclass classification model trained with the <see cref="PairwiseCouplingTrainer"/>.
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
        /// <param name="labelColumnName">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="maximumCalibrationExampleCount">Number of instances to train the calibrator.</param>
        /// <typeparam name="TModel">The type of the model. This type parameter will usually be inferred automatically from <paramref name="binaryEstimator"/>.</typeparam>
        public static PairwiseCouplingTrainer PairwiseCoupling<TModel>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            ITrainerEstimator<ISingleFeaturePredictionTransformer<TModel>, TModel> binaryEstimator,
            string labelColumnName = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            IEstimator<ISingleFeaturePredictionTransformer<ICalibrator>> calibrator = null,
            int maximumCalibrationExampleCount = 1_000_000_000)
            where TModel : class
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            if (!(binaryEstimator is ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>> est))
                throw env.ExceptParam(nameof(binaryEstimator), "Trainer estimator does not appear to produce the right kind of model.");
            return new PairwiseCouplingTrainer(env, est, labelColumnName, imputeMissingLabelsAsNegative,
                                                GetCalibratorTrainerOrThrow(env, calibrator), maximumCalibrationExampleCount);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the <see cref="LinearSvmTrainer"/> trainer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The idea behind support vector machines (SVM), is to map instances into a high dimensional space
        /// in which the two classes are linearly separable, i.e., there exists a hyperplane such that all the positive examples are on one side of it,
        /// and all the negative examples are on the other.
        /// </para>
        /// <para>
        /// After this mapping, quadratic programming is used to find the separating hyperplane that maximizes the
        /// margin, i.e., the minimal distance between it and the instances.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. </param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfIterations">The number of training iteraitons.</param>
        public static LinearSvmTrainer LinearSvm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = OnlineLinearOptions.OnlineDefault.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            return new LinearSvmTrainer(CatalogUtils.GetEnvironment(catalog), labelColumnName, featureColumnName, exampleWeightColumnName, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the <see cref="LinearSvmTrainer"/> trainer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The idea behind support vector machines (SVM), is to map instances into a high dimensional space
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
        public static LinearSvmTrainer LinearSvm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            LinearSvmTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            return new LinearSvmTrainer(CatalogUtils.GetEnvironment(catalog), options);
        }

        /// <summary>
        /// Predict a target using a binary classification model trained with <see cref="PriorTrainer"/> trainer.
        /// </summary>
        /// <remarks>
        /// This trainer uses the proportion of a label in the training set as the probability of that label.
        /// This trainer is often used as a baseline for other more sophisticated mdels.
        /// </remarks>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. </param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/PriorTrainerSample.cs)]
        /// ]]></format>
        /// </example>
        public static PriorTrainer Prior(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string exampleWeightColumnName = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            return new PriorTrainer(CatalogUtils.GetEnvironment(catalog), labelColumnName, exampleWeightColumnName);
        }
    }
}
