// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    using LROptions = LbfgsLogisticRegressionBinaryTrainer.Options;

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
        /// <param name="learningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SgdCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SgdCalibrated.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SgdCalibratedTrainer SgdCalibrated(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = SgdCalibratedTrainer.Options.Defaults.NumberOfIterations,
            double learningRate = SgdCalibratedTrainer.Options.Defaults.LearningRate,
            float l2Regularization = SgdCalibratedTrainer.Options.Defaults.L2Regularization)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SgdCalibratedTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName,
                numberOfIterations, learningRate, l2Regularization);
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
        /// [!code-csharp[SgdCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SgdCalibratedWithOptions.cs)]
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
        /// <param name="lossFunction">The <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a> function minimized in the training process. Using, for example, <see cref="HingeLoss"/> leads to a support vector machine trainer.</param>
        /// <param name="numberOfIterations">The maximum number of passes through the training dataset; set to 1 to simulate online learning.</param>
        /// <param name="learningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SgdNonCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SgdNonCalibrated.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SgdNonCalibratedTrainer SgdNonCalibrated(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            IClassificationLoss lossFunction = null,
            int numberOfIterations = SgdNonCalibratedTrainer.Options.Defaults.NumberOfIterations,
            double learningRate = SgdNonCalibratedTrainer.Options.Defaults.LearningRate,
            float l2Regularization = SgdNonCalibratedTrainer.Options.Defaults.L2Regularization)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SgdNonCalibratedTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName,
                numberOfIterations, learningRate, l2Regularization, lossFunction);
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
        /// [!code-csharp[SgdNonCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SgdNonCalibratedWithOptions.cs)]
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
        /// <param name="lossFunction">The <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a> function minimized in the training process. Using, for example, its default <see cref="SquaredLoss"/> leads to a least square trainer.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/Sdca.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaRegressionTrainer Sdca(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            ISupportSdcaRegressionLoss lossFunction = null,
            float? l2Regularization = null,
            float? l1Regularization = null,
            int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, lossFunction, l2Regularization, l1Regularization, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with <see cref="SdcaRegressionTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/SdcaWithOptions.cs)]
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
        /// Predict a target using a linear classification model trained with <see cref="SdcaLogisticRegressionBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SdcaLogisticRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SdcaLogisticRegression.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaLogisticRegressionBinaryTrainer SdcaLogisticRegression(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                string labelColumnName = DefaultColumnNames.Label,
                string featureColumnName = DefaultColumnNames.Features,
                string exampleWeightColumnName = null,
                float? l2Regularization = null,
                float? l1Regularization = null,
                int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaLogisticRegressionBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l2Regularization, l1Regularization, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SdcaLogisticRegressionBinaryTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SdcaLogisticRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SdcaLogisticRegressionWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaLogisticRegressionBinaryTrainer SdcaLogisticRegression(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                SdcaLogisticRegressionBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaLogisticRegressionBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SdcaNonCalibratedBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="lossFunction">The <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a> function minimized in the training process. Defaults to <see cref="LogLoss"/> if not specified.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SdcaNonCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SdcaNonCalibrated.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaNonCalibratedBinaryTrainer SdcaNonCalibrated(
                this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
                string labelColumnName = DefaultColumnNames.Label,
                string featureColumnName = DefaultColumnNames.Features,
                string exampleWeightColumnName = null,
                ISupportSdcaClassificationLoss lossFunction = null,
                float? l2Regularization = null,
                float? l1Regularization = null,
                int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaNonCalibratedBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, lossFunction, l2Regularization, l1Regularization, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a linear classification model trained with <see cref="SdcaNonCalibratedBinaryTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SdcaNonCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/SdcaNonCalibratedWithOptions.cs)]
        /// ]]></format>
        /// </example>
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
        /// Predict a target using a maximum entropy classification model trained with <see cref="SdcaMaximumEntropyMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/SdcaMaximumEntropy.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaMaximumEntropyMulticlassTrainer SdcaMaximumEntropy(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
        string labelColumnName = DefaultColumnNames.Label,
                    string featureColumnName = DefaultColumnNames.Features,
                    string exampleWeightColumnName = null,
                    float? l2Regularization = null,
                    float? l1Regularization = null,
                    int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaMaximumEntropyMulticlassTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l2Regularization, l1Regularization, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with <see cref="SdcaMaximumEntropyMulticlassTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/SdcaMaximumEntropyWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaMaximumEntropyMulticlassTrainer SdcaMaximumEntropy(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
        SdcaMaximumEntropyMulticlassTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaMaximumEntropyMulticlassTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with <see cref="SdcaNonCalibratedMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="lossFunction">The <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a> function to be minimized. Defaults to <see cref="LogLoss"/> if not specified.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maximumNumberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/SdcaNonCalibrated.cs)]
        /// ]]></format>
        /// </example>
        public static SdcaNonCalibratedMulticlassTrainer SdcaNonCalibrated(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
        string labelColumnName = DefaultColumnNames.Label,
                    string featureColumnName = DefaultColumnNames.Features,
                    string exampleWeightColumnName = null,
                    ISupportSdcaClassificationLoss lossFunction = null,
                    float? l2Regularization = null,
                    float? l1Regularization = null,
                    int? maximumNumberOfIterations = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new SdcaNonCalibratedMulticlassTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, lossFunction, l2Regularization, l1Regularization, maximumNumberOfIterations);
        }

        /// <summary>
        /// Predict a target using linear multiclass classification model trained with <see cref="SdcaNonCalibratedMulticlassTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/SdcaNonCalibratedWithOptions.cs)]
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
        /// Create an <see cref="AveragedPerceptronTrainer"/>, which predicts a target using a linear binary classification model trained over boolean label data.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Boolean"/>.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="lossFunction">The <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a> function minimized in the training process. If <see langword="null"/>, <see cref="HingeLoss"/> would be used and lead to a max-margin averaged perceptron trainer.</param>
        /// <param name="learningRate">The initial learning rate used by SGD.</param>
        /// <param name="decreaseLearningRate">
        /// <see langword="true" /> to decrease the <paramref name="learningRate"/> as iterations progress; otherwise, <see langword="false" />.
        /// Default is <see langword="false" />.
        /// </param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
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
        /// Create an <see cref="AveragedPerceptronTrainer"/> with advanced options, which predicts a target using a linear binary classification model trained over boolean label data.
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
        /// Predict a target using a linear regression model trained with the <see cref="OnlineGradientDescentTrainer"/>.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="lossFunction">The <a href="https://en.wikipedia.org/wiki/Loss_function">loss</a> function minimized in the training process. Using, for example, <see cref="SquaredLoss"/> leads to a least square trainer.</param>
        /// <param name="learningRate">The initial learning rate used by SGD.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="numberOfIterations">The number of passes through the training dataset.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[OnlineGradientDescent](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/OnlineGradientDescent.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
        /// Predict a target using a linear regression model trained with the <see cref="OnlineGradientDescentTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[OnlineGradientDescent](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/OnlineGradientDescentWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OnlineGradientDescentTrainer OnlineGradientDescent(this RegressionCatalog.RegressionTrainers catalog,
            OnlineGradientDescentTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new OnlineGradientDescentTrainer(env, options);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Trainers.LbfgsLogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="historySize">Memory size for <see cref="Trainers.LbfgsLogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Logistic Regression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LbfgsLogisticRegression.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LbfgsLogisticRegressionBinaryTrainer LbfgsLogisticRegression(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
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
            return new LbfgsLogisticRegressionBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Trainers.LbfgsLogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Logistic Regression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LbfgsLogisticRegressionWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LbfgsLogisticRegressionBinaryTrainer LbfgsLogisticRegression(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog, LROptions options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new LbfgsLogisticRegressionBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.LbfgsPoissonRegressionTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="historySize">Number of previous iterations to remember for estimating the Hessian. Lower values mean faster but less accurate estimates.</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[PoissonRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/LbfgsPoissonRegression.cs)]
        /// ]]></format>
        /// </example>
        public static LbfgsPoissonRegressionTrainer LbfgsPoissonRegression(this RegressionCatalog.RegressionTrainers catalog,
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
            return new LbfgsPoissonRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.LbfgsPoissonRegressionTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[PoissonRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/LbfgsPoissonRegressionWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static LbfgsPoissonRegressionTrainer LbfgsPoissonRegression(this RegressionCatalog.RegressionTrainers catalog, LbfgsPoissonRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new LbfgsPoissonRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the L-BFGS method implemented in <see cref="LbfgsMaximumEntropyMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Regularization">The L1 <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a> hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="l2Regularization">The L2 weight for <a href='https://en.wikipedia.org/wiki/Regularization_(mathematics)'>regularization</a>.</param>
        /// <param name="historySize">Memory size for <see cref="Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[PoissonRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LbfgsPoissonMaximumEntropy.cs)]
        /// ]]></format>
        /// </example>
        public static LbfgsMaximumEntropyMulticlassTrainer LbfgsMaximumEntropy(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
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
            return new LbfgsMaximumEntropyMulticlassTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the L-BFGS method implemented in <see cref="LbfgsMaximumEntropyMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[PoissonRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LbfgsPoissonMaximumEntropyWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static LbfgsMaximumEntropyMulticlassTrainer LbfgsMaximumEntropy(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            LbfgsMaximumEntropyMulticlassTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new LbfgsMaximumEntropyMulticlassTrainer(env, options);
        }

        /// <summary>
        /// Predicts a target using a linear multiclass classification model trained with the <see cref="NaiveBayesMulticlassTrainer"/>.
        /// The <see cref="NaiveBayesMulticlassTrainer"/> trains a multiclass Naive Bayes predictor that supports binary feature values.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[NaiveBayes](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/NaiveBayes.cs)]
        /// ]]></format>
        /// </example>
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
        /// <param name="maximumCalibrationExampleCount">Number of instances to train the calibrator.</param>
        /// <param name="useProbabilities">Use probabilities (vs. raw outputs) to identify top-score category.</param>
        /// <typeparam name="TModel">The type of the model. This type parameter will usually be inferred automatically from <paramref name="binaryEstimator"/>.</typeparam>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[OneVersusAll](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/OneVersusAll.cs)]
        /// ]]></format>
        /// </example>
        public static OneVersusAllTrainer OneVersusAll<TModel>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            ITrainerEstimator<BinaryPredictionTransformer<TModel>, TModel> binaryEstimator,
            string labelColumnName = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            IEstimator<ISingleFeaturePredictionTransformer<ICalibrator>> calibrator = null,
            int maximumCalibrationExampleCount = 1000000000,
            bool useProbabilities = true)
            where TModel : class
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            if (!(binaryEstimator is ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>> est))
                throw env.ExceptParam(nameof(binaryEstimator), "Trainer estimator does not appear to produce the right kind of model.");
            return new OneVersusAllTrainer(env, est, labelColumnName, imputeMissingLabelsAsNegative, GetCalibratorTrainerOrThrow(env, calibrator), maximumCalibrationExampleCount, useProbabilities);
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
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[PairwiseCoupling](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/PairwiseCoupling.cs)]
        /// ]]></format>
        /// </example>
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
        ///  [!code-csharp[Prior](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/PriorTrainer.cs)]
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
