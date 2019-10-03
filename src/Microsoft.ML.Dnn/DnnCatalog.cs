// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;
using static Microsoft.ML.Transforms.ImageClassificationEstimator;
using Options = Microsoft.ML.Transforms.DnnRetrainEstimator.Options;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="DnnRetrainTransformer"]/*' />
    public static class DnnCatalog
    {

        /// <summary>
        /// Retrain the dnn model on new data.
        /// usage of this API requires additional NuGet dependencies on TensorFlow redist, see linked document for more information.
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
        /// ]]>
        /// </format>
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <param name="labelColumnName">Name of the label column.</param>
        /// <param name="dnnLabel">Name of the node in DNN graph that is used as label during training in Dnn.
        /// The value of <paramref name="labelColumnName"/> from <see cref="IDataView"/> is fed to this node.</param>
        /// <param name="optimizationOperation">The name of the optimization operation in the Dnn graph.</param>
        /// <param name="modelPath">Path to model file to retrain.</param>
        /// <param name="epoch">Number of training iterations.</param>
        /// <param name="batchSize">Number of samples to use for mini-batch training.</param>
        /// <param name="lossOperation">The name of the operation in the Dnn graph to compute training loss (Optional).</param>
        /// <param name="metricOperation">The name of the operation in the Dnn graph to compute performance metric during training (Optional).</param>
        /// <param name="learningRateOperation">The name of the operation in the Dnn graph which sets optimizer learning rate (Optional).</param>
        /// <param name="learningRate">Learning rate to use during optimization (Optional).</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        /// <remarks>
        /// The support for retraining is under preview.
        /// </remarks>
        public static DnnRetrainEstimator RetrainDnnModel(
            this ModelOperationsCatalog catalog,
            string[] outputColumnNames,
            string[] inputColumnNames,
            string labelColumnName,
            string dnnLabel,
            string optimizationOperation,
            string modelPath,
            int epoch = 10,
            int batchSize = 20,
            string lossOperation = null,
            string metricOperation = null,
            string learningRateOperation = null,
            float learningRate = 0.01f,
            bool addBatchDimensionInput = false)
        {
            var options = new Options()
            {
                ModelLocation = modelPath,
                InputColumns = inputColumnNames,
                OutputColumns = outputColumnNames,
                LabelColumn = labelColumnName,
                TensorFlowLabel = dnnLabel,
                OptimizationOperation = optimizationOperation,
                LossOperation = lossOperation,
                MetricOperation = metricOperation,
                Epoch = epoch,
                LearningRateOperation = learningRateOperation,
                LearningRate = learningRate,
                BatchSize = batchSize,
                AddBatchDimensionInputs = addBatchDimensionInput
            };

            var env = CatalogUtils.GetEnvironment(catalog);
            return new DnnRetrainEstimator(env, options, DnnUtils.LoadDnnModel(env, modelPath, true));
        }

        /// <summary>
        /// Performs image classification using transfer learning.
        /// usage of this API requires additional NuGet dependencies on TensorFlow redist, see linked document for more information.
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
        /// ]]>
        /// </format>
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="featuresColumnName">The name of the input features column.</param>
        /// <param name="labelColumnName">The name of the labels column.</param>
        /// <param name="scoreColumnName">The name of the output score column.</param>
        /// <param name="predictedLabelColumnName">The name of the output predicted label columns.</param>
        /// <param name="arch">The architecture of the image recognition DNN model.</param>
        /// <param name="epoch">Number of training iterations. Each iteration/epoch refers to one pass over the dataset.</param>
        /// <param name="batchSize">The batch size for training.</param>
        /// <param name="learningRate">The learning rate for training.</param>
        /// <param name="disableEarlyStopping">Whether to disable use of early stopping technique. Training will go on for the full epoch count.</param>
        /// <param name="earlyStopping">Early stopping technique parameters to be used to terminate training when training metric stops improving.</param>
        /// <param name="metricsCallback">Callback for reporting model statistics during training phase.</param>
        /// <param name="statisticFrequency">Indicates the frequency of epochs at which to report model statistics during training phase.</param>
        /// <param name="framework">Indicates the choice of DNN training framework. Currently only tensorflow is supported.</param>
        /// <param name="modelSavePath">Optional name of the path where a copy new graph should be saved. The graph will be saved as part of model.</param>
        /// <param name="finalModelPrefix">The name of the prefix for the final mode and checkpoint files.</param>
        /// <param name="validationSet">Validation set.</param>
        /// <param name="testOnTrainSet">Indicates to evaluate the model on train set after every epoch.</param>
        /// <param name="reuseTrainSetBottleneckCachedValues">Indicates to not re-compute cached trainset bottleneck values if already available in the bin folder.</param>
        /// <param name="reuseValidationSetBottleneckCachedValues">Indicates to not re-compute validataionset cached bottleneck validationset values if already available in the bin folder.</param>
        /// <param name="trainSetBottleneckCachedValuesFilePath">Indicates the file path to store trainset bottleneck values for caching.</param>
        /// <param name="validationSetBottleneckCachedValuesFilePath">Indicates the file path to store validationset bottleneck values for caching.</param>
        /// <remarks>
        /// The support for image classification is under preview.
        /// </remarks>
        public static ImageClassificationEstimator ImageClassification(
            this ModelOperationsCatalog catalog,
            string featuresColumnName,
            string labelColumnName,
            string scoreColumnName = "Score",
            string predictedLabelColumnName = "PredictedLabel",
            Architecture arch = Architecture.InceptionV3,
            int epoch = 100,
            int batchSize = 10,
            float learningRate = 0.01f,
            bool disableEarlyStopping = false,
            EarlyStopping earlyStopping = null,
            ImageClassificationMetricsCallback metricsCallback = null,
            int statisticFrequency = 1,
            DnnFramework framework = DnnFramework.Tensorflow,
            string modelSavePath = null,
            string finalModelPrefix = "custom_retrained_model_based_on_",
            IDataView validationSet = null,
            bool testOnTrainSet = true,
            bool reuseTrainSetBottleneckCachedValues = false,
            bool reuseValidationSetBottleneckCachedValues = false,
            string trainSetBottleneckCachedValuesFilePath = "trainSetBottleneckFile.csv",
            string validationSetBottleneckCachedValuesFilePath = "validationSetBottleneckFile.csv"
            )
        {
            var options = new ImageClassificationEstimator.Options()
            {
                ModelLocation = arch == Architecture.ResnetV2101 ? @"resnet_v2_101_299.meta" : @"InceptionV3.meta",
                InputColumns = new[] { featuresColumnName },
                OutputColumns = new[] { scoreColumnName, predictedLabelColumnName },
                LabelColumn = labelColumnName,
                TensorFlowLabel = labelColumnName,
                Epoch = epoch,
                LearningRate = learningRate,
                BatchSize = batchSize,
                EarlyStoppingCriteria = disableEarlyStopping ? null : earlyStopping == null ? new EarlyStopping() : earlyStopping,
                ScoreColumnName = scoreColumnName,
                PredictedLabelColumnName = predictedLabelColumnName,
                FinalModelPrefix = finalModelPrefix,
                Arch = arch,
                MetricsCallback = metricsCallback,
                StatisticsFrequency = statisticFrequency,
                Framework = framework,
                ModelSavePath = modelSavePath,
                ValidationSet = validationSet,
                TestOnTrainSet = testOnTrainSet,
                TrainSetBottleneckCachedValuesFilePath = trainSetBottleneckCachedValuesFilePath,
                ValidationSetBottleneckCachedValuesFilePath = validationSetBottleneckCachedValuesFilePath,
                ReuseTrainSetBottleneckCachedValues = reuseTrainSetBottleneckCachedValues,
                ReuseValidationSetBottleneckCachedValues = reuseValidationSetBottleneckCachedValues
            };

            if (!File.Exists(options.ModelLocation))
            {
                if (options.Arch == Architecture.InceptionV3)
                {
                    var baseGitPath = @"https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/InceptionV3.meta";
                    using (WebClient client = new WebClient())
                    {
                        client.DownloadFile(new Uri($"{baseGitPath}"), @"InceptionV3.meta");
                    }

                    baseGitPath = @"https://github.com/SciSharp/TensorFlow.NET/raw/master/data/tfhub_modules.zip";
                    using (WebClient client = new WebClient())
                    {
                        client.DownloadFile(new Uri($"{baseGitPath}"), @"tfhub_modules.zip");
                        ZipFile.ExtractToDirectory(Path.Combine(Directory.GetCurrentDirectory(), @"tfhub_modules.zip"), @"tfhub_modules");
                    }
                }
                else if(options.Arch == Architecture.ResnetV2101)
                {
                    var baseGitPath = @"https://aka.ms/mlnet-resources/image/ResNet101Tensorflow/resnet_v2_101_299.meta";
                    using (WebClient client = new WebClient())
                    {
                        client.DownloadFile(new Uri($"{baseGitPath}"), @"resnet_v2_101_299.meta");
                    }
                }
            }

            var env = CatalogUtils.GetEnvironment(catalog);
            return new ImageClassificationEstimator(env, options, DnnUtils.LoadDnnModel(env, options.ModelLocation, true));
        }
    }
}
