// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        /// <summary>
        /// OVA with all constructor args.
        /// </summary>
        [Fact]
        public void OVAWithAllConstructorArgs()
        {
            var (pipeline, data) = GetMultiClassPipeline();
            var calibrator = new PlattCalibratorTrainer(Env);
            var averagePerceptron = ML.BinaryClassification.Trainers.AveragedPerceptron(
                new AveragedPerceptronTrainer.Options { Shuffle = true, Calibrator = null });

            pipeline = pipeline.Append(new Ova(Env, averagePerceptron, "Label", true, calibrator: calibrator, 10000, true))
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// OVA un-calibrated
        /// </summary>
        [Fact]
        public void OVAUncalibrated()
        {
            var (pipeline, data) = GetMultiClassPipeline();
            var sdcaTrainer = ML.BinaryClassification.Trainers.StochasticDualCoordinateAscent(
                new SdcaBinaryTrainer.Options { MaxIterations = 100, Shuffle = true, NumThreads = 1, Calibrator = null });

            pipeline = pipeline.Append(new Ova(Env, sdcaTrainer, useProbabilities: false))
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// Pkpd trainer
        /// </summary>
        [Fact]
        public void Pkpd()
        {
            var (pipeline, data) = GetMultiClassPipeline();

            var sdcaTrainer = ML.BinaryClassification.Trainers.StochasticDualCoordinateAscent(
                new SdcaBinaryTrainer.Options { MaxIterations = 100, Shuffle = true, NumThreads = 1 });

            pipeline = pipeline.Append(new Pkpd(Env, sdcaTrainer))
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [Fact]
        public void MetacomponentsFeaturesRenamed()
        {
            var data = new TextLoader(Env, TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',')
                .Read(GetDataPath(TestDatasets.irisData.trainFilename));

            var sdcaTrainer = ML.BinaryClassification.Trainers.StochasticDualCoordinateAscent(
                new SdcaBinaryTrainer.Options {
                    LabelColumn = "Label",
                    FeatureColumn = "Vars",
                    MaxIterations = 100,
                    Shuffle = true,
                    NumThreads = 1, });

            var pipeline = new ColumnConcatenatingEstimator(Env, "Vars", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(new ValueToKeyMappingEstimator(Env, "Label"), TransformerScope.TrainTest)
                .Append(new Ova(Env, sdcaTrainer))
                .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            var model = pipeline.Fit(data);

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
