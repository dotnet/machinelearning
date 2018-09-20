// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        /// <summary>
        /// OVA with calibrator argument
        /// </summary>
        [Fact]
        public void OVAWithExplicitCalibrator()
        {
            var (pipeline, data) = GetMultiClassPipeline();
            var calibrator = new PavCalibratorTrainer(Env);

            var sdcaTrainer = new LinearClassificationTrainer(Env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 }, "Features", "Label");
            pipeline.Append(new Ova(Env, sdcaTrainer, "Label", calibrator: calibrator, maxCalibrationExamples: 990000))
                    .Append(new KeyToValueEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// OVA with all constructor args.
        /// </summary>
        [Fact]
        public void OVAWithAllConstructorArgs()
        {
            var (pipeline, data) = GetMultiClassPipeline();
            var calibrator = new PlattCalibratorTrainer(Env);
            var averagePerceptron = new AveragedPerceptronTrainer(Env, new AveragedPerceptronTrainer.Arguments { FeatureColumn = "Features", LabelColumn = "Label", Shuffle = true, Calibrator = null });

            pipeline.Append(new Ova(Env, averagePerceptron, "Label", true, calibrator: calibrator, 10000, true))
                    .Append(new KeyToValueEstimator(Env, "PredictedLabel"));

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

            var sdcaTrainer = new LinearClassificationTrainer(Env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1, Calibrator = null }, "Features", "Label");

            pipeline.Append(new Ova(Env, sdcaTrainer, useProbabilities: false))
                    .Append(new KeyToValueEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// Pkpd trainer
        /// </summary>
        [Fact(Skip = "The test fails the check for valid input to fit")]
        public void Pkpd()
        {
            var (pipeline, data) = GetMultiClassPipeline(); 

            var sdcaTrainer = new LinearClassificationTrainer(Env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 }, "Features", "Label");
            pipeline.Append(new Pkpd(Env, sdcaTrainer))
                    .Append(new KeyToValueEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
