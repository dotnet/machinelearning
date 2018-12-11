// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Online;
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
            var averagePerceptron = new AveragedPerceptronTrainer(Env, "Label", "Features", advancedSettings: s =>
             {
                 s.Shuffle = true;
                 s.Calibrator = null;
             });

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
            var sdcaTrainer = new SdcaBinaryTrainer(Env, "Label", "Features", advancedSettings: (s) => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; s.Calibrator = null; });

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

            var sdcaTrainer = new SdcaBinaryTrainer(Env, "Label", "Features", advancedSettings: (s) => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; });
            pipeline = pipeline.Append(new Pkpd(Env, sdcaTrainer))
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
