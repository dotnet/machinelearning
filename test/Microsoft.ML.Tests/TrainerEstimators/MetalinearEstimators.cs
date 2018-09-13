// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Tests.Scenarios.Api;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class MetalinearEstimators : TestDataPipeBase
    {

        public MetalinearEstimators(ITestOutputHelper output) : base(output)
        {
        }


        /// <summary>
        /// OVA with calibrator argument
        /// </summary>
        [Fact]
        public void OVAWithExplicitCalibrator()
        {
            var dataPath = GetDataPath(IrisDataPath);

            using (var env = new TlcEnvironment())
            {
                var calibrator = new PavCalibratorTrainer(env);

                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));

                var sdcaTrainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 }, "Features", "Label");
                var pipeline = new MyConcatTransform(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, "Label"), TransformerScope.TrainTest)
                    .Append(new Ova(env, sdcaTrainer, "Label", calibrator: calibrator, maxCalibrationExamples: 990000))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data);
            }
        }

        /// <summary>
        /// OVA with all constructor args.
        /// </summary>
        [Fact]
        public void OVAWithAllConstructorArgs()
        {
            var dataPath = GetDataPath(IrisDataPath);
            string featNam = "Features";
            string labNam = "Label";

            using (var env = new TlcEnvironment())
            {
                var calibrator = new FixedPlattCalibratorTrainer(env, new FixedPlattCalibratorTrainer.Arguments());

                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));

                var averagePerceptron = new AveragedPerceptronTrainer(env, new AveragedPerceptronTrainer.Arguments { FeatureColumn = featNam, LabelColumn = labNam, Shuffle = true, Calibrator = null });
                var pipeline = new MyConcatTransform(env, featNam, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, labNam), TransformerScope.TrainTest)
                    .Append(new Ova(env, averagePerceptron, labNam, true, calibrator: calibrator, 10000, true))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                // TestEstimatorCore(pipeline, data);
                var model = pipeline.Fit(data);
            }
        }

        /// <summary>
        /// OVA un-calibrated
        /// </summary>
        [Fact]
        public void OVAUncalibrated()
        {
            var dataPath = GetDataPath(IrisDataPath);

            using (var env = new TlcEnvironment())
            {
                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));

                var sdcaTrainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1, Calibrator = null }, "Features", "Label");
                var pipeline = new MyConcatTransform(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, "Label"), TransformerScope.TrainTest)
                    .Append(new Ova(env, sdcaTrainer, useProbabilities: false))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data);
            }
        }

        /// <summary>
        /// Pkpd trainer
        /// </summary>
        [Fact]
        public void Pkpd()
        {
            var dataPath = GetDataPath(IrisDataPath);

            using (var env = new TlcEnvironment())
            {
                var calibrator = new PavCalibratorTrainer(env);

                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));

                var sdcaTrainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 }, "Features", "Label");
                var pipeline = new MyConcatTransform(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, "Label"), TransformerScope.TrainTest)
                    .Append(new Pkpd(env, sdcaTrainer))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data);
            }
        }
    }
}
