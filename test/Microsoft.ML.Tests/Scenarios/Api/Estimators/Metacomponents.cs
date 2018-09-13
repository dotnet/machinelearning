// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Meta-components: Meta-components (e.g., components that themselves instantiate components) should not be booby-trapped.
        /// When specifying what trainer OVA should use, a user will be able to specify any binary classifier.
        /// If they specify a regression or multi-class classifier ideally that should be a compile error.
        /// </summary>
        [Fact]
        public void New_Metacomponents()
        {
            var dataPath = GetDataPath(IrisDataPath);
            using (var env = new TlcEnvironment())
            {
                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));

                var sdcaTrainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1 }, "Features", "Label");
                var pipeline = new MyConcatTransform(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, "Label"), TransformerScope.TrainTest)
                    .Append(new Ova(env, sdcaTrainer))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data);
            }
        }

        /// <summary>
        /// OVA with calibrator
        /// </summary>
        [Fact]
        public void New_OVAWithCalibrator()
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
        /// OVA with calibrator
        /// </summary>
        [Fact]
        public void New_OVAWithAllConstructorArgs()
        {
            var dataPath = GetDataPath(IrisDataPath);

            using (var env = new TlcEnvironment())
            {
                var calibrator = new PavCalibratorTrainer(env);

                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(dataPath));

                var sdcaTrainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments { MaxIterations = 100, Shuffle = true, NumThreads = 1, Calibrator = null }, "Features", "Label");
                var pipeline = new MyConcatTransform(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, "Label"), TransformerScope.TrainTest)
                    .Append(new Ova(env, sdcaTrainer, "Label", true, calibrator: calibrator, 10000, true))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data);
            }
        }

        /// <summary>
        /// OVA with uncalibrated
        /// </summary>
        [Fact]
        public void New_OVAUncalibrated()
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
                    .Append(new Ova(env, sdcaTrainer, useProbabilities: false))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data);
            }
        }

        /// <summary>
        /// OVA with calibrator
        /// </summary>
        [Fact]
        public void New_Pkpd()
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
