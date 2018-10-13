// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Meta-components: Meta-components (for example, components that themselves instantiate components) should not be booby-trapped.
        /// When specifying what trainer OVA should use, a user will be able to specify any binary classifier.
        /// If they specify a regression or multi-class classifier ideally that should be a compile error.
        /// </summary>
        [Fact]
        public void New_Metacomponents()
        {
            using (var env = new LocalEnvironment())
            {
                var data = new TextLoader(env, MakeIrisTextLoaderArgs())
                    .Read(new MultiFileSource(GetDataPath(TestDatasets.irisData.trainFilename)));

                var sdcaTrainer = new LinearClassificationTrainer(env, "Features", "Label", advancedSettings: (s) => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; });
                var pipeline = new ConcatEstimator(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                    .Append(new TermEstimator(env, "Label"), TransformerScope.TrainTest)
                    .Append(new Ova(env, sdcaTrainer))
                    .Append(new KeyToValueEstimator(env, "PredictedLabel"));

                var model = pipeline.Fit(data);
            }
        }
    }
}