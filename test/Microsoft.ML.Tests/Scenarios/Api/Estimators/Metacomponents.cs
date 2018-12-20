// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.RunTests;
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
            var ml = new MLContext();
            var data = ml.Data.CreateTextReader(TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',')
                .Read(GetDataPath(TestDatasets.irisData.trainFilename));

            var sdcaTrainer = ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features", advancedSettings: (s) => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; });

            var pipeline = ml.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(ml.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.OneVersusAll(sdcaTrainer))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);
        }
    }
}