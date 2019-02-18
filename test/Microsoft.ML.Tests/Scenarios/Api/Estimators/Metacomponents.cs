// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
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
        public void Metacomponents()
        {
            var ml = new MLContext();
            var data = ml.Data.ReadFromTextFile<IrisData>(GetDataPath(TestDatasets.irisData.trainFilename), separatorChar: ',');

            var sdcaTrainer = ml.BinaryClassification.Trainers.StochasticDualCoordinateAscentNonCalibrated(
                new SdcaNonCalibratedBinaryTrainer.Options { MaxIterations = 100, Shuffle = true, NumThreads = 1, });

            var pipeline = new ColumnConcatenatingEstimator (ml, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(ml.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.OneVersusAll(sdcaTrainer))
                .Append(ml.Transforms.Conversion.MapKeyToValue(("PredictedLabel")));

            var model = pipeline.Fit(data);
        }
    }
}