// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.CategoricalTransforms;
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
            var ml = new MLContext();
            var data = ml.Data.TextReader(MakeIrisTextLoaderArgs())
                .Read(GetDataPath(TestDatasets.irisData.trainFilename));

            var sdcaTrainer = ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: (s) => { s.MaxIterations = 100; s.Shuffle = true; s.NumThreads = 1; });

            var pipeline = new ColumnConcatenatingEstimator (ml, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(new ValueToKeyMappingEstimator(ml, "Label"), TransformerScope.TrainTest)
                .Append(new Ova(ml, sdcaTrainer))
                .Append(new KeyToValueEstimator(ml, "PredictedLabel"));

            var model = pipeline.Fit(data);
        }
    }
}