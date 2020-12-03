// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.TestFrameworkCommon;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.IntegrationTests
{
    public class SchemaDefinitionTests : IntegrationTestBaseClass
    {
        private MLContext _ml;

        public SchemaDefinitionTests(ITestOutputHelper output) : base(output)
        {
        }

        protected override void Initialize()
        {
            base.Initialize();

            _ml = new MLContext(42);
            _ml.Log += LogTestOutput;
        }

        [Fact]
        public void SchemaDefinitionForPredictionEngine()
        {
            var fileName = TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename);
            var loader = _ml.Data.CreateTextLoader(new TextLoader.Options(), new MultiFileSource(fileName));
            var data = loader.Load(new MultiFileSource(fileName));
            var pipeline1 = _ml.Transforms.Categorical.OneHotEncoding("Cat", "Workclass", maximumNumberOfKeys: 3)
                .Append(_ml.Transforms.Concatenate("Features", "Cat", "NumericFeatures"));
            var model1 = pipeline1.Fit(data);

            var pipeline2 = _ml.Transforms.Categorical.OneHotEncoding("Cat", "Workclass", maximumNumberOfKeys: 4)
            .Append(_ml.Transforms.Concatenate("Features", "Cat", "NumericFeatures"));
            var model2 = pipeline2.Fit(data);

            var outputSchemaDefinition = SchemaDefinition.Create(typeof(OutputData));
            outputSchemaDefinition["Features"].ColumnType = model1.GetOutputSchema(data.Schema)["Features"].Type;
            var engine1 = _ml.Model.CreatePredictionEngine<InputData, OutputData>(model1, outputSchemaDefinition: outputSchemaDefinition);

            outputSchemaDefinition = SchemaDefinition.Create(typeof(OutputData));
            outputSchemaDefinition["Features"].ColumnType = model2.GetOutputSchema(data.Schema)["Features"].Type;
            var engine2 = _ml.Model.CreatePredictionEngine<InputData, OutputData>(model2, outputSchemaDefinition: outputSchemaDefinition);

            var prediction = engine1.Predict(new InputData() { Workclass = "Self-emp-not-inc", NumericFeatures = new float[6] });
            Assert.Equal((engine1.OutputSchema["Features"].Type as VectorDataViewType).Size, prediction.Features.Length);
            Assert.True(prediction.Features.All(x => x == 0));
            prediction = engine2.Predict(new InputData() { Workclass = "Self-emp-not-inc", NumericFeatures = new float[6] });
            Assert.Equal((engine2.OutputSchema["Features"].Type as VectorDataViewType).Size, prediction.Features.Length);
            Assert.True(prediction.Features.Select((x, i) => i == 3 && x == 1 || x == 0).All(b => b));
        }

        [Fact]
        public void SchemaDefinitionForCustomMapping()
        {
            var fileName = TestCommon.GetDataPath(DataDir, TestDatasets.adult.trainFilename);
            var data = new MultiFileSource(fileName);
            var loader = _ml.Data.CreateTextLoader(new TextLoader.Options(), new MultiFileSource(fileName));
            var pipeline = _ml.Transforms.Categorical.OneHotEncoding("Categories")
                .Append(_ml.Transforms.Categorical.OneHotEncoding("Workclass"))
                .Append(_ml.Transforms.Concatenate("Features", "NumericFeatures", "Categories", "Workclass"))
                .Append(_ml.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("Features"));
            var model = pipeline.Fit(loader.Load(data));
            var schema = model.GetOutputSchema(loader.GetOutputSchema());

            var inputSchemaDefinition = SchemaDefinition.Create(typeof(OutputData));
            inputSchemaDefinition["Features"].ColumnType = schema["Features"].Type;
            var outputSchemaDefinition = SchemaDefinition.Create(typeof(OutputData));
            outputSchemaDefinition["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, (schema["Features"].Type as VectorDataViewType).Size * 2);

            var custom = _ml.Transforms.CustomMapping(
                (OutputData src, OutputData dst) =>
                {
                    dst.Features = new float[src.Features.Length * 2];
                    for (int i = 0; i < src.Features.Length; i++)
                    {
                        dst.Features[2 * i] = src.Features[i];
                        dst.Features[2 * i + 1] = (float)Math.Log(src.Features[i]);
                    }
                }, null, inputSchemaDefinition, outputSchemaDefinition);

            model = model.Append(custom.Fit(model.Transform(loader.Load(data))) as ITransformer);
            schema = model.GetOutputSchema(loader.GetOutputSchema());
            Assert.Equal(168, (schema["Features"].Type as VectorDataViewType).Size);
        }

        private sealed class InputData
        {
            [LoadColumn(0)]
            public float Label { get; set; }
            [LoadColumn(1)]
            public string Workclass { get; set; }
            [LoadColumn(2, 8)]
            public string[] Categories { get; set; }
            [LoadColumn(9, 14)]
            [VectorType(6)]
            public float[] NumericFeatures { get; set; }
        }

        private sealed class OutputData
        {
            public float Label { get; set; }
            public float[] Features { get; set; }
        }
    }
}
