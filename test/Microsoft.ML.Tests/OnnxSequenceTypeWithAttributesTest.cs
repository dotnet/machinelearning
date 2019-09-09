using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using Xunit;
using Xunit.Abstractions;
using System.Linq;
using System.IO;

namespace Microsoft.ML.Tests
{
    public class OnnxSequenceTypeWithAttributesTest : BaseTestBaseline
    {
        public class ImagePrediction
        {
            [ColumnName("classLabel")]
            [VectorType]
            public string[] Prediction;

            [ColumnName("loss")]
            [OnnxSequenceType(typeof(IDictionary<string, float>))]
            public IEnumerable<IDictionary<string, float>> Loss;
        }
        public class ImageInput
        {
            [ImageType(224, 224)]
            public Bitmap Image { get; set; }
        }

        public OnnxSequenceTypeWithAttributesTest(ITestOutputHelper output) : base(output)
        {
        }
        public static PredictionEngine<ImageInput, ImagePrediction> LoadModel(string onnxModelFilePath)
        {
            var ctx = new MLContext();
            var dataView = ctx.Data.LoadFromEnumerable(new List<ImageInput>());

            var pipeline = ctx.Transforms.ResizeImages(
                                resizing: ImageResizingEstimator.ResizingKind.Fill,
                                outputColumnName: "data",
                                imageWidth: 224,
                                imageHeight: 224,
                                inputColumnName: nameof(ImageInput.Image))
                            .Append(ctx.Transforms.ExtractPixels(outputColumnName: "data"))
                            .Append(ctx.Transforms.ApplyOnnxModel(
                                modelFile: onnxModelFilePath,
                                outputColumnNames: new[] { "classLabel", "loss" }, inputColumnNames: new[] { "data" }));

            var model = pipeline.Fit(dataView);
            return ctx.Model.CreatePredictionEngine<ImageInput, ImagePrediction>(model);
        }

        [Fact]
        public void OnnxSequenceTypeWithColumnNameAttributeTest()
        {
            var modelFile = @"column_name_test/model.onnx";
            var predictor = LoadModel(modelFile);
            string image_path = Path.Combine(DataDir, "images", "banana.jpg");

            var output = predictor.Predict(new ImageInput { Image = (Bitmap)Image.FromFile(image_path) });
            Assert.NotEmpty(output.Prediction);
            var loss = output.Loss.FirstOrDefault();
            Assert.NotEmpty(loss);
            Assert.True(loss[output.Prediction[0]] > 0, "Invalid output");
        }
    }
}
