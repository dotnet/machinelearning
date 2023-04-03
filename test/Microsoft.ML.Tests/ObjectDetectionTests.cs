// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms.Image;
using Microsoft.VisualBasic;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.TorchSharp.AutoFormerV2;

namespace Microsoft.ML.Tests
{
    [Collection("NoParallelization")]

    public class ObjectDetectionTests : TestDataPipeBase
    {
        public ObjectDetectionTests(ITestOutputHelper helper) : base(helper)
        {
        }

        [Fact]
        public void SimpleObjDetectionTest()
        {
            var dataFile = GetDataPath("images/object-detection/fruit-detection-ten.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = TextLoader.Create(ML, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                    new TextLoader.Column("Labels", DataKind.String, 1),
                    new TextLoader.Column("Box", DataKind.String, 2)
                },
                MaxRows = 2
            }, new MultiFileSource(dataFile));
            var trainTest = ML.Data.TrainTestSplit(data, .1);

            var chain = new EstimatorChain<ITransformer>();

            var filteredPipeline = chain.Append(ML.Transforms.Text.TokenizeIntoWords("Labels", separators: new char[] { ',' }), TransformerScope.Training)
                .Append(ML.Transforms.Conversion.MapValueToKey("Labels"), TransformerScope.Training)
                .Append(ML.Transforms.Text.TokenizeIntoWords("Box", separators: new char[] { ',' }), TransformerScope.Training)
                .Append(ML.Transforms.Conversion.ConvertType("Box"), TransformerScope.Training)
                .Append(ML.Transforms.LoadImages("Image", imageFolder, "ImagePath"))
                .Append(ML.MulticlassClassification.Trainers.ObjectDetection("Labels", boundingBoxColumnName: "Box"))
                .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            var options = new ObjectDetectionTrainer.Options()
            {
                LabelColumnName = "Labels",
                BoundingBoxColumnName = "Box",
                ScoreThreshold = .5,
                MaxEpoch = 1
            };

            var pipeline = ML.Transforms.Text.TokenizeIntoWords("Labels", separators: new char[] { ',' })
                .Append(ML.Transforms.Conversion.MapValueToKey("Labels"))
                .Append(ML.Transforms.Text.TokenizeIntoWords("Box", separators: new char[] { ',' }))
                .Append(ML.Transforms.Conversion.ConvertType("Box"))
                .Append(ML.Transforms.LoadImages("Image", imageFolder, "ImagePath"))
                .Append(ML.MulticlassClassification.Trainers.ObjectDetection(options))
                .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainTest.TrainSet);
            var idv = model.Transform(trainTest.TestSet);

            // Make sure the metrics work.
            var metrics = ML.MulticlassClassification.EvaluateObjectDetection(idv, idv.Schema[2], idv.Schema[6], idv.Schema["PredictedLabel"], idv.Schema["Box"], idv.Schema["Score"]);

            Assert.True(metrics.MAP50 != float.NaN);
            Assert.True(metrics.MAP50_95 != float.NaN);

            // Make sure the filtered pipeline can run without any columns but image column AFTER training
            var dataFiltered = TextLoader.Create(ML, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                },
                MaxRows = 2
            }, new MultiFileSource(dataFile));

            var prev = filteredPipeline.Fit(trainTest.TrainSet).Transform(dataFiltered).Preview();
            Assert.Equal(2, prev.RowView.Count());

            TestEstimatorCore(pipeline, data);
        }
    }
}
