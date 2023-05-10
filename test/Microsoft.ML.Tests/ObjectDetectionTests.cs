// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.VisualBasic;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.TorchSharp.AutoFormerV2;
using Microsoft.ML.Runtime;
using System.Collections.Generic;

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
                MaxRows = 1
            }, new MultiFileSource(dataFile));

            var chain = new EstimatorChain<ITransformer>();

            var filteredPipeline = chain.Append(ML.Transforms.Text.TokenizeIntoWords("Labels", separators: new char[] { ',' }), TransformerScope.Training)
                .Append(ML.Transforms.Conversion.MapValueToKey("Labels"), TransformerScope.Training)
                .Append(ML.Transforms.Text.TokenizeIntoWords("Box", separators: new char[] { ',' }), TransformerScope.Training)
                .Append(ML.Transforms.Conversion.ConvertType("Box"), TransformerScope.Training)
                .Append(ML.Transforms.LoadImages("Image", imageFolder, "ImagePath"))
                .Append(ML.MulticlassClassification.Trainers.ObjectDetection("Labels", boundingBoxColumnName: "Box", maxEpoch: 1))
                .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var options = new ObjectDetectionTrainer.Options()
            {
                LabelColumnName = "Labels",
                BoundingBoxColumnName = "Box",
                ScoreThreshold = .5,
                MaxEpoch = 1,
                LogEveryNStep = 1,
            };

            var pipeline = ML.Transforms.Text.TokenizeIntoWords("Labels", separators: new char[] { ',' })
                .Append(ML.Transforms.Conversion.MapValueToKey("Labels"))
                .Append(ML.Transforms.Text.TokenizeIntoWords("Box", separators: new char[] { ',' }))
                .Append(ML.Transforms.Conversion.ConvertType("Box"))
                .Append(ML.Transforms.LoadImages("Image", imageFolder, "ImagePath"))
                .Append(ML.MulticlassClassification.Trainers.ObjectDetection(options))
                .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var logs = new List<LoggingEventArgs>();

            ML.Log += (o, e) =>
            {
                if (e.Source.StartsWith("ObjectDetectionTrainer") && e.Kind == ChannelMessageKind.Info && e.Message.Contains("Loss:"))
                {
                    logs.Add(e);
                }
            };

            var model = pipeline.Fit(data);
            var idv = model.Transform(data);
            // Make sure the metrics work.
            var metrics = ML.MulticlassClassification.EvaluateObjectDetection(idv, idv.Schema[2], idv.Schema["Box"], idv.Schema["PredictedLabel"], idv.Schema["PredictedBoundingBoxes"], idv.Schema["Score"]);
            Assert.True(!float.IsNaN(metrics.MAP50));
            Assert.True(!float.IsNaN(metrics.MAP50_95));

            // We aren't doing enough training to get a consistent loss, so just make sure its being logged
            Assert.True(logs.Count > 0);

            // Make sure the filtered pipeline can run without any columns but image column AFTER training
            var dataFiltered = TextLoader.Create(ML, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.String, 0),
                },
                MaxRows = 2
            }, new MultiFileSource(dataFile));
            var prev = filteredPipeline.Fit(data).Transform(dataFiltered).Preview();
            Assert.Equal(2, prev.RowView.Count());

            TestEstimatorCore(pipeline, data);
        }
    }
}
