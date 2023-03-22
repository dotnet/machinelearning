// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms.Image;
using Microsoft.VisualBasic;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    [Collection("NoParallelization")]

    public class ObjectDetectionTests : TestDataPipeBase
    {
        public ObjectDetectionTests(ITestOutputHelper helper) : base(helper)
        {
        }

        private class SimpleInput
        {
            [ImageType(10, 10)]
            public MLImage Image { get; set; }
            public string Labels;
            public string Box;

            public SimpleInput(int width, int height, byte red, byte green, byte blue, string labels, string box)
            {
                byte[] imageData = new byte[width * height * 4]; // 4 for the red, green, blue and alpha colors
                for (int i = 0; i < imageData.Length; i += 4)
                {
                    // Fill the buffer with the Bgra32 format
                    imageData[i] = blue;
                    imageData[i + 1] = green;
                    imageData[i + 2] = red;
                    imageData[i + 3] = 255;
                }

                Image = MLImage.CreateFromPixels(width, height, MLPixelFormat.Bgra32, imageData);
                Labels = labels;
                Box = box;
            }
        }

        [Fact]
        public void SimpleObjDetectionTest()
        {
            var images = new List<SimpleInput>() { new SimpleInput(10, 10, red: 0, green: 0, blue: 255, "dog cat", "1 1 1 1 2 2 2 2"), new SimpleInput(10, 10, red: 255, green: 0, blue: 0, "dog monkey", "1 1 1 1 2 2 2 2") };

            // Convert the list of data points to an IDataView object, which is consumable by ML.NET API.
            var data = ML.Data.LoadFromEnumerable(images);

            var pipeline = ML.Transforms.Text.TokenizeIntoWords("Labels")
                .Append(ML.Transforms.Conversion.MapValueToKey("Labels"))
                .Append(ML.Transforms.Text.TokenizeIntoWords("Box"))
                .Append(ML.Transforms.Conversion.ConvertType("Box"))
                .Append(ML.MulticlassClassification.Trainers.ObjectDetection("Labels", boundingBoxColumnName: "Box"));
            //var p1 = ML.Transforms.Text.TokenizeIntoWords("Box").Append(ML.Transforms.Conversion.ConvertType("Box"));
            //var prev1 = p1.Fit(data).Transform(data).Preview();

            var prev = pipeline.Fit(data).Transform(data).Preview();
            //TestEstimatorCore(pipeline, data);
        }
    }
}
