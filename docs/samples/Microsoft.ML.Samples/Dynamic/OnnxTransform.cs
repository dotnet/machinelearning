// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Transforms;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML;
using System;
using System.IO;

namespace Microsoft.ML.Samples.Dynamic
{
    class OnnxTransformExample
    {
        public static void OnnxTransformSample(string[] args)
        {
            // Download the squeeznet image model from ONNX model zoo, version 1.2
            // https://github.com/onnx/models/tree/master/squeezenet
            var model_location = @"squeezenet\model.onnx";

            var env = new MLContext();

            // Use the utility functions to inspect models inputs, outputs, shape and type
            // Load the model using the OnnxModel class
            var onnxModel = new OnnxModel(model_location);

            // This model has only 1 input, so inspect 0th index for input node metadata
            var inputSchema = onnxModel.ModelInfo.InputsInfo[0];
            var inputName = inputSchema.Name;
            var inputShape = inputSchema.Shape;

            // Deduce image dimensions from inputShape
            var numChannels = (int) inputShape[1];
            var imageHeight = (int) inputShape[2];
            var imageWidth =  (int) inputShape[3];

            // Similarly, get output node metadata
            var outputSchema = onnxModel.ModelInfo.OutputsInfo[0];
            var outputName = outputSchema.Name;
            var outputShape = outputSchema.Shape;

            var dataFile = @"test\data\images\images.tsv";
            var imageFolder = Path.GetDirectoryName(dataFile);

            // Use Textloader to load the text file which references the images to load
            // Preview ...
            // banana.jpg banana
            // hotdog.jpg hotdog
            // tomato.jpg tomato
            var data = TextLoader.Create(env, new TextLoader.Arguments()
            {
                Column = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.TX, 0),
                    new TextLoader.Column("Name", DataKind.TX, 1),
                 }
            }, new MultiFileSource(dataFile));

            // Load the images referenced in the text file
            var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);

            // Resize the images to match model dimensions
            var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column(){ Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}}
            }, images);

            // Extract out the RBG pixel values. 
            // InterleaveArgb = true makes the values RGBRGBRGB. Otherwise it's RRR...GGG...BBB.
            var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
            {
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = inputName, InterleaveArgb=true}
                    }
            }, cropped);

            // Create OnnxTransform, passing in the input and output names the model expects.
            IDataView trans = OnnxTransform.Create(env, pixels, model_location, new[] { inputName }, new[] { outputName });

            trans.Schema.TryGetColumnIndex(outputName, out int output);
            using (var cursor = trans.GetRowCursor(col => col == output))
            {
                var numRows = 0;
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(output);
                // For each image, retrieve the model scores
                while (cursor.MoveNext())
                {
                    int i = 0;
                    getter(ref buffer);
                    // print scores for first 3 classes
                    foreach(var score in buffer.GetValues())
                    {
                        Console.WriteLine(String.Format("Example # {0} :Score for class {1} = {2} ",numRows, i, score));
                        if (++i > 2) break;
                    }
                    numRows += 1;
                }
            }
            // Results look like below...
            // Example # 0 :Score for class 0 = 1.133263E-06
            // Example # 0 :Score for class 1 = 1.80478E-07
            // Example # 0 :Score for class 2 = 1.595297E-07
            // Example # 1 :Score for class 0 = 1.805106E-05
            // Example # 1 :Score for class 1 = 1.257452E-05
            // Example # 1 :Score for class 2 = 2.412128E-06
            // Example # 2 :Score for class 0 = 1.346096E-06
            // Example # 2 :Score for class 1 = 1.918751E-07
            // Example # 2 :Score for class 2 = 7.203341E-08
        }
    }
}