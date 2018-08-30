// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.IO;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact(Skip = "Disabled due to this bug https://github.com/dotnet/machinelearning/issues/770")]
        public void TensorFlowTransformCifarLearningPipelineTest()
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_model/frozen_model.pb";
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var pipeline = new LearningPipeline(seed: 1);
            pipeline.Add(new Microsoft.ML.Data.TextLoader(dataFile).CreateFrom<CifarData>(useHeader: false));
            pipeline.Add(new ImageLoader(("ImagePath", "ImageReal"))
            {
                ImageFolder = imageFolder
            });

            pipeline.Add(new ImageResizer(("ImageReal", "ImageCropped"))
            {
                ImageHeight = imageHeight,
                ImageWidth = imageWidth,
                Resizing = ImageResizerTransformResizingKind.IsoCrop
            });

            pipeline.Add(new ImagePixelExtractor(("ImageCropped", "Input"))
            {
                UseAlpha = false,
                InterleaveArgb = true
            });

            pipeline.Add(new TensorFlowScorer()
            {
                ModelFile = model_location,
                InputColumns = new[] { "Input" },
                OutputColumn = "Output"
            });

            using (var environment = new TlcEnvironment())
            {
                IDataView trans = pipeline.Execute(environment);
                Assert.NotNull(trans);

                trans.Schema.TryGetColumnIndex("Output", out int output);
                using (var cursor = trans.GetRowCursor(col => col == output))
                {
                    var buffer = default(VBuffer<float>);
                    var getter = cursor.GetGetter<VBuffer<float>>(output);
                    while (cursor.MoveNext())
                    {
                        getter(ref buffer);
                        Assert.Equal(10, buffer.Length);
                    }
                }
            }
        }
    }

    public class CifarData
    {
        [Column("0")]
        public string ImagePath;

        [Column("1")]
        public string Name;
    }
}
