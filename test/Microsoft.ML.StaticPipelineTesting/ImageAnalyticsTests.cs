// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.StaticPipelineTesting
{
    public sealed class ImageAnalyticsTests : BaseTestClassWithConsole
    {
        public ImageAnalyticsTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void SimpleImageSmokeTest()
        {
            var env = new ConsoleEnvironment(0, verbose: true);

            var reader = TextLoader.CreateReader(env,
                ctx => ctx.LoadText(0).LoadAsImage().AsGrayscale().Resize(10, 8).ExtractPixels());

            var schema = reader.AsDynamic.GetOutputSchema();
            Assert.True(schema.TryGetColumnIndex("Data", out int col), "Could not find 'Data' column");
            var type = schema.GetColumnType(col);
            Assert.True(type.IsKnownSizeVector, $"Type was supposed to be known size vector but was instead '{type}'");
            var vecType = type.AsVector;
            Assert.Equal(NumberType.R4, vecType.ItemType);
            Assert.Equal(3, vecType.DimCount);
            Assert.Equal(3, vecType.GetDim(0));
            Assert.Equal(8, vecType.GetDim(1));
            Assert.Equal(10, vecType.GetDim(2));

            var readAsImage = TextLoader.CreateReader(env,
                ctx => ctx.LoadText(0).LoadAsImage());
            var est = readAsImage.MakeNewEstimator().Append(r => r.AsGrayscale().Resize(10, 8).ExtractPixels());
            var pipe= readAsImage.Append(est);
        }
    }
}
