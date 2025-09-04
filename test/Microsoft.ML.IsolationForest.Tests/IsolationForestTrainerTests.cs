// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Xunit;

namespace Microsoft.ML.IsolationForest.Tests
{
    public class IsolationForestTrainerTests()
    {
        private class Row
        {
            public float X1 { get; set; }
            public float X2 { get; set; }

            public Row(float x1, float x2)
            {
                X1 = x1;
                X2 = x2;
            }
        }

        private class OutRow
        {
#pragma warning disable S1144 // Unused properties, required for ML.NET schema binding
            public float X1 { get; set; }
            public float X2 { get; set; }
            // Initialize so Sonar S3459 is satisfied; ML.NET will overwrite these.
            public float Score { get; set; } = float.NaN;
            public bool PredictedLabel { get; set; }
#pragma warning restore S1144
        }

        [Fact]
        public void TrainAndScore_Works()
        {
            var ml = new MLContext(seed: 1);
            var data = new[]
            {
                new Row(0.2f, 0.1f),
                new Row(-0.1f, 0.05f),
                new Row(6.0f, 6.0f)
            };
            var dv = ml.Data.LoadFromEnumerable(data);
            var pipeline = ml.Transforms.Concatenate("Features", nameof(Row.X1), nameof(Row.X2))
                .Append(new IsolationForestTrainer(ml, new IsolationForestTrainer.Options
                {
                    Contamination = 0.2,
                    Trees = 50,
                    SampleSize = 64
                }));

            var model = pipeline.Fit(dv);
            var scored = model.Transform(dv);
            var rows = ml.Data.CreateEnumerable<OutRow>(scored, reuseRowObject: false).ToArray();

            Assert.Equal(3, rows.Length);

            if (!(rows[0].Score > 0 && rows[0].Score < 1))
            {
                throw new Xunit.Sdk.XunitException("Row[0] score out of expected range (0,1).");
            }

            if (rows[2].Score < rows[0].Score)
            {
                throw new Xunit.Sdk.XunitException("Row[2] score should be greater than or equal to Row[0] score.");
            }
        }
    }
}
