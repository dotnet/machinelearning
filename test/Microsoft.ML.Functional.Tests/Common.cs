// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.ML.Functional.Tests
{
    internal static class Common
    {
        public static void CheckMetrics(RegressionMetrics metrics)
        {
            // Perform sanity checks on the metrics
            Assert.True(metrics.Rms >= 0);
            Assert.True(metrics.L1 >= 0);
            Assert.True(metrics.L2 >= 0);
            Assert.True(metrics.RSquared <= 1);
        }

        public static void AssertEqual(float[] array1, float[] array2)
        {
            Assert.NotNull(array1);
            Assert.NotNull(array2);
            Assert.Equal(array1.Length, array2.Length);

            for (int i = 0; i < array1.Length; i++)
                Assert.Equal(array1[i], array2[i]);
        }

        public static void AssertEqual(Schema schema1, Schema schema2)
        {
            Assert.NotNull(schema1);
            Assert.NotNull(schema2);

            Assert.Equal(schema1.Count(), schema2.Count());

            foreach (var schemaPair in schema1.Zip(schema2, Tuple.Create))
            {
                Assert.Equal(schemaPair.Item1.Name, schemaPair.Item2.Name);
                Assert.Equal(schemaPair.Item1.Index, schemaPair.Item2.Index);
                Assert.Equal(schemaPair.Item1.IsHidden, schemaPair.Item2.IsHidden);
                // Can probably do a better comparison of Metadata
                AssertEqual(schemaPair.Item1.Metadata.Schema, schemaPair.Item1.Metadata.Schema);
                Assert.True((schemaPair.Item1.Type == schemaPair.Item2.Type) ||
                    (schemaPair.Item1.Type.RawType == schemaPair.Item2.Type.RawType));
            }
        }
    }
}
