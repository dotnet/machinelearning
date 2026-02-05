// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public static class DataFrameAssert
    {
        public static void Equal(DataFrame expected, DataFrame actual)
        {
            Assert.Equal(expected.Columns.Count, actual.Columns.Count);
            Assert.Equal(expected.Rows.Count, actual.Rows.Count);

            for (int c = 0; c < expected.Columns.Count; c++)
            {
                var expectedColumn = expected.Columns[c];
                var actualColumn = actual.Columns[c];

                Assert.Equal(expectedColumn.Name, actualColumn.Name);
                Assert.Equal(expectedColumn.GetType(), actualColumn.GetType());

                for (int r = 0; r < expected.Rows.Count; r++)
                {
                    var expectedValue = expectedColumn[r];
                    var actualValue = actualColumn[r];

                    if (expectedValue == null || actualValue == null)
                    {
                        Assert.Null(expectedValue);
                        Assert.Null(actualValue);
                    }
                    else
                    {
                        Assert.Equal(expectedValue, actualValue);
                    }
                }
            }
        }
    }
}
