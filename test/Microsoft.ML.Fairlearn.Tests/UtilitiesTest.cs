// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.Data.Analysis;
using Microsoft.ML.Fairlearn.reductions;
using Xunit;

namespace Microsoft.ML.Fairlearn.Tests
{
    public class UtilitiesTest
    {
        [Fact]
        public void Generate_binary_classification_lambda_search_space_test()
        {
            var context = new MLContext();
            var moment = new ClassificationMoment();
            var X = this.CreateDummyDataset();
            moment.LoadData(X, X["y_true"], X["sentitiveFeature"] as StringDataFrameColumn);

            var searchSpace = Utilities.GenerateBinaryClassificationLambdaSearchSpace(context, moment, 5);
            searchSpace.Keys.Should().BeEquivalentTo("a_pos", "a_neg", "b_pos", "b_neg");

        }
        private DataFrame CreateDummyDataset()
        {
            var df = new DataFrame();
            df["X"] = DataFrameColumn.Create("X", new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            df["y_true"] = DataFrameColumn.Create("y_true", new[] { true, true, true, true, true, true, true, false, false, false });
            df["y_pred"] = DataFrameColumn.Create("y_pred", new[] { true, true, true, true, false, false, false, true, false, false });
            df["sentitiveFeature"] = DataFrameColumn.Create("sentitiveFeature", new[] { "a", "b", "a", "a", "b", "a", "b", "b", "a", "b" });

            return df;
        }
    }
}
