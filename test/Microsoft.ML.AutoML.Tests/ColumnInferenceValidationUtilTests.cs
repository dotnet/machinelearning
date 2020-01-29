﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.ML.AutoML.Test
{
    
    public class ColumnInferenceValidationUtilTests
    {
        [Fact]
        public void ValidateColumnNotContainedInData()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = DataViewTestFixture.BuildDummyDataView(schema);
            var columnInfo = new ColumnInformation();
            columnInfo.CategoricalColumnNames.Add("Categorical");
            Assert.Throws<ArgumentException>(() => ColumnInferenceValidationUtil.ValidateSpecifiedColumnsExist(columnInfo, dataView));
        }
    }
}
