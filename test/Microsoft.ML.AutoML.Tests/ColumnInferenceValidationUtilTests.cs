// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class ColumnInferenceValidationUtilTests
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateColumnNotContainedInData()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);
            var columnInfo = new ColumnInformation();
            columnInfo.CategoricalColumnNames.Add("Categorical");
            ColumnInferenceValidationUtil.ValidateSpecifiedColumnsExist(columnInfo, dataView);
        }
    }
}
