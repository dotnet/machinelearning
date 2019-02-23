// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Auto
{
    internal static class AutoMlUtils
    {
        public static Random Random = new Random();

        public static void Assert(bool boolVal, string message = null)
        {
            if(!boolVal)
            {
                message = message ?? "Assertion failed";
                throw new InvalidOperationException(message);
            }
        }

        public static IDataView DropLastColumn(this IDataView data, MLContext context)
        {
            return context.Transforms.DropColumns(data.Schema[data.Schema.Count - 1].Name).Fit(data).Transform(data);
        }

        public static (IDataView testData, IDataView validationData) TestValidateSplit(this TrainCatalogBase catalog, 
            MLContext context, IDataView trainData)
        {
            IDataView validationData;
            var splitData = catalog.TrainTestSplit(trainData);
            trainData = splitData.TrainSet;
            validationData = splitData.TestSet;
            trainData = trainData.DropLastColumn(context);
            validationData = validationData.DropLastColumn(context);
            return (trainData, validationData);
        }

        public static (string, DataViewType, ColumnPurpose, ColumnDimensions)[] GetColumnInfoTuples(MLContext context,
            IDataView data, ColumnInformation columnInfo)
        {
            var purposes = PurposeInference.InferPurposes(context, data, columnInfo);
            var colDimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, purposes);
            var cols = new (string, DataViewType, ColumnPurpose, ColumnDimensions)[data.Schema.Count];
            for (var i = 0; i < cols.Length; i++)
            {
                var schemaCol = data.Schema[i];
                var col = (schemaCol.Name, schemaCol.Type, purposes[i].Purpose, colDimensions[i]);
                cols[i] = col;
            }
            return cols;
        }
    }
}