// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.Analysis;


namespace Microsoft.DotNet.Interactive.Formatting.TabularData
{
    public static class TabularDataResourceExtensions
    {
        public static DataFrame ToDataFrame(this TabularDataResource tabularDataResource)
        {
            if (tabularDataResource == null)
            {
                throw new ArgumentNullException(nameof(tabularDataResource));
            }

            var dataFrame = new DataFrame();

            foreach (var fieldDescriptor in tabularDataResource.Schema.Fields)
            {
                switch (fieldDescriptor.Type)
                {
                    case TableSchemaFieldType.Number:
                        dataFrame.Columns.Add(new DoubleDataFrameColumn(fieldDescriptor.Name, tabularDataResource.Data.Select(d => Convert.ToDouble(d[fieldDescriptor.Name]))));
                        break;
                    case TableSchemaFieldType.Integer:
                        dataFrame.Columns.Add(new Int64DataFrameColumn(fieldDescriptor.Name, tabularDataResource.Data.Select(d => Convert.ToInt64(d[fieldDescriptor.Name]))));
                        break;
                    case TableSchemaFieldType.Boolean:
                        dataFrame.Columns.Add(new BooleanDataFrameColumn(fieldDescriptor.Name, tabularDataResource.Data.Select(d => Convert.ToBoolean(d[fieldDescriptor.Name]))));
                        break;
                    case TableSchemaFieldType.String:
                        dataFrame.Columns.Add(new StringDataFrameColumn(fieldDescriptor.Name, tabularDataResource.Data.Select(d => Convert.ToString(d[fieldDescriptor.Name]))));
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            return dataFrame;
        }
    }
}
