// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
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
                var fieldName = fieldDescriptor.Name;
                var column = tabularDataResource.Data.Select(row =>
                {
                    if (row is IDictionary<string, object> dictionary)
                    {
                        return dictionary[fieldName];
                    }
                    else
                    {
                        return row.FirstOrDefault(kvp => kvp.Key == fieldName).Value;
                    }
                });

                switch (fieldDescriptor.Type)
                {
                    case TableSchemaFieldType.Number:
                        dataFrame.Columns.Add(new DoubleDataFrameColumn(fieldDescriptor.Name, column.Select(Convert.ToDouble)));
                        break;
                    case TableSchemaFieldType.Integer:
                        dataFrame.Columns.Add(new Int64DataFrameColumn(fieldDescriptor.Name, column.Select(Convert.ToInt64)));
                        break;
                    case TableSchemaFieldType.Boolean:
                        dataFrame.Columns.Add(new BooleanDataFrameColumn(fieldDescriptor.Name, column.Select(Convert.ToBoolean)));
                        break;
                    case TableSchemaFieldType.String:
                        dataFrame.Columns.Add(new StringDataFrameColumn(fieldDescriptor.Name, column.Select(Convert.ToString)));
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            return dataFrame;
        }
    }
}
