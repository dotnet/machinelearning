// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Auto;

namespace Samples.Helpers
{
    internal static class ConsoleHelper
    {
        public static void Print(ColumnInferenceResults results)
        {
            Console.WriteLine("Inferred dataset columns --");
            new ColumnInferencePrinter(results).Print();
            Console.WriteLine();
        }

        public static string BuildStringTable(IList<string[]> arrValues)
        {
            int[] maxColumnsWidth = GetMaxColumnsWidth(arrValues);
            var headerSpliter = new string('-', maxColumnsWidth.Sum(i => i + 3) - 1);

            var sb = new StringBuilder();
            for (int rowIndex = 0; rowIndex < arrValues.Count; rowIndex++)
            {
                if (rowIndex == 0)
                {
                    sb.AppendFormat("  {0} ", headerSpliter);
                    sb.AppendLine();
                }

                for (int colIndex = 0; colIndex < arrValues[0].Length; colIndex++)
                {
                    // Print cell
                    string cell = arrValues[rowIndex][colIndex];
                    cell = cell.PadRight(maxColumnsWidth[colIndex]);
                    sb.Append(" | ");
                    sb.Append(cell);
                }

                // Print end of line
                sb.Append(" | ");
                sb.AppendLine();

                // Print splitter
                if (rowIndex == 0)
                {
                    sb.AppendFormat(" |{0}| ", headerSpliter);
                    sb.AppendLine();
                }

                if (rowIndex == arrValues.Count - 1)
                {
                    sb.AppendFormat("  {0} ", headerSpliter);
                }
            }

            return sb.ToString();
        }

        private static int[] GetMaxColumnsWidth(IList<string[]> arrValues)
        {
            var maxColumnsWidth = new int[arrValues[0].Length];
            for (int colIndex = 0; colIndex < arrValues[0].Length; colIndex++)
            {
                for (int rowIndex = 0; rowIndex < arrValues.Count; rowIndex++)
                {
                    int newLength = arrValues[rowIndex][colIndex].Length;
                    int oldLength = maxColumnsWidth[colIndex];

                    if (newLength > oldLength)
                    {
                        maxColumnsWidth[colIndex] = newLength;
                    }
                }
            }

            return maxColumnsWidth;
        }
    }

    internal class ColumnInferencePrinter
    {
        private static readonly string[] TableHeaders = new[] { "Name", "Data Type", "Purpose" };

        private readonly ColumnInferenceResults _results;

        public ColumnInferencePrinter(ColumnInferenceResults results)
        {
            _results = results;
        }

        public void Print()
        {
            var tableRows = new List<string[]>();

            // add headers
            tableRows.Add(TableHeaders);

            // add column data
            var info = _results.ColumnInformation;
            AppendTableRow(tableRows, info.LabelColumn, "Label");
            AppendTableRow(tableRows, info.WeightColumn, "Weight");
            AppendTableRows(tableRows, info.CategoricalColumns, "Categorical");
            AppendTableRows(tableRows, info.NumericColumns, "Numeric");
            AppendTableRows(tableRows, info.TextColumns, "Text");
            AppendTableRows(tableRows, info.IgnoredColumns, "Ignored");

            Console.WriteLine(ConsoleHelper.BuildStringTable(tableRows));
        }

        private void AppendTableRow(ICollection<string[]> tableRows,
            string columnName, string columnPurpose)
        {
            if (columnName == null)
            {
                return;
            }

            tableRows.Add(new[]
            {
                columnName,
                GetColumnDataType(columnName),
                columnPurpose
            });
        }

        private void AppendTableRows(ICollection<string[]> tableRows,
            IEnumerable<string> columnNames, string columnPurpose)
        {
            foreach (var columnName in columnNames)
            {
                AppendTableRow(tableRows, columnName, columnPurpose);
            }
        }

        private string GetColumnDataType(string columnName)
        {
            return _results.TextLoaderOptions.Columns.First(c => c.Name == columnName).DataKind.ToString();
        }
    }
}