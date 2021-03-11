// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public partial class DataFrame
    {
        private const int DefaultStreamReaderBufferSize = 1024;

        private static Type GuessKind(int col, List<string[]> read)
        {
            Type res = typeof(string);
            int nbline = 0;
            foreach (var line in read)
            {
                if (col >= line.Length)
                    throw new FormatException(string.Format(Strings.LessColumnsThatExpected, nbline + 1));

                string val = line[col];

                if (string.Equals(val, "null", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (!string.IsNullOrEmpty(val))
                {
                    bool boolParse = bool.TryParse(val, out bool boolResult);
                    if (boolParse)
                    {
                        res = DetermineType(nbline == 0, typeof(bool), res);
                        ++nbline;
                        continue;
                    }
                    bool floatParse = float.TryParse(val, out float floatResult);
                    if (floatParse)
                    {
                        res = DetermineType(nbline == 0, typeof(float), res);
                        ++nbline;
                        continue;
                    }

                    res = DetermineType(nbline == 0, typeof(string), res);
                    ++nbline;
                }
            }
            return res;
        }

        private static Type DetermineType(bool first, Type suggested, Type previous)
        {
            if (first)
                return suggested;
            else
                return MaxKind(suggested, previous);
        }

        private static Type MaxKind(Type a, Type b)
        {
            if (a == typeof(string) || b == typeof(string))
                return typeof(string);
            if (a == typeof(float) || b == typeof(float))
                return typeof(float);
            if (a == typeof(bool) || b == typeof(bool))
                return typeof(bool);
            return typeof(string);
        }

        /// <summary>
        /// Reads a text file as a DataFrame.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="separator">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="columnNames">column names (can be empty)</param>
        /// <param name="dataTypes">column types (can be empty)</param>
        /// <param name="numRows">number of rows to read</param>
        /// <param name="guessRows">number of rows used to guess types</param>
        /// <param name="addIndexColumn">add one column with the row index</param>
        /// <param name="encoding">The character encoding. Defaults to UTF8 if not specified</param>
        /// <returns>DataFrame</returns>
        public static DataFrame LoadCsv(string filename,
                                char separator = ',', bool header = true,
                                string[] columnNames = null, Type[] dataTypes = null,
                                int numRows = -1, int guessRows = 10,
                                bool addIndexColumn = false, Encoding encoding = null)
        {
            using (Stream fileStream = new FileStream(filename, FileMode.Open))
            {
                return LoadCsv(fileStream,
                                  separator: separator, header: header, columnNames: columnNames, dataTypes: dataTypes, numberOfRowsToRead: numRows,
                                  guessRows: guessRows, addIndexColumn: addIndexColumn, encoding: encoding);
            }
        }

        private static string GetColumnName(string[] columnNames, int columnIndex)
        {
            return columnNames == null ? "Column" + columnIndex.ToString() : columnNames[columnIndex];
        }

        private static DataFrameColumn CreateColumn(Type kind, string[] columnNames, int columnIndex)
        {
            DataFrameColumn ret;
            if (kind == typeof(bool))
            {
                ret = new BooleanDataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(int))
            {
                ret = new Int32DataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(float))
            {
                ret = new SingleDataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(string))
            {
                ret = new StringDataFrameColumn(GetColumnName(columnNames, columnIndex), 0);
            }
            else if (kind == typeof(long))
            {
                ret = new Int64DataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(decimal))
            {
                ret = new DecimalDataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(byte))
            {
                ret = new ByteDataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(char))
            {
                ret = new CharDataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(double))
            {
                ret = new DoubleDataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(sbyte))
            {
                ret = new SByteDataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(short))
            {
                ret = new Int16DataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(uint))
            {
                ret = new UInt32DataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(ulong))
            {
                ret = new UInt64DataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else if (kind == typeof(ushort))
            {
                ret = new UInt16DataFrameColumn(GetColumnName(columnNames, columnIndex));
            }
            else
            {
                throw new NotSupportedException(nameof(kind));
            }
            return ret;
        }

        private static DataFrame ReadCsvLinesIntoDataFrame(IEnumerable<string> lines,
                                char separator = ',', bool header = true,
                                string[] columnNames = null, Type[] dataTypes = null,
                                long numberOfRowsToRead = -1, int guessRows = 10, bool addIndexColumn = false
                                )
        {
            if (dataTypes == null && guessRows <= 0)
            {
                throw new ArgumentException(string.Format(Strings.ExpectedEitherGuessRowsOrDataTypes, nameof(guessRows), nameof(dataTypes)));
            }

            var linesForGuessType = new List<string[]>();
            long rowline = 0;
            int numberOfColumns = dataTypes?.Length ?? 0;

            if (header == true && numberOfRowsToRead != -1)
            {
                numberOfRowsToRead++;
            }

            List<DataFrameColumn> columns;
            // First pass: schema and number of rows.
            string line = null;

            var enumerator = lines.GetEnumerator();
            while (enumerator.MoveNext())
            {
                line = enumerator.Current;
                if ((numberOfRowsToRead == -1) || rowline < numberOfRowsToRead)
                {
                    if (linesForGuessType.Count < guessRows || (header && rowline == 0))
                    {
                        var spl = line.Split(separator);
                        if (header && rowline == 0)
                        {
                            if (columnNames == null)
                            {
                                columnNames = spl;
                            }
                        }
                        else
                        {
                            linesForGuessType.Add(spl);
                            numberOfColumns = Math.Max(numberOfColumns, spl.Length);
                        }
                    }
                }
                ++rowline;
                if (rowline == guessRows || guessRows == 0)
                {
                    break;
                }
            }

            if (rowline == 0)
            {
                throw new FormatException(Strings.EmptyFile);
            }

            columns = new List<DataFrameColumn>(numberOfColumns);
            // Guesses types or looks up dataTypes and adds columns.
            for (int i = 0; i < numberOfColumns; ++i)
            {
                Type kind = dataTypes == null ? GuessKind(i, linesForGuessType) : dataTypes[i];
                columns.Add(CreateColumn(kind, columnNames, i));
            }

            DataFrame ret = new DataFrame(columns);
            line = null;

            // Fill values.
            enumerator.Reset();
            rowline = 0;
            while (enumerator.MoveNext() && (numberOfRowsToRead == -1 || rowline < numberOfRowsToRead))
            {
                line = enumerator.Current;
                var spl = line.Split(separator);
                if (header && rowline == 0)
                {
                    // Skips.
                }
                else
                {
                    ret.Append(spl, inPlace: true);
                }
                ++rowline;
            }

            if (addIndexColumn)
            {
                PrimitiveDataFrameColumn<int> indexColumn = new PrimitiveDataFrameColumn<int>("IndexColumn", columns[0].Length);
                for (int i = 0; i < columns[0].Length; i++)
                {
                    indexColumn[i] = i;
                }
                columns.Insert(0, indexColumn);
            }
            return ret;
        }

        private class CsvLines : IEnumerable<string>
        {
            private CsvLineEnumerator enumerator;
            public CsvLines(CsvLineEnumerator csvLineEnumerator)
            {
                enumerator = csvLineEnumerator;
            }

            public IEnumerator<string> GetEnumerator() => enumerator;

            IEnumerator IEnumerable.GetEnumerator() => enumerator;
        }

        private class CsvLineEnumerator : IEnumerator<string>
        {
            private StreamReader streamReader;
            private string currentLine;
            private long streamStartPosition;
            public CsvLineEnumerator(StreamReader csvStream)
            {
                streamStartPosition = csvStream.BaseStream.Position;
                streamReader = csvStream;
                currentLine = null;
            }

            public string Current => currentLine;

            object IEnumerator.Current => currentLine;

            public void Dispose()
            {
                throw new NotImplementedException();
            }

            public bool MoveNext()
            {
                currentLine = streamReader.ReadLine();
                return currentLine != null;
            }

            public void Reset()
            {
                streamReader.DiscardBufferedData();
                streamReader.BaseStream.Seek(streamStartPosition, SeekOrigin.Begin);
            }
        }

        /// <summary>
        /// Reads CSV data passed in as a string into a DataFrame.
        /// </summary>
        /// <param name="csvString">csv data passed in as a string</param>
        /// <param name="separator">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="columnNames">column names (can be empty)</param>
        /// <param name="dataTypes">column types (can be empty)</param>
        /// <param name="numberOfRowsToRead">number of rows to read not including the header(if present)</param>
        /// <param name="guessRows">number of rows used to guess types</param>
        /// <param name="addIndexColumn">add one column with the row index</param>
        /// <returns><see cref="DataFrame"/></returns>
        public static DataFrame LoadCsvFromString(string csvString,
                                char separator = ',', bool header = true,
                                string[] columnNames = null, Type[] dataTypes = null,
                                long numberOfRowsToRead = -1, int guessRows = 10, bool addIndexColumn = false)
        {
            string[] lines = csvString.Split(new[] { Environment.NewLine }, StringSplitOptions.None);
            return ReadCsvLinesIntoDataFrame(lines, separator, header, columnNames, dataTypes, numberOfRowsToRead, guessRows, addIndexColumn);
        }

        /// <summary>
        /// Reads a seekable stream of CSV data into a DataFrame.
        /// </summary>
        /// <param name="csvStream">stream of CSV data to be read in</param>
        /// <param name="separator">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="columnNames">column names (can be empty)</param>
        /// <param name="dataTypes">column types (can be empty)</param>
        /// <param name="numberOfRowsToRead">number of rows to read not including the header(if present)</param>
        /// <param name="guessRows">number of rows used to guess types</param>
        /// <param name="addIndexColumn">add one column with the row index</param>
        /// <param name="encoding">The character encoding. Defaults to UTF8 if not specified</param>
        /// <returns><see cref="DataFrame"/></returns>
        public static DataFrame LoadCsv(Stream csvStream,
                                char separator = ',', bool header = true,
                                string[] columnNames = null, Type[] dataTypes = null,
                                long numberOfRowsToRead = -1, int guessRows = 10, bool addIndexColumn = false,
                                Encoding encoding = null)
        {
            if (!csvStream.CanSeek)
            {
                throw new ArgumentException(Strings.NonSeekableStream, nameof(csvStream));
            }

            if (dataTypes == null && guessRows <= 0)
            {
                throw new ArgumentException(string.Format(Strings.ExpectedEitherGuessRowsOrDataTypes, nameof(guessRows), nameof(dataTypes)));
            }

            using (var streamReader = new StreamReader(csvStream, encoding ?? Encoding.UTF8, detectEncodingFromByteOrderMarks: true, DefaultStreamReaderBufferSize, leaveOpen: true))
            {
                CsvLineEnumerator linesEnumerator = new CsvLineEnumerator(streamReader);
                IEnumerable<string> lines = new CsvLines(linesEnumerator);
                return ReadCsvLinesIntoDataFrame(lines, separator, header, columnNames, dataTypes, numberOfRowsToRead, guessRows, addIndexColumn);
            }
        }

        /// <summary>
        /// Writes a DataFrame into a CSV.
        /// </summary>
        /// <param name="dataFrame"><see cref="DataFrame"/></param>
        /// <param name="path">CSV file path</param>
        /// <param name="separator">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="encoding">The character encoding. Defaults to UTF8 if not specified</param>
        /// <param name="cultureInfo">culture info for formatting values</param>
        public static void WriteCsv(DataFrame dataFrame, string path,
                                   char separator = ',', bool header = true,
                                   Encoding encoding = null, CultureInfo cultureInfo = null)
        {
            using (FileStream csvStream = new FileStream(path, FileMode.Create))
            {
                WriteCsv(dataFrame: dataFrame, csvStream: csvStream,
                           separator: separator, header: header,
                           encoding: encoding, cultureInfo: cultureInfo);
            }
        }

        /// <summary>
        /// Writes a DataFrame into a CSV.
        /// </summary>
        /// <param name="dataFrame"><see cref="DataFrame"/></param>
        /// <param name="csvStream">stream of CSV data to be write out</param>
        /// <param name="separator">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="encoding">the character encoding. Defaults to UTF8 if not specified</param>
        /// <param name="cultureInfo">culture info for formatting values</param>
        public static void WriteCsv(DataFrame dataFrame, Stream csvStream,
                           char separator = ',', bool header = true,
                           Encoding encoding = null, CultureInfo cultureInfo = null)
        {
            if (cultureInfo is null)
            {
                cultureInfo = CultureInfo.CurrentCulture;
            }

            if (cultureInfo.NumberFormat.NumberDecimalSeparator.Equals(separator.ToString()))
            {
                throw new ArgumentException("Decimal separator cannot match the column separator");
            }

            if (encoding is null)
            {
                encoding = Encoding.ASCII;
            }

            using (StreamWriter csvFile = new StreamWriter(csvStream, encoding, bufferSize: DefaultStreamReaderBufferSize, leaveOpen: true))
            {
                if (dataFrame != null)
                {
                    var columnNames = dataFrame.Columns.GetColumnNames();

                    if (header)
                    {
                        var headerColumns = string.Join(separator.ToString(), columnNames);
                        csvFile.WriteLine(headerColumns);
                    }

                    var record = new StringBuilder();

                    foreach (var row in dataFrame.Rows)
                    {
                        bool firstRow = true;
                        foreach (var cell in row)
                        {
                            if (!firstRow)
                            {
                                record.Append(separator);
                            }
                            else
                            {
                                firstRow = false;
                            }

                            Type t = cell?.GetType();

                            if (t == typeof(bool))
                            {
                                record.AppendFormat(cultureInfo, "{0}", cell);
                                continue;
                            }

                            if (t == typeof(float))
                            {
                                record.AppendFormat(cultureInfo, "{0:G9}", cell);
                                continue;
                            }

                            if (t == typeof(double))
                            {
                                record.AppendFormat(cultureInfo, "{0:G17}", cell);
                                continue;
                            }

                            if (t == typeof(decimal))
                            {
                                record.AppendFormat(cultureInfo, "{0:G31}", cell);
                                continue;
                            }

                            record.Append(cell);
                        }

                        csvFile.WriteLine(record);

                        record.Clear();
                    }
                }
            }
        }
    }
}
