// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Common;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Data.SQLite;
using System.Data.SQLite.EF6;
using Xunit;
using Microsoft.ML.TestFramework.Attributes;
using System.Threading;
using Microsoft.ML.Data;
using System.Threading.Tasks;
using Xunit.Abstractions;
using Microsoft.ML.TestFramework;

namespace Microsoft.Data.Analysis.Tests
{
    public class DataFrameIOTests : BaseTestClass
    {
        public DataFrameIOTests(ITestOutputHelper output) : base(output, true)
        {
        }

        internal static void VerifyColumnTypes(DataFrame df, bool testArrowStringColumn = false)
        {
            foreach (DataFrameColumn column in df.Columns)
            {
                Type dataType = column.DataType;
                if (dataType == typeof(bool))
                {
                    Assert.IsType<BooleanDataFrameColumn>(column);

                }
                else if (dataType == typeof(decimal))
                {
                    Assert.IsType<DecimalDataFrameColumn>(column);

                }
                else if (dataType == typeof(byte))
                {
                    Assert.IsType<ByteDataFrameColumn>(column);

                }
                else if (dataType == typeof(char))
                {
                    Assert.IsType<CharDataFrameColumn>(column);

                }
                else if (dataType == typeof(double))
                {
                    Assert.IsType<DoubleDataFrameColumn>(column);

                }
                else if (dataType == typeof(float))
                {
                    Assert.IsType<SingleDataFrameColumn>(column);

                }
                else if (dataType == typeof(int))
                {
                    Assert.IsType<Int32DataFrameColumn>(column);

                }
                else if (dataType == typeof(long))
                {

                    Assert.IsType<Int64DataFrameColumn>(column);
                }
                else if (dataType == typeof(sbyte))
                {
                    Assert.IsType<SByteDataFrameColumn>(column);

                }
                else if (dataType == typeof(short))
                {
                    Assert.IsType<Int16DataFrameColumn>(column);

                }
                else if (dataType == typeof(uint))
                {
                    Assert.IsType<UInt32DataFrameColumn>(column);

                }
                else if (dataType == typeof(ulong))
                {

                    Assert.IsType<UInt64DataFrameColumn>(column);
                }
                else if (dataType == typeof(ushort))
                {
                    Assert.IsType<UInt16DataFrameColumn>(column);

                }
                else if (dataType == typeof(string))
                {
                    if (!testArrowStringColumn)
                    {
                        Assert.IsType<StringDataFrameColumn>(column);
                    }
                    else
                    {
                        Assert.IsType<ArrowStringDataFrameColumn>(column);
                    }
                }
                else if (dataType == typeof(DateTime))
                {
                    Assert.IsType<DateTimeDataFrameColumn>(column);
                }
                else
                {
                    throw new NotImplementedException("Unit test has to be updated");
                }
            }
        }

        private static Stream GetStream(string streamData)
        {
            return new MemoryStream(Encoding.Default.GetBytes(streamData));
        }

        [Fact]
        public void TestReadCsvWithHeaderCultureInfoAndColumnTypeAutoGuess()
        {
            //see https://github.com/dotnet/machinelearning/issues/7240

            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture; // or en-US

            string csv =
@"""Col1"",""Col2"",""Col3"",""Col4""
""v1.1"",""5/7/2017"",""v3.1"",""v4.1""
"""","""",""v3.2"",""v4.2""
";

            var dataFrame = DataFrame.LoadCsvFromString(csv, separator: ',', header: true,
                dataTypes: null, // guess the column types
                renameDuplicatedColumns: true, // try to rename the duplicated columns, if any
                cultureInfo: CultureInfo.InvariantCulture);
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void TestReadCsvWithHeader(bool useQuotes)
        {
            string CMT = useQuotes ? @"""C,MT""" : "CMT";
            string verifyCMT = useQuotes ? "C,MT" : "CMT";
            string data = @$"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
{CMT},1,1,1271,3.8,CRD,17.5
{CMT},1,1,474,1.5,CRD,8
{CMT},1,1,637,1.4,CRD,8.5
{CMT},1,1,181,0.6,CSH,4.5";

            void RegularTest(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal(verifyCMT, df.Columns["vendor_id"][3]);
                VerifyColumnTypes(df);
            }
            DataFrame df = DataFrame.LoadCsv(GetStream(data));
            RegularTest(df);
            DataFrame csvDf = DataFrame.LoadCsvFromString(data);
            RegularTest(csvDf);

            void ReducedRowsTest(DataFrame reducedRows)
            {
                Assert.Equal(3, reducedRows.Rows.Count);
                Assert.Equal(7, reducedRows.Columns.Count);
                Assert.Equal(verifyCMT, reducedRows.Columns["vendor_id"][2]);
                VerifyColumnTypes(df);
            }
            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), numberOfRowsToRead: 3);
            ReducedRowsTest(reducedRows);
            csvDf = DataFrame.LoadCsvFromString(data, numberOfRowsToRead: 3);
            ReducedRowsTest(csvDf);
        }

        [Fact]
        public void TestReadCsvWithHeaderCultureInfoAndSeparator()
        {
            string data = @$"vendor_id;rate_code;passenger_count;trip_time_in_secs;trip_distance;payment_type;fare_amount
CMT;1;1;1271;3,8;CRD;17,5
CMT;1;1;474;1,5;CRD;8
CMT;1;1;637;1,4;CRD;8,5
CMT;1;1;181;0,6;CSH;4,5";

            void RegularTest(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);

                Assert.Equal(3.8f, (float)df["trip_distance"][0]);
                Assert.Equal(17.5f, (float)df["fare_amount"][0]);

                Assert.Equal(1.5f, (float)df["trip_distance"][1]);
                Assert.Equal(8f, (float)df["fare_amount"][1]);

                Assert.Equal(1.4f, (float)df["trip_distance"][2]);
                Assert.Equal(8.5f, (float)df["fare_amount"][2]);

                VerifyColumnTypes(df);
            }

            // de-DE has ',' as decimal separator
            var cultureInfo = new CultureInfo("de-DE");
            DataFrame df = DataFrame.LoadCsv(GetStream(data), separator: ';', cultureInfo: cultureInfo);

            RegularTest(df);

            DataFrame csvDf = DataFrame.LoadCsvFromString(data, separator: ';', cultureInfo: cultureInfo);
            RegularTest(csvDf);
        }

        [Fact]
        public void TestReadCsvWithHeaderAndDuplicatedColumns_WithoutRenaming()
        {

            string data = @$"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD,CRD,17.5
CMT,1,1,474,1.5,CRD,CRD,8
CMT,1,1,637,1.4,CRD,CRD,8.5
CMT,1,1,181,0.6,CSH,CSH,4.5";

            Assert.Throws<System.ArgumentException>(() => DataFrame.LoadCsv(GetStream(data)));
        }

        [Fact]
        public void TestReadCsvWithHeaderAndDuplicatedColumns_WithDuplicateColumnRenaming()
        {

            string data = @$"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,payment_type,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD,CRD_1,Test,17.5
CMT,1,1,474,1.5,CRD,CRD,Test,8
CMT,1,1,637,1.4,CRD,CRD,Test,8.5
CMT,1,1,181,0.6,CSH,CSH,Test,4.5";

            DataFrame df = DataFrame.LoadCsv(GetStream(data), renameDuplicatedColumns: true);

            Assert.Equal(4, df.Rows.Count);
            Assert.Equal(9, df.Columns.Count);
            Assert.Equal("CMT", df.Columns["vendor_id"][3]);

            Assert.Equal("payment_type", df.Columns[5].Name);
            Assert.Equal("payment_type.1", df.Columns[6].Name);
            Assert.Equal("payment_type.2", df.Columns[7].Name);

            Assert.Equal("CRD", df.Columns["payment_type"][0]);
            Assert.Equal("CRD_1", df.Columns["payment_type.1"][0]);
            Assert.Equal("Test", df.Columns["payment_type.2"][0]);

            VerifyColumnTypes(df);
        }

        [Fact]
        public void TestReadCsvSplitAcrossMultipleLines()
        {
            string CMT = @"""C
MT""";
            string verifyCMT = @"C
MT";
            string data = @$"{CMT},1,1,1271,3.8,CRD,17.5
{CMT},1,1,474,1.5,CRD,8
{CMT},1,1,637,1.4,CRD,8.5
{CMT},1,1,181,0.6,CSH,4.5";

            void RegularTest(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal(verifyCMT, df.Columns["Column0"][3]);
                VerifyColumnTypes(df);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data), header: false);
            RegularTest(df);
            DataFrame csvDf = DataFrame.LoadCsvFromString(data, header: false);
            RegularTest(csvDf);

            void ReducedRowsTest(DataFrame reducedRows)
            {
                Assert.Equal(3, reducedRows.Rows.Count);
                Assert.Equal(7, reducedRows.Columns.Count);
                Assert.Equal(verifyCMT, reducedRows.Columns["Column0"][2]);
                VerifyColumnTypes(df);
            }

            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), header: false, numberOfRowsToRead: 3);
            ReducedRowsTest(reducedRows);
            csvDf = DataFrame.LoadCsvFromString(data, header: false, numberOfRowsToRead: 3);
            ReducedRowsTest(csvDf);
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void TestLoadCsvNoHeader(bool useQuotes)
        {
            string CMT = useQuotes ? @"""C,MT""" : "CMT";
            string verifyCMT = useQuotes ? "C,MT" : "CMT";
            string data = @$"{CMT},1,1,1271,3.8,CRD,17.5
{CMT},1,1,474,1.5,CRD,8
{CMT},1,1,637,1.4,CRD,8.5
{CMT},1,1,181,0.6,CSH,4.5";

            void RegularTest(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal(verifyCMT, df.Columns["Column0"][3]);
                VerifyColumnTypes(df);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data), header: false);
            RegularTest(df);
            DataFrame csvDf = DataFrame.LoadCsvFromString(data, header: false);
            RegularTest(csvDf);

            void ReducedRowsTest(DataFrame reducedRows)
            {
                Assert.Equal(3, reducedRows.Rows.Count);
                Assert.Equal(7, reducedRows.Columns.Count);
                Assert.Equal(verifyCMT, reducedRows.Columns["Column0"][2]);
                VerifyColumnTypes(df);
            }

            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), header: false, numberOfRowsToRead: 3);
            ReducedRowsTest(reducedRows);
            csvDf = DataFrame.LoadCsvFromString(data, header: false, numberOfRowsToRead: 3);
            ReducedRowsTest(csvDf);
        }

        void VerifyDataFrameWithNamedColumnsAndDataTypes(DataFrame df, bool verifyColumnDataType, bool verifyNames)
        {
            Assert.Equal(4, df.Rows.Count);
            Assert.Equal(7, df.Columns.Count);

            if (verifyColumnDataType)
            {
                Assert.True(typeof(string) == df.Columns[0].DataType);
                Assert.True(typeof(short) == df.Columns[1].DataType);
                Assert.True(typeof(int) == df.Columns[2].DataType);
                Assert.True(typeof(long) == df.Columns[3].DataType);
                Assert.True(typeof(float) == df.Columns[4].DataType);
                Assert.True(typeof(string) == df.Columns[5].DataType);
                Assert.True(typeof(double) == df.Columns[6].DataType);
            }

            if (verifyNames)
            {
                Assert.Equal("vendor_id", df.Columns[0].Name);
                Assert.Equal("rate_code", df.Columns[1].Name);
                Assert.Equal("passenger_count", df.Columns[2].Name);
                Assert.Equal("trip_time_in_secs", df.Columns[3].Name);
                Assert.Equal("trip_distance", df.Columns[4].Name);
                Assert.Equal("payment_type", df.Columns[5].Name);
                Assert.Equal("fare_amount", df.Columns[6].Name);
            }

            VerifyColumnTypes(df);

            foreach (var column in df.Columns)
            {
                Assert.Equal(0, column.NullCount);
            }
        }

        [Theory]
        [InlineData(true, 0)]
        [InlineData(false, 0)]
        [InlineData(true, 10)]
        [InlineData(false, 10)]
        public void TestLoadCsvWithTypesAndGuessRows(bool header, int guessRows)
        {
            /* Tests this matrix
             * 
                header	GuessRows	DataTypes	
                True	0	        NotNull	    
                False 	0	        NotNull	    
                True	10	        NotNull	    
                False 	10	        NotNull	    
                True	0	        Null  -----> Throws an exception
                False 	0	        Null  -----> Throws an exception
                True	10	        Null	    
                False 	10	        Null	    
             * 
             */
            string headerLine = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
";
            string dataLines =
@"CMT,1,1,1271,3.8,CRD,17.5
CMT,1,1,474,1.5,CRD,8
CMT,1,1,637,1.4,CRD,8.5
CMT,1,1,181,0.6,CSH,4.5";


            string data = header ? headerLine + dataLines : dataLines;
            DataFrame df = DataFrame.LoadCsv(GetStream(data),
                                             header: header,
                                             guessRows: guessRows,
                                             dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long), typeof(float), typeof(string), typeof(double) }
                                             );
            VerifyDataFrameWithNamedColumnsAndDataTypes(df, verifyColumnDataType: true, verifyNames: header);
            DataFrame csvDf = DataFrame.LoadCsvFromString(data,
                                             header: header,
                                             guessRows: guessRows,
                                             dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long), typeof(float), typeof(string), typeof(double) }
                                             );
            VerifyDataFrameWithNamedColumnsAndDataTypes(csvDf, verifyColumnDataType: true, verifyNames: header);

            if (guessRows == 10)
            {
                df = DataFrame.LoadCsv(GetStream(data),
                                                 header: header,
                                                 guessRows: guessRows
                                                 );
                VerifyDataFrameWithNamedColumnsAndDataTypes(df, verifyColumnDataType: false, verifyNames: header);
                csvDf = DataFrame.LoadCsvFromString(data,
                                          header: header,
                                          guessRows: guessRows);
                VerifyDataFrameWithNamedColumnsAndDataTypes(csvDf, verifyColumnDataType: false, verifyNames: header);
            }
            else
            {
                Assert.ThrowsAny<ArgumentException>(() => DataFrame.LoadCsv(GetStream(data),
                                                 header: header,
                                                 guessRows: guessRows
                                                 ));
                Assert.ThrowsAny<ArgumentException>(() => DataFrame.LoadCsvFromString(data,
                                                 header: header,
                                                 guessRows: guessRows
                                                 ));
            }
        }

        [Fact]
        public void TestReadCsvWithTypes()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD,17.5
CMT,1,1,474,1.5,CRD,8
CMT,1,1,637,1.4,CRD,8.5
,,,,,,
CMT,1,1,181,0.6,CSH,4.5";

            void Verify(DataFrame df)
            {
                Assert.Equal(5, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);

                Assert.True(typeof(string) == df.Columns[0].DataType);
                Assert.True(typeof(short) == df.Columns[1].DataType);
                Assert.True(typeof(int) == df.Columns[2].DataType);
                Assert.True(typeof(long) == df.Columns[3].DataType);
                Assert.True(typeof(float) == df.Columns[4].DataType);
                Assert.True(typeof(string) == df.Columns[5].DataType);
                Assert.True(typeof(double) == df.Columns[6].DataType);

                Assert.Equal("vendor_id", df.Columns[0].Name);
                Assert.Equal("rate_code", df.Columns[1].Name);
                Assert.Equal("passenger_count", df.Columns[2].Name);
                Assert.Equal("trip_time_in_secs", df.Columns[3].Name);
                Assert.Equal("trip_distance", df.Columns[4].Name);
                Assert.Equal("payment_type", df.Columns[5].Name);
                Assert.Equal("fare_amount", df.Columns[6].Name);
                VerifyColumnTypes(df);

                foreach (var column in df.Columns)
                {
                    if (column.DataType != typeof(string))
                    {
                        Assert.Equal(1, column.NullCount);
                    }
                    else
                    {
                        Assert.Equal(0, column.NullCount);
                    }
                }
                var nullRow = df.Rows[3];
                Assert.Equal("", nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);
                Assert.Null(nullRow[4]);
                Assert.Equal("", nullRow[5]);
                Assert.Null(nullRow[6]);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data), dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long), typeof(float), typeof(string), typeof(double) });
            Verify(df);
            df = DataFrame.LoadCsvFromString(data, dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long), typeof(float), typeof(string), typeof(double) });
            Verify(df);
        }

        [Fact]
        public void TestLoadCsvWithTypesDateTime()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount,date
CMT,1,1,1271,3.8,CRD,17.5,1-june-2020
CMT,1,1,474,1.5,CRD,8,2-june-2020
CMT,1,1,637,1.4,CRD,8.5,3-june-2020
,,,,,,,
CMT,1,1,181,0.6,CSH,4.5,4-june-2020";

            void Verify(DataFrame df, bool verifyDataTypes)
            {
                Assert.Equal(5, df.Rows.Count);
                Assert.Equal(8, df.Columns.Count);

                if (verifyDataTypes)
                {
                    Assert.True(typeof(string) == df.Columns[0].DataType);
                    Assert.True(typeof(short) == df.Columns[1].DataType);
                    Assert.True(typeof(int) == df.Columns[2].DataType);
                    Assert.True(typeof(long) == df.Columns[3].DataType);
                    Assert.True(typeof(float) == df.Columns[4].DataType);
                    Assert.True(typeof(string) == df.Columns[5].DataType);
                    Assert.True(typeof(double) == df.Columns[6].DataType);
                    Assert.True(typeof(DateTime) == df.Columns[7].DataType);
                }

                Assert.Equal("vendor_id", df.Columns[0].Name);
                Assert.Equal("rate_code", df.Columns[1].Name);
                Assert.Equal("passenger_count", df.Columns[2].Name);
                Assert.Equal("trip_time_in_secs", df.Columns[3].Name);
                Assert.Equal("trip_distance", df.Columns[4].Name);
                Assert.Equal("payment_type", df.Columns[5].Name);
                Assert.Equal("fare_amount", df.Columns[6].Name);
                Assert.Equal("date", df.Columns[7].Name);
                VerifyColumnTypes(df);

                foreach (var column in df.Columns)
                {
                    if (column.DataType != typeof(string))
                    {
                        Assert.Equal(1, column.NullCount);
                    }
                    else
                    {
                        Assert.Equal(0, column.NullCount);
                    }
                }
                var nullRow = df.Rows[3];
                Assert.Equal("", nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);
                Assert.Null(nullRow[4]);
                Assert.Equal("", nullRow[5]);
                Assert.Null(nullRow[6]);
                Assert.Null(nullRow[7]);

                var dateTimeColumn = df.Columns["date"];
                Assert.Equal(new DateTime(2020, 06, 01), dateTimeColumn[0]);
                Assert.Equal(new DateTime(2020, 06, 02), dateTimeColumn[1]);
                Assert.Equal(new DateTime(2020, 06, 03), dateTimeColumn[2]);
                Assert.Null(dateTimeColumn[3]);
                Assert.Equal(new DateTime(2020, 06, 04), dateTimeColumn[4]);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data), dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long), typeof(float), typeof(string), typeof(double), typeof(DateTime) });
            Verify(df, true);
            df = DataFrame.LoadCsvFromString(data, dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long), typeof(float), typeof(string), typeof(double), typeof(DateTime) });
            Verify(df, true);
            // Verify without dataTypes
            df = DataFrame.LoadCsv(GetStream(data));
            Verify(df, false);
            df = DataFrame.LoadCsvFromString(data);
            Verify(df, false);
        }

        [Fact]
        public void TestLoadCsvWithPipeSeparator()
        {
            string data = @"vendor_id|rate_code|passenger_count|trip_time_in_secs|trip_distance|payment_type|fare_amount
CMT|1|1|1271|3.8|CRD|17.5
CMT|1|1|474|1.5|CRD|8
CMT|1|1|637|1.4|CRD|8.5
||||||
CMT|1|1|181|0.6|CSH|4.5";

            void Verify(DataFrame df)
            {
                Assert.Equal(5, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal("CMT", df.Columns["vendor_id"][4]);
                VerifyColumnTypes(df);

                DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), separator: '|', numberOfRowsToRead: 3);
                Assert.Equal(3, reducedRows.Rows.Count);
                Assert.Equal(7, reducedRows.Columns.Count);
                Assert.Equal("CMT", reducedRows.Columns["vendor_id"][2]);
                VerifyColumnTypes(df);

                var nullRow = df.Rows[3];
                Assert.Equal("", nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);
                Assert.Null(nullRow[4]);
                Assert.Equal("", nullRow[5]);
                Assert.Null(nullRow[6]);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data), separator: '|');
            Verify(df);
            df = DataFrame.LoadCsvFromString(data, separator: '|');
            Verify(df);
        }

        [Fact]
        public void TestLoadCsvWithSemicolonSeparator()
        {
            string data = @"vendor_id;rate_code;passenger_count;trip_time_in_secs;trip_distance;payment_type;fare_amount
CMT;1;1;1271;3.8;CRD;17.5
CMT;1;1;474;1.5;CRD;8
CMT;1;1;637;1.4;CRD;8.5
;;;;;;
CMT;1;1;181;0.6;CSH;4.5";

            void Verify(DataFrame df)
            {
                Assert.Equal(5, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal("CMT", df.Columns["vendor_id"][4]);
                VerifyColumnTypes(df);

                DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), separator: ';', numberOfRowsToRead: 3);
                Assert.Equal(3, reducedRows.Rows.Count);
                Assert.Equal(7, reducedRows.Columns.Count);
                Assert.Equal("CMT", reducedRows.Columns["vendor_id"][2]);
                VerifyColumnTypes(df);

                var nullRow = df.Rows[3];
                Assert.Equal("", nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);
                Assert.Null(nullRow[4]);
                Assert.Equal("", nullRow[5]);
                Assert.Null(nullRow[6]);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data), separator: ';');
            Verify(df);
            df = DataFrame.LoadCsvFromString(data, separator: ';');
            Verify(df);
        }

        [Fact]
        public void TestLoadCsvWithExtraColumnInHeader()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount,extra
CMT,1,1,1271,3.8,CRD,17.5
CMT,1,1,474,1.5,CRD,8
CMT,1,1,637,1.4,CRD,8.5
CMT,1,1,181,0.6,CSH,4.5";

            void Verify(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal("CMT", df.Columns["vendor_id"][3]);
                VerifyColumnTypes(df);

                DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), numberOfRowsToRead: 3);
                Assert.Equal(3, reducedRows.Rows.Count);
                Assert.Equal(7, reducedRows.Columns.Count);
                Assert.Equal("CMT", reducedRows.Columns["vendor_id"][2]);
                VerifyColumnTypes(df);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data));
            Verify(df);
            df = DataFrame.LoadCsvFromString(data);
            Verify(df);
        }

        [Fact]
        public void TestLoadCsvWithMultipleEmptyColumnNameInHeaderWithoutGivenColumn()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,,,,
CMT,1,1,1271,3.8,CRD,17.5,0
CMT,1,1,474,1.5,CRD,8,0
CMT,1,1,637,1.4,CRD,8.5,0
CMT,1,1,181,0.6,CSH,4.5,0";

            var df = DataFrame.LoadCsvFromString(data, header: true);
            var columnName = df.Columns.Select(c => c.Name);

            Assert.Equal(columnName, new[] { "vendor_id", "rate_code", "passenger_count", "trip_time_in_secs", "Column4", "Column5", "Column6", "Column7" });
        }

        [Fact]
        public void TestLoadCsvWithMultipleEmptyColumnNameInHeaderWithGivenColumn()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,,,,
CMT,1,1,1271,3.8,CRD,17.5,0
CMT,1,1,474,1.5,CRD,8,0
CMT,1,1,637,1.4,CRD,8.5,0
CMT,1,1,181,0.6,CSH,4.5,0";

            var df = DataFrame.LoadCsvFromString(data, header: true, columnNames: new[] { "vendor_id", "rate_code", "passenger_count", "trip_time_in_secs", "Column0", "Column1", "Column2", "Column3" });
            var columnName = df.Columns.Select(c => c.Name);

            Assert.Equal(columnName, new[] { "vendor_id", "rate_code", "passenger_count", "trip_time_in_secs", "Column0", "Column1", "Column2", "Column3" });
        }

        [Fact]
        public void TestReadCsvWithRepeatColumnNameInHeader()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,Column,Column,,
CMT,1,1,1271,3.8,CRD,17.5,0
CMT,1,1,474,1.5,CRD,8,0
CMT,1,1,637,1.4,CRD,8.5,0
CMT,1,1,181,0.6,CSH,4.5,0";

            var exp = Assert.ThrowsAny<ArgumentException>(() => DataFrame.LoadCsvFromString(data, header: true));
            // .NET Core and .NET Framework return the parameter name slightly different. Using regex so the test will work for both frameworks.
            Assert.Matches(@"DataFrame already contains a column called Column( \(Parameter 'column'\)|\r\nParameter name: column)", exp.Message);
        }

        [Fact]
        public void TestLoadCsvWithAddIndexColumn()
        {
            var dataFrame = DataFrame.LoadCsvFromString("11\r\n22\r\n33", header: false, addIndexColumn: true);

            Assert.Equal(2, dataFrame.Columns.Count);
            Assert.Equal("IndexColumn", dataFrame.Columns[0].Name);
            Assert.Equal(3, dataFrame.Columns[0].Length);

            for (long i = 0; i < dataFrame.Columns[0].Length; i++)
                Assert.Equal(i, dataFrame.Columns[0][i]);
        }

        [Fact]
        public void TestLoadCsvWithExtraColumnInRow()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD,17.5,0
CMT,1,1,474,1.5,CRD,8,0
CMT,1,1,637,1.4,CRD,8.5,0
CMT,1,1,181,0.6,CSH,4.5,0";

            Assert.Throws<IndexOutOfRangeException>(() => DataFrame.LoadCsv(GetStream(data)));
            Assert.Throws<IndexOutOfRangeException>(() => DataFrame.LoadCsvFromString(data));
        }

        [Fact]
        public void TestLoadCsvWithLessColumnsInRow()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD
CMT,1,1,474,1.5,CRD
CMT,1,1,637,1.4,CRD
CMT,1,1,181,0.6,CSH";

            void Verify(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(6, df.Columns.Count);
                Assert.Equal("CMT", df.Columns["vendor_id"][3]);
                VerifyColumnTypes(df);

                DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), numberOfRowsToRead: 3);
                Assert.Equal(3, reducedRows.Rows.Count);
                Assert.Equal(6, reducedRows.Columns.Count);
                Assert.Equal("CMT", reducedRows.Columns["vendor_id"][2]);
                VerifyColumnTypes(df);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data));
            Verify(df);
            df = DataFrame.LoadCsvFromString(data);
            Verify(df);
        }

        [Fact]
        public void TestLoadCsvWithAllNulls()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs
null,null,null,null
Null,Null,Null,Null
null,null,null,null
Null,Null,Null,Null
null,null,null,null
null,null,null,null";

            void Verify(DataFrame df)
            {
                Assert.Equal(6, df.Rows.Count);
                Assert.Equal(4, df.Columns.Count);

                Assert.True(typeof(string) == df.Columns[0].DataType);
                Assert.True(typeof(string) == df.Columns[1].DataType);
                Assert.True(typeof(string) == df.Columns[2].DataType);
                Assert.True(typeof(string) == df.Columns[3].DataType);

                Assert.Equal("vendor_id", df.Columns[0].Name);
                Assert.Equal("rate_code", df.Columns[1].Name);
                Assert.Equal("passenger_count", df.Columns[2].Name);
                Assert.Equal("trip_time_in_secs", df.Columns[3].Name);
                VerifyColumnTypes(df);

                foreach (var column in df.Columns)
                {
                    Assert.Equal(6, column.NullCount);
                    foreach (var value in column)
                    {
                        Assert.Null(value);
                    }
                }
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data));
            Verify(df);
            df = DataFrame.LoadCsvFromString(data);
            Verify(df);
        }

        [Fact]
        public void TestLoadCsvWithNullsAndDataTypes()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs
null,1,1,1271
CMT,Null,1,474
CMT,1,null,637
Null,,,
,,,
CMT,1,1,null";

            void Verify(DataFrame df)
            {
                Assert.Equal(6, df.Rows.Count);
                Assert.Equal(4, df.Columns.Count);

                Assert.True(typeof(string) == df.Columns[0].DataType);
                Assert.True(typeof(short) == df.Columns[1].DataType);
                Assert.True(typeof(int) == df.Columns[2].DataType);
                Assert.True(typeof(long) == df.Columns[3].DataType);

                Assert.Equal("vendor_id", df.Columns[0].Name);
                Assert.Equal("rate_code", df.Columns[1].Name);
                Assert.Equal("passenger_count", df.Columns[2].Name);
                Assert.Equal("trip_time_in_secs", df.Columns[3].Name);
                VerifyColumnTypes(df);

                foreach (var column in df.Columns)
                {
                    if (column.DataType != typeof(string))
                    {
                        Assert.Equal(3, column.NullCount);
                    }
                    else
                    {
                        Assert.Equal(2, column.NullCount);
                    }
                }
                var nullRow = df.Rows[3];
                Assert.Null(nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);

                nullRow = df.Rows[4];
                Assert.Equal("", nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);

                Assert.Null(df[0, 0]);
                Assert.Null(df[1, 1]);
                Assert.Null(df[2, 2]);
                Assert.Null(df[5, 3]);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data), dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long) });
            Verify(df);
            df = DataFrame.LoadCsvFromString(data, dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long) });
            Verify(df);
        }

        [Fact]
        public void TestLoadCsvWithNulls()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs
null,1,1,1271
CMT,Null,1,474
CMT,1,null,637
Null,,,
,,,
CMT,1,1,null";

            void Verify(DataFrame df)
            {
                Assert.Equal(6, df.Rows.Count);
                Assert.Equal(4, df.Columns.Count);

                Assert.True(typeof(string) == df.Columns[0].DataType);
                Assert.True(typeof(float) == df.Columns[1].DataType);
                Assert.True(typeof(float) == df.Columns[2].DataType);
                Assert.True(typeof(float) == df.Columns[3].DataType);

                Assert.Equal("vendor_id", df.Columns[0].Name);
                Assert.Equal("rate_code", df.Columns[1].Name);
                Assert.Equal("passenger_count", df.Columns[2].Name);
                Assert.Equal("trip_time_in_secs", df.Columns[3].Name);
                VerifyColumnTypes(df);

                foreach (var column in df.Columns)
                {
                    if (column.DataType != typeof(string))
                    {
                        Assert.Equal(3, column.NullCount);
                    }
                    else
                    {
                        Assert.Equal(2, column.NullCount);
                    }
                }
                var nullRow = df.Rows[3];
                Assert.Null(nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);

                nullRow = df.Rows[4];
                Assert.Equal("", nullRow[0]);
                Assert.Null(nullRow[1]);
                Assert.Null(nullRow[2]);
                Assert.Null(nullRow[3]);

                Assert.Null(df[0, 0]);
                Assert.Null(df[1, 1]);
                Assert.Null(df[2, 2]);
                Assert.Null(df[5, 3]);
            }

            DataFrame df = DataFrame.LoadCsv(GetStream(data));
            Verify(df);
            df = DataFrame.LoadCsvFromString(data);
            Verify(df);
        }

        [Fact]
        public void TestSaveCsvVBufferColumn()
        {
            var vBuffers = new[]
            {
                new VBuffer<int> (3, new int[] { 1, 2, 3 }),
                new VBuffer<int> (3, new int[] { 2, 3, 4 }),
                new VBuffer<int> (3, new int[] { 3, 4, 5 }),
            };

            var vBufferColumn = new VBufferDataFrameColumn<int>("VBuffer", vBuffers);
            DataFrame dataFrame = new DataFrame(vBufferColumn);

            using MemoryStream csvStream = new MemoryStream();

            DataFrame.SaveCsv(dataFrame, csvStream);

            csvStream.Seek(0, SeekOrigin.Begin);
            DataFrame readIn = DataFrame.LoadCsv(csvStream);

            Assert.Equal(dataFrame.Rows.Count, readIn.Rows.Count);
            Assert.Equal(dataFrame.Columns.Count, readIn.Columns.Count);

            Assert.Equal(typeof(string), readIn.Columns[0].DataType);
            Assert.Equal("(1 2 3)", readIn[0, 0]);
            Assert.Equal("(2 3 4)", readIn[1, 0]);
            Assert.Equal("(3 4 5)", readIn[2, 0]);
        }

        [Fact]
        public void TestSaveCsvWithHeader()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, true);

            DataFrame.SaveCsv(dataFrame, csvStream);

            csvStream.Seek(0, SeekOrigin.Begin);
            DataFrame readIn = DataFrame.LoadCsv(csvStream);

            Assert.Equal(dataFrame.Rows.Count, readIn.Rows.Count);
            Assert.Equal(dataFrame.Columns.Count, readIn.Columns.Count);
            Assert.Equal(1F, readIn[1, 0]);
            Assert.Equal(1F, readIn[1, 1]);
            Assert.Equal(1F, readIn[1, 2]);
            Assert.Equal(1F, readIn[1, 3]);
            Assert.Equal(1F, readIn[1, 4]);
            Assert.Equal(1F, readIn[1, 5]);
            Assert.Equal(1F, readIn[1, 6]);
            Assert.Equal(1F, readIn[1, 7]);
            Assert.Equal(1F, readIn[1, 8]);
            Assert.Equal(1F, readIn[1, 9]);
            Assert.Equal(1F, readIn[1, 10]);
        }

        [Fact]
        public void TestSaveCsvWithCultureInfoRomanianAndSemiColon()
        {
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithNumericColumns(10, true);
            dataFrame[1, 1] = 1.1M;
            dataFrame[1, 2] = 1.2D;
            dataFrame[1, 3] = 1.3F;

            using MemoryStream csvStream = new MemoryStream();
            var cultureInfo = new CultureInfo("ro-RO");
            var separator = ';';
            DataFrame.SaveCsv(dataFrame, csvStream, separator: separator, cultureInfo: cultureInfo);

            csvStream.Seek(0, SeekOrigin.Begin);
            DataFrame readIn = DataFrame.LoadCsv(csvStream, separator: separator);

            Assert.Equal(dataFrame.Rows.Count, readIn.Rows.Count);
            Assert.Equal(dataFrame.Columns.Count, readIn.Columns.Count);
            Assert.Equal(1F, readIn[1, 0]);

            // LoadCsv does not support culture info, therefore decimal point comma (,) is seen as thousand separator and is ignored when read
            Assert.Equal(11F, readIn[1, 1]);
            Assert.Equal(12F, readIn[1, 2]);
            Assert.Equal(129999992F, readIn[1, 3]);

            Assert.Equal(1F, readIn[1, 4]);
            Assert.Equal(1F, readIn[1, 5]);
            Assert.Equal(1F, readIn[1, 6]);
            Assert.Equal(1F, readIn[1, 7]);
            Assert.Equal(1F, readIn[1, 8]);
            Assert.Equal(1F, readIn[1, 9]);
            Assert.Equal(1F, readIn[1, 10]);
        }

        [Fact]
        public void TestSaveCsvWithCultureInfo()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithNumericColumns(10, true);
            dataFrame[1, 1] = 1.1M;
            dataFrame[1, 2] = 1.2D;
            dataFrame[1, 3] = 1.3F;

            var cultureInfo = new CultureInfo("en-US");
            DataFrame.SaveCsv(dataFrame, csvStream, cultureInfo: cultureInfo);

            csvStream.Seek(0, SeekOrigin.Begin);
            DataFrame readIn = DataFrame.LoadCsv(csvStream);

            Assert.Equal(dataFrame.Rows.Count, readIn.Rows.Count);
            Assert.Equal(dataFrame.Columns.Count, readIn.Columns.Count);
            Assert.Equal(1F, readIn[1, 0]);
            Assert.Equal(1.1F, readIn[1, 1]);
            Assert.Equal(1.2F, readIn[1, 2]);
            Assert.Equal(1.3F, readIn[1, 3]);
            Assert.Equal(1F, readIn[1, 4]);
            Assert.Equal(1F, readIn[1, 5]);
            Assert.Equal(1F, readIn[1, 6]);
            Assert.Equal(1F, readIn[1, 7]);
            Assert.Equal(1F, readIn[1, 8]);
            Assert.Equal(1F, readIn[1, 9]);
            Assert.Equal(1F, readIn[1, 10]);
        }

        [Fact]
        public void TestSaveCsvWithCultureInfoRomanianAndComma()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithNumericColumns(10, true);

            var cultureInfo = new CultureInfo("ro-RO");
            var separator = cultureInfo.NumberFormat.NumberDecimalSeparator.First();

            Assert.Throws<ArgumentException>(() => DataFrame.SaveCsv(dataFrame, csvStream, separator: separator, cultureInfo: cultureInfo));
        }

        [Fact]
        public void TestSaveCsvWithNoHeader()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, true);

            DataFrame.SaveCsv(dataFrame, csvStream, header: false);

            csvStream.Seek(0, SeekOrigin.Begin);
            DataFrame readIn = DataFrame.LoadCsv(csvStream, header: false);

            Assert.Equal(dataFrame.Rows.Count, readIn.Rows.Count);
            Assert.Equal(dataFrame.Columns.Count, readIn.Columns.Count);
            Assert.Equal(1F, readIn[1, 0]);
            Assert.Equal(1F, readIn[1, 1]);
            Assert.Equal(1F, readIn[1, 2]);
            Assert.Equal(1F, readIn[1, 3]);
            Assert.Equal(1F, readIn[1, 4]);
            Assert.Equal(1F, readIn[1, 5]);
            Assert.Equal(1F, readIn[1, 6]);
            Assert.Equal(1F, readIn[1, 7]);
            Assert.Equal(1F, readIn[1, 8]);
            Assert.Equal(1F, readIn[1, 9]);
            Assert.Equal(1F, readIn[1, 10]);
        }

        [Fact]
        public void TestSaveCsvWithSemicolonSeparator()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, true);

            var separator = ';';
            DataFrame.SaveCsv(dataFrame, csvStream, separator: separator);

            csvStream.Seek(0, SeekOrigin.Begin);
            DataFrame readIn = DataFrame.LoadCsv(csvStream, separator: separator);

            Assert.Equal(dataFrame.Rows.Count, readIn.Rows.Count);
            Assert.Equal(dataFrame.Columns.Count, readIn.Columns.Count);
            Assert.Equal(1F, readIn[1, 0]);
            Assert.Equal(1F, readIn[1, 1]);
            Assert.Equal(1F, readIn[1, 2]);
            Assert.Equal(1F, readIn[1, 3]);
            Assert.Equal(1F, readIn[1, 4]);
            Assert.Equal(1F, readIn[1, 5]);
            Assert.Equal(1F, readIn[1, 6]);
            Assert.Equal(1F, readIn[1, 7]);
            Assert.Equal(1F, readIn[1, 8]);
            Assert.Equal(1F, readIn[1, 9]);
            Assert.Equal(1F, readIn[1, 10]);
        }

        [Fact]
        public void TestMixedDataTypesInCsv()
        {
            string data = @"vendor_id,empty
null,
1,
true,
Null,
,
CMT,";

            DataFrame df = DataFrame.LoadCsv(GetStream(data));
            Assert.Equal(6, df.Rows.Count);
            Assert.Equal(2, df.Columns.Count);

            Assert.True(typeof(string) == df.Columns[0].DataType);
            Assert.True(typeof(string) == df.Columns[1].DataType);

            Assert.Equal("vendor_id", df.Columns[0].Name);
            Assert.Equal("empty", df.Columns[1].Name);
            VerifyColumnTypes(df);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(0, df.Columns[1].NullCount);

            var nullRow = df.Rows[3];
            Assert.Null(nullRow[0]);

            nullRow = df.Rows[4];
            Assert.Equal("", nullRow[0]);

            Assert.Null(df[0, 0]);
            Assert.Null(df[3, 0]);

            StringDataFrameColumn emptyColumn = (StringDataFrameColumn)df.Columns[1];
            for (long i = 0; i < emptyColumn.Length; i++)
            {
                Assert.Equal("", emptyColumn[i]);
            }
        }

        [Fact]
        public void TestLoadFromEnumerable()
        {
            var (columns, vals) = GetTestData();
            var dataFrame = DataFrame.LoadFrom(vals, columns);
            AssertEqual(dataFrame, columns, vals);
        }

        [Fact]
        public void TestSaveToDataTable()
        {
            var (columns, vals) = GetTestData();
            var dataFrame = DataFrame.LoadFrom(vals, columns);

            using var table = dataFrame.ToTable();

            var resColumns = table.Columns.Cast<DataColumn>().Select(column => (column.ColumnName, column.DataType)).ToArray();
            Assert.Equal(columns, resColumns);

            var resVals = table.Rows.Cast<DataRow>().Select(row => row.ItemArray).ToArray();
            Assert.Equal(vals, resVals);
        }

        [X86X64Fact("The SQLite un-managed code, SQLite.interop, only supports x86/x64 architectures.")]
        public async Task TestSQLite()
        {
            var (columns, vals) = GetTestData();
            var dataFrame = DataFrame.LoadFrom(vals, columns);

            try
            {
                var (factory, connection) = InitSQLiteDb();
                using (factory)
                {
                    using (connection)
                    {
                        using var dataAdapter = factory.CreateDataAdapter(connection, TableName);
                        dataFrame.SaveTo(dataAdapter, factory);

                        var resDataFrame = await DataFrame.LoadFrom(dataAdapter);

                        AssertEqual(resDataFrame, columns, vals);
                    }
                }
            }
            finally
            {
                CleanupSQLiteDb();
            }
        }

        static void AssertEqual(DataFrame dataFrame, (string name, Type type)[] columns, object[][] vals)
        {
            var resColumns = dataFrame.Columns.Select(column => (column.Name, column.DataType)).ToArray();
            Assert.Equal(columns, resColumns);
            var resVals = dataFrame.Rows.Select(row => row.ToArray()).ToArray();
            Assert.Equal(vals, resVals);
        }

        static ((string name, Type type)[] columns, object[][] vals) GetTestData()
        {
            const int RowsCount = 10_000;

            var columns = new[]
            {
                ("ID", typeof(long)),
                ("Text", typeof(string))
            };

            var vals = new object[RowsCount][];
            for (var i = 0L; i < RowsCount; i++)
            {
                var row = new object[columns.Length];
                row[0] = i;
                row[1] = $"test {i}";
                vals[i] = row;
            }

            return (columns, vals);
        }

        static (SQLiteProviderFactory factory, DbConnection connection) InitSQLiteDb()
        {
            var connectionString = $"DataSource={SQLitePath};Version=3;New=True;Compress=True;";

            SQLiteConnection.CreateFile(SQLitePath);
            var factory = new SQLiteProviderFactory();

            var connection = factory.CreateConnection();
            connection.ConnectionString = connectionString;
            connection.Open();

            using var command = connection.CreateCommand();
            command.CommandText = $"CREATE TABLE {TableName} (ID INTEGER NOT NULL PRIMARY KEY ASC, Text VARCHAR(25))";
            command.ExecuteNonQuery();

            return (factory, connection);
        }

        static void CleanupSQLiteDb()
        {
            if (File.Exists(SQLitePath))
                File.Delete(SQLitePath);
        }

        static readonly string BasePath =
            Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + "/";

        const string DbName = "TestDb";
        const string TableName = "TestTable";

        static readonly string SQLitePath = $@"{BasePath}/{DbName}.sqlite";

        public readonly struct LoadCsvVerifyingHelper
        {
            private readonly int _columnCount;
            private readonly long _rowCount;
            private readonly string[] _columnNames;
            private readonly Type[] _columnTypes;
            private readonly object[][] _cells;

            public LoadCsvVerifyingHelper(int columnCount, long rowCount, string[] columnNames, Type[] columnTypes, object[][] cells)
            {
                _columnCount = columnCount;
                _rowCount = rowCount;
                _columnNames = columnNames;
                _columnTypes = columnTypes;
                _cells = cells;

            }

            public void VerifyLoadCsv(DataFrame df)
            {
                Assert.Equal(_rowCount, df.Rows.Count);
                Assert.Equal(_columnCount, df.Columns.Count);

                for (int j = 0; j < _columnCount; j++)
                {
                    Assert.True(_columnTypes[j] == df.Columns[j].DataType);
                    Assert.Equal(_columnNames[j], df.Columns[j].Name);

                }

                VerifyColumnTypes(df);

                for (int i = 0; i < _rowCount; i++)
                {
                    Assert.Equal(_cells[i], df.Rows[i]);
                }
            }
        }

        public static IEnumerable<object[]> CsvWithTextQualifiers_TestData()
        {
            yield return new object[] // Comma Separators in Data
            {
                """
                Name,Age,Description
                Paul,34,"Paul lives in Vermont, VA."
                Victor,29,"Victor: Funny guy"
                Maria,31,
                """,
                ',',
                new Type[] { typeof(string), typeof(int), typeof(string) },
                new LoadCsvVerifyingHelper(
                    3,
                    3,
                    new string[] { "Name", "Age", "Description" },
                    new Type[] { typeof(string), typeof(int), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34, "Paul lives in Vermont, VA." },
                        new object[] { "Victor", 29, "Victor: Funny guy" },
                        new object[] { "Maria", 31, "" }
                    }
                )
            };
            yield return new object[] // Colon Separators in Data
            {
                """
                Name:Age:Description
                Paul:34:"Paul lives in Vermont, VA."
                Victor:29:"Victor: Funny guy"
                Maria:31:
                """,
                ':',
                new Type[] { typeof(string), typeof(int), typeof(string) },
                new LoadCsvVerifyingHelper(
                    3,
                    3,
                    new string[] { "Name", "Age", "Description" },
                    new Type[] { typeof(string), typeof(int), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34, "Paul lives in Vermont, VA." },
                        new object[] { "Victor", 29, "Victor: Funny guy" },
                        new object[] { "Maria", 31, "" }
                    }
                )
            };
            yield return new object[] // Comma Separators in Header
            {
                """
                "Na,me",Age,Description
                Paul,34,"Paul lives in Vermont, VA."
                Victor,29,"Victor: Funny guy"
                Maria,31,
                """,
                ',',
                new Type[] { typeof(string), typeof(int), typeof(string) },
                new LoadCsvVerifyingHelper(
                    3,
                    3,
                    new string[] { "Na,me", "Age", "Description" },
                    new Type[] { typeof(string), typeof(int), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34, "Paul lives in Vermont, VA." },
                        new object[] { "Victor", 29, "Victor: Funny guy" },
                        new object[] { "Maria", 31, "" }
                    }
                )
            };
            yield return new object[] // Newlines In Data
            {
                """
                Name,Age,Description
                Paul,34,"Paul lives in Vermont
                VA."
                Victor,29,"Victor: Funny guy"
                Maria,31,
                """,
                ',',
                new Type[] { typeof(string), typeof(int), typeof(string) },
                new LoadCsvVerifyingHelper(
                    3,
                    3,
                    new string[] { "Name", "Age", "Description" },
                    new Type[] { typeof(string), typeof(int), typeof(string) },
                    new object[][]
                    {
                        new object[]
                        {
                            "Paul",
                            34,
                            """
                            Paul lives in Vermont
                            VA.
                            """
                        },
                        new object[] { "Victor", 29, "Victor: Funny guy" },
                        new object[] { "Maria", 31, "" }
                    }
                )
            };
            yield return new object[] // Newlines In Header
            {
                """
                "Na
                me":Age:Description
                Paul:34:"Paul lives in Vermont, VA."
                Victor:29:"Victor: Funny guy"
                Maria:31:
                """,
                ':',
                new Type[] { typeof(string), typeof(int), typeof(string) },
                new LoadCsvVerifyingHelper(
                    3,
                    3,
                    new string[]
                    {
                        """
                        Na
                        me
                        """,
                        "Age",
                        "Description"
                    },
                    new Type[] { typeof(string), typeof(int), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34, "Paul lives in Vermont, VA." },
                        new object[] { "Victor", 29, "Victor: Funny guy" },
                        new object[] { "Maria", 31, "" }
                    }
                )
            };
            yield return new object[] // Quotations in Data
            {
                """
                Name,Age,Description
                Paul,34,"Paul lives in ""Vermont VA""."
                Victor,29,"Victor: Funny guy"
                Maria,31,
                """,
                ',',
                new Type[] { typeof(string), typeof(int), typeof(string) },
                new LoadCsvVerifyingHelper(
                    3,
                    3,
                    new string[] { "Name", "Age", "Description" },
                    new Type[] { typeof(string), typeof(int), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34, """Paul lives in "Vermont VA".""" },
                        new object[] { "Victor", 29, "Victor: Funny guy" },
                        new object[] { "Maria", 31, "" }
                    }
                )
            };
            yield return new object[] // Quotations in Header
            {
                """
                Name,Age,"De""script""ion"
                Paul,34,"Paul lives in Vermont, VA."
                Victor,29,"Victor: Funny guy"
                Maria,31,
                """,
                ',',
                new Type[] { typeof(string), typeof(int), typeof(string) },
                new LoadCsvVerifyingHelper(
                    3,
                    3,
                    new string[] { "Name", "Age", """De"script"ion""" },
                    new Type[] { typeof(string), typeof(int), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34, "Paul lives in Vermont, VA." },
                        new object[] { "Victor", 29, "Victor: Funny guy" },
                        new object[] { "Maria", 31, "" }
                    }
                )
            };
        }

        [Theory]
        [MemberData(nameof(CsvWithTextQualifiers_TestData))]
        public void TestLoadCsvWithTextQualifiersFromStream(string data, char separator, Type[] dataTypes, LoadCsvVerifyingHelper helper)
        {
            DataFrame df = DataFrame.LoadCsv(GetStream(data), dataTypes: dataTypes, separator: separator);
            helper.VerifyLoadCsv(df);
        }

        [Theory]
        [MemberData(nameof(CsvWithTextQualifiers_TestData))]
        public void TestLoadCsvWithTextQualifiersFromString(string data, char separator, Type[] dataTypes, LoadCsvVerifyingHelper helper)
        {
            DataFrame df = DataFrame.LoadCsvFromString(data, dataTypes: dataTypes, separator: separator);
            helper.VerifyLoadCsv(df);
        }

        [Theory]
        [MemberData(nameof(CsvWithTextQualifiers_TestData))]
        public void TestSaveCsvWithTextQualifiers(string data, char separator, Type[] dataTypes, LoadCsvVerifyingHelper helper)
        {
            DataFrame df = DataFrame.LoadCsv(GetStream(data), dataTypes: dataTypes, separator: separator);

            using MemoryStream csvStream = new MemoryStream();
            DataFrame.SaveCsv(df, csvStream, separator: separator);

            // We are verifying that SaveCsv works by reading the result back to a DataFrame and verifying correctness,
            // ensuring no information loss
            csvStream.Seek(0, SeekOrigin.Begin);
            DataFrame df2 = DataFrame.LoadCsv(csvStream, dataTypes: dataTypes, separator: separator);
            helper.VerifyLoadCsv(df2);
        }

        [Fact]
        public void TestLoadCsvWithGuessTypes()
        {
            string csvString = """
                Name,Age,Description,UpdatedOn,Weight,LargeNumber,NullColumn
                Paul,34,"Paul lives in Vermont, VA.",2024-01-23T05:06:15.028,195.48,123,null
                Victor,29,"Victor: Funny guy",2023-11-04T17:27:59.167,175.3,2147483648,null
                Clara,,,,,,null
                Ellie,null,null,null,null,null,null
                Maria,31,,2024-03-31T07:20:47.250,126,456,null
                """;

            var defaultResultVerifyingHelper = new LoadCsvVerifyingHelper(
                    7,
                    5,
                    new string[] { "Name", "Age", "Description", "UpdatedOn", "Weight", "LargeNumber", "NullColumn" },
                    new Type[] { typeof(string), typeof(float), typeof(string), typeof(DateTime), typeof(float), typeof(float), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34f, "Paul lives in Vermont, VA.",  DateTime.Parse("2024-01-23T05:06:15.028"), 195.48f, 123f, null },
                        new object[] { "Victor", 29f, "Victor: Funny guy", DateTime.Parse("2023-11-04T17:27:59.167"), 175.3f, 2147483648f, null },
                        new object[] { "Clara", null, "", null, null, null, null },
                        new object[] { "Ellie", null, null, null, null, null, null },
                        new object[] { "Maria", 31f, "", DateTime.Parse("2024-03-31T07:20:47.250"), 126f, 456f, null }
                    }
                );

            var customResultVerifyingHelper = new LoadCsvVerifyingHelper(
                    7,
                    5,
                    new string[] { "Name", "Age", "Description", "UpdatedOn", "Weight", "LargeNumber", "NullColumn" },
                    new Type[] { typeof(string), typeof(int), typeof(string), typeof(DateTime), typeof(double), typeof(long), typeof(string) },
                    new object[][]
                    {
                        new object[] { "Paul", 34, "Paul lives in Vermont, VA.",  DateTime.Parse("2024-01-23T05:06:15.028"), 195.48, 123L, null },
                        new object[] { "Victor", 29, "Victor: Funny guy", DateTime.Parse("2023-11-04T17:27:59.167"), 175.3, 2147483648L, null },
                        new object[] { "Clara", null, "", null, null, null, null },
                        new object[] { "Ellie", null, null, null, null, null, null },
                        new object[] { "Maria", 31, "", DateTime.Parse("2024-03-31T07:20:47.250"), 126.0, 456L, null }
                    }
                );

            Type CustomGuessTypeFunction(IEnumerable<string> columnValues)
            {
                List<Type> types = [
                    typeof(bool),
                    typeof(int),
                    typeof(long),
                    typeof(double),
                    typeof(DateTime)
                    ];

                bool allNullData = true;

                HashSet<Type> possibleTypes = new HashSet<Type>(types);

                foreach (var item in columnValues)
                {
                    if (string.IsNullOrEmpty(item) || string.Equals(item, "null", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }
                    else
                    {
                        allNullData = false;
                    }

                    List<Type> typesToRemove = new List<Type>(possibleTypes.Count);

                    foreach (var type in possibleTypes)
                    {
                        if (type == typeof(bool))
                        {
                            if (!bool.TryParse(item, out bool result))
                            {
                                typesToRemove.Add(type);
                            }
                        }
                        else if (type == typeof(int))
                        {
                            if (!int.TryParse(item, out int result))
                            {
                                typesToRemove.Add(type);
                            }
                        }
                        else if (type == typeof(long))
                        {
                            if (!long.TryParse(item, out long result))
                            {
                                typesToRemove.Add(type);
                            }
                        }
                        else if (type == typeof(double))
                        {
                            if (!double.TryParse(item, out double result))
                            {
                                typesToRemove.Add(type);
                            }
                        }
                        else if (type == typeof(DateTime))
                        {
                            if (!DateTime.TryParse(item, out DateTime result))
                            {
                                typesToRemove.Add(type);
                            }
                        }
                    }

                    foreach (var type in typesToRemove)
                    {
                        possibleTypes.Remove(type);
                    }
                }

                if (allNullData)
                {
                    // Could not determine type since all data was null
                    return typeof(string);
                }

                foreach (var type in types)
                {
                    if (possibleTypes.Contains(type))
                    {
                        return type;
                    }
                }

                return typeof(string);
            }

            DataFrame defaultDf = DataFrame.LoadCsvFromString(csvString);

            defaultResultVerifyingHelper.VerifyLoadCsv(defaultDf);

            DataFrame customDf = DataFrame.LoadCsvFromString(csvString, guessTypeFunction: CustomGuessTypeFunction);

            customResultVerifyingHelper.VerifyLoadCsv(customDf);
        }

        [Fact]
        public void TestLoadCsvWithMismatchedNumberOfColumnsInDataRows()
        {
            // Victor line is missing the "LargeNumber" row
            string csvString = """
                Name,Age,Description,UpdatedOn,Weight,LargeNumber
                Paul,34,"Paul lives in Vermont, VA.",2024-01-23T05:06:15.028,195.48,123
                Victor,29,"Victor: Funny guy",2023-11-04T17:27:59.167,175.3
                Clara,,,,,
                Ellie,null,null,null,null,null
                Maria,31,,2024-03-31T07:20:47.250,126,456
                """;

            Assert.Throws<FormatException>(() => DataFrame.LoadCsvFromString(csvString));
        }
    }
}
