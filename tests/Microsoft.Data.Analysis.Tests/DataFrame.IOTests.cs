// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public class DataFrameIOTests
    {
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
                else
                {
                    throw new NotImplementedException("Unit test has to be updated");
                }
            }
        }

        [Fact]
        public void TestReadCsvWithHeader()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD,17.5
CMT,1,1,474,1.5,CRD,8
CMT,1,1,637,1.4,CRD,8.5
CMT,1,1,181,0.6,CSH,4.5";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }
            void RegularTest(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal("CMT", df.Columns["vendor_id"][3]);
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
                Assert.Equal("CMT", reducedRows.Columns["vendor_id"][2]);
                VerifyColumnTypes(df);
            }
            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), numberOfRowsToRead: 3);
            ReducedRowsTest(reducedRows);
            csvDf = DataFrame.LoadCsvFromString(data, numberOfRowsToRead: 3);
            ReducedRowsTest(csvDf);
        }

        [Fact]
        public void TestReadCsvNoHeader()
        {
            string data = @"CMT,1,1,1271,3.8,CRD,17.5
CMT,1,1,474,1.5,CRD,8
CMT,1,1,637,1.4,CRD,8.5
CMT,1,1,181,0.6,CSH,4.5";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }
            void RegularTest(DataFrame df)
            {
                Assert.Equal(4, df.Rows.Count);
                Assert.Equal(7, df.Columns.Count);
                Assert.Equal("CMT", df.Columns["Column0"][3]);
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
                Assert.Equal("CMT", reducedRows.Columns["Column0"][2]);
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
        public void TestReadCsvWithTypesAndGuessRows(bool header, int guessRows)
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

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }

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

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }
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
        public void TestReadCsvWithPipeSeparator()
        {
            string data = @"vendor_id|rate_code|passenger_count|trip_time_in_secs|trip_distance|payment_type|fare_amount
CMT|1|1|1271|3.8|CRD|17.5
CMT|1|1|474|1.5|CRD|8
CMT|1|1|637|1.4|CRD|8.5
||||||
CMT|1|1|181|0.6|CSH|4.5";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }
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
        public void TestReadCsvWithSemicolonSeparator()
        {
            string data = @"vendor_id;rate_code;passenger_count;trip_time_in_secs;trip_distance;payment_type;fare_amount
CMT;1;1;1271;3.8;CRD;17.5
CMT;1;1;474;1.5;CRD;8
CMT;1;1;637;1.4;CRD;8.5
;;;;;;
CMT;1;1;181;0.6;CSH;4.5";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }
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
        public void TestReadCsvWithExtraColumnInHeader()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount,extra
CMT,1,1,1271,3.8,CRD,17.5
CMT,1,1,474,1.5,CRD,8
CMT,1,1,637,1.4,CRD,8.5
CMT,1,1,181,0.6,CSH,4.5";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }
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
        public void TestReadCsvWithExtraColumnInRow()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD,17.5,0
CMT,1,1,474,1.5,CRD,8,0
CMT,1,1,637,1.4,CRD,8.5,0
CMT,1,1,181,0.6,CSH,4.5,0";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }

            Assert.Throws<IndexOutOfRangeException>(() => DataFrame.LoadCsv(GetStream(data)));
            Assert.Throws<IndexOutOfRangeException>(() => DataFrame.LoadCsvFromString(data));
        }

        [Fact]
        public void TestReadCsvWithLessColumnsInRow()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
CMT,1,1,1271,3.8,CRD
CMT,1,1,474,1.5,CRD
CMT,1,1,637,1.4,CRD
CMT,1,1,181,0.6,CSH";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }

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
        public void TestReadCsvWithAllNulls()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs
null,null,null,null
Null,Null,Null,Null
null,null,null,null
Null,Null,Null,Null
null,null,null,null
null,null,null,null";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }

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
        public void TestReadCsvWithNullsAndDataTypes()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs
null,1,1,1271
CMT,Null,1,474
CMT,1,null,637
Null,,,
,,,
CMT,1,1,null";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }

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
        public void TestReadCsvWithNulls()
        {
            string data = @"vendor_id,rate_code,passenger_count,trip_time_in_secs
null,1,1,1271
CMT,Null,1,474
CMT,1,null,637
Null,,,
,,,
CMT,1,1,null";

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }

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
        public void TestWriteCsvWithHeader()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, true);

            DataFrame.WriteCsv(dataFrame, csvStream);

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
        public void TestWriteCsvWithCultureInfoRomanianAndSemiColon()
        {
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithNumericColumns(10, true);
            dataFrame[1, 1] = 1.1M;
            dataFrame[1, 2] = 1.2D;
            dataFrame[1, 3] = 1.3F;

            using MemoryStream csvStream = new MemoryStream();
            var cultureInfo = new CultureInfo("ro-RO");
            var separator = ';';
            DataFrame.WriteCsv(dataFrame, csvStream, separator: separator, cultureInfo: cultureInfo);

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
        public void TestWriteCsvWithCultureInfo()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithNumericColumns(10, true);
            dataFrame[1, 1] = 1.1M;
            dataFrame[1, 2] = 1.2D;
            dataFrame[1, 3] = 1.3F;

            var cultureInfo = new CultureInfo("en-US");
            DataFrame.WriteCsv(dataFrame, csvStream, cultureInfo: cultureInfo);

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
        public void TestWriteCsvWithCultureInfoRomanianAndComma()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithNumericColumns(10, true);

            var cultureInfo = new CultureInfo("ro-RO");
            var separator = cultureInfo.NumberFormat.NumberDecimalSeparator.First();

            Assert.Throws<ArgumentException>(() => DataFrame.WriteCsv(dataFrame, csvStream, separator: separator, cultureInfo: cultureInfo));
        }

        [Fact]
        public void TestWriteCsvWithNoHeader()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, true);

            DataFrame.WriteCsv(dataFrame, csvStream, header: false);

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
        public void TestWriteCsvWithSemicolonSeparator()
        {
            using MemoryStream csvStream = new MemoryStream();
            DataFrame dataFrame = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, true);

            var separator = ';';
            DataFrame.WriteCsv(dataFrame, csvStream, separator: separator);

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

            Stream GetStream(string streamData)
            {
                return new MemoryStream(Encoding.Default.GetBytes(streamData));
            }
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
    }
}
