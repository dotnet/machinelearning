﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Apache.Arrow;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests
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
            DataFrame df = DataFrame.LoadCsv(GetStream(data));
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
            DataFrame df = DataFrame.LoadCsv(GetStream(data), header: false);
            Assert.Equal(4, df.Rows.Count);
            Assert.Equal(7, df.Columns.Count);
            Assert.Equal("CMT", df.Columns["Column0"][3]);
            VerifyColumnTypes(df);

            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), header: false, numberOfRowsToRead: 3);
            Assert.Equal(3, reducedRows.Rows.Count);
            Assert.Equal(7, reducedRows.Columns.Count);
            Assert.Equal("CMT", reducedRows.Columns["Column0"][2]);
            VerifyColumnTypes(df);
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

            if (guessRows == 10)
            {
                df = DataFrame.LoadCsv(GetStream(data),
                                                 header: header,
                                                 guessRows: guessRows
                                                 );
                VerifyDataFrameWithNamedColumnsAndDataTypes(df, verifyColumnDataType: false, verifyNames: header);
            }
            else
            {
                Assert.ThrowsAny<ArgumentException>(() => DataFrame.LoadCsv(GetStream(data),
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
            DataFrame df = DataFrame.LoadCsv(GetStream(data), dataTypes: new Type[] { typeof(string), typeof(short), typeof(int), typeof(long), typeof(float), typeof(string), typeof(double) });
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
            DataFrame df = DataFrame.LoadCsv(GetStream(data), separator: '|');

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
            DataFrame df = DataFrame.LoadCsv(GetStream(data), separator: ';');

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
            DataFrame df = DataFrame.LoadCsv(GetStream(data));

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

            DataFrame df = DataFrame.LoadCsv(GetStream(data));
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
    }
}
