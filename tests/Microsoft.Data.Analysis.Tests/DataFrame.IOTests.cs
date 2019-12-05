// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests
    {
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
            Assert.Equal("CMT", df["vendor_id"][3]);

            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), numberOfRowsToRead: 3);
            Assert.Equal(3, reducedRows.Rows.Count);
            Assert.Equal(7, reducedRows.Columns.Count);
            Assert.Equal("CMT", reducedRows["vendor_id"][2]);
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
            Assert.Equal("CMT", df["Column0"][3]);

            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), header: false, numberOfRowsToRead: 3);
            Assert.Equal(3, reducedRows.Rows.Count);
            Assert.Equal(7, reducedRows.Columns.Count);
            Assert.Equal("CMT", reducedRows["Column0"][2]);
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
    }
}
