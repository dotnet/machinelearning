// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Runtime.CompilerServices;
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
            Assert.Equal(4, df.RowCount);
            Assert.Equal(7, df.Columns.Count);
            Assert.Equal("CMT", df["vendor_id"][3]);

            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), numberOfRowsToRead: 3);
            Assert.Equal(3, reducedRows.RowCount);
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
            Assert.Equal(4, df.RowCount);
            Assert.Equal(7, df.Columns.Count);
            Assert.Equal("CMT", df["Column0"][3]);

            DataFrame reducedRows = DataFrame.LoadCsv(GetStream(data), header: false, numberOfRowsToRead: 3);
            Assert.Equal(3, reducedRows.RowCount);
            Assert.Equal(7, reducedRows.Columns.Count);
            Assert.Equal("CMT", reducedRows["Column0"][2]);
        }
    }
}
