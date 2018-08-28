// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TextAnalytics;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    /// <summary>
    /// A class for non-baseline data pipe tests.
    /// </summary>
    public sealed partial class TestDataPipeNoBaseline : TestDataViewBase
    {
        [Fact]
        public void TestLDATransform()
        {
            var builder = new ArrayDataViewBuilder(Env);
            var data = new[]
            {
                new[] {  (Float)1.0,  (Float)0.0,  (Float)0.0 },
                new[] {  (Float)0.0,  (Float)1.0,  (Float)0.0 },
                new[] {  (Float)0.0,  (Float)0.0,  (Float)1.0 },
            };

            builder.AddColumn("F1V", NumberType.Float, data);

            var srcView = builder.GetDataView();

            LdaTransform.Column col = new LdaTransform.Column();
            col.Source = "F1V";
            col.NumTopic = 20;
            col.NumTopic = 3;
            col.NumSummaryTermPerTopic = 3;
            col.AlphaSum = 3;
            col.NumThreads = 1;
            col.ResetRandomGenerator = true;
            LdaTransform.Arguments args = new LdaTransform.Arguments();
            args.Column = new LdaTransform.Column[] { col };

            LdaTransform ldaTransform = new LdaTransform(Env, args, srcView);

            using (var cursor = ldaTransform.GetRowCursor(c => true))
            {
                var resultGetter = cursor.GetGetter<VBuffer<Float>>(1);
                VBuffer<Float> resultFirstRow = new VBuffer<Float>();
                VBuffer<Float> resultSecondRow = new VBuffer<Float>();
                VBuffer<Float> resultThirdRow = new VBuffer<Float>();

                Assert.True(cursor.MoveNext());
                resultGetter(ref resultFirstRow);
                Assert.True(cursor.MoveNext());
                resultGetter(ref resultSecondRow);
                Assert.True(cursor.MoveNext());
                resultGetter(ref resultThirdRow);
                Assert.False(cursor.MoveNext());

                Assert.True(resultFirstRow.Length == 3);
                Assert.True(resultFirstRow.GetItemOrDefault(0) == 0);
                Assert.True(resultFirstRow.GetItemOrDefault(2) == 0);
                Assert.True(resultFirstRow.GetItemOrDefault(1) == 1.0);
                Assert.True(resultSecondRow.Length == 3);
                Assert.True(resultSecondRow.GetItemOrDefault(0) == 0);
                Assert.True(resultSecondRow.GetItemOrDefault(2) == 0);
                Assert.True(resultSecondRow.GetItemOrDefault(1) == 1.0);
                Assert.True(resultThirdRow.Length == 3);
                Assert.True(resultThirdRow.GetItemOrDefault(0) == 0);
                Assert.True(resultThirdRow.GetItemOrDefault(1) == 0);
                Assert.True(resultThirdRow.GetItemOrDefault(2) == 1.0);
            }

            using (var cursor = ldaTransform.GetRowCursor(c => true))
            {
                var resultGetter = cursor.GetGetter<VBuffer<Float>>(1);
                VBuffer<Float> resultFirstRow = new VBuffer<Float>();
                VBuffer<Float> resultSecondRow = new VBuffer<Float>();
                VBuffer<Float> resultThirdRow = new VBuffer<Float>();

                Assert.True(cursor.MoveNext());
                resultGetter(ref resultFirstRow);
                Assert.True(cursor.MoveNext());
                resultGetter(ref resultSecondRow);
                Assert.True(cursor.MoveNext());
                resultGetter(ref resultThirdRow);
                Assert.False(cursor.MoveNext());

                Assert.True(resultFirstRow.Length == 3);
                Assert.True(resultFirstRow.GetItemOrDefault(0) == 0);
                Assert.True(resultFirstRow.GetItemOrDefault(2) == 0);
                Assert.True(resultFirstRow.GetItemOrDefault(1) == 1.0);
                Assert.True(resultSecondRow.Length == 3);
                Assert.True(resultSecondRow.GetItemOrDefault(0) == 0);
                Assert.True(resultSecondRow.GetItemOrDefault(2) == 0);
                Assert.True(resultSecondRow.GetItemOrDefault(1) == 1.0);
                Assert.True(resultThirdRow.Length == 3);
                Assert.True(resultThirdRow.GetItemOrDefault(0) == 0);
                Assert.True(resultThirdRow.GetItemOrDefault(1) == 0);
                Assert.True(resultThirdRow.GetItemOrDefault(2) == 1.0);
            }
        }

        [Fact]
        public void TestLdaTransformEmptyDocumentException()
        {
            var builder = new ArrayDataViewBuilder(Env);
            var data = new[]
            {
                new[] {  (Float)0.0,  (Float)0.0,  (Float)0.0 },
                new[] {  (Float)0.0,  (Float)0.0,  (Float)0.0 },
                new[] {  (Float)0.0,  (Float)0.0,  (Float)0.0 },
            };

            builder.AddColumn("Zeros", NumberType.Float, data);

            var srcView = builder.GetDataView();
            var col = new LdaTransform.Column()
            {
                Source = "Zeros"
            };
            var args = new LdaTransform.Arguments()
            {
                Column = new[] { col }
            };

            try
            {
                var lda = new LdaTransform(Env, args, srcView);
            }
            catch (InvalidOperationException ex)
            {
                Assert.Equal(ex.Message, string.Format("The specified documents are all empty in column '{0}'.", col.Source));
                return;
            }

            Assert.True(false, "The LDA transform does not throw expected error on empty documents.");
        }
    }

    public sealed partial class TestDataPipe : TestDataPipeBase
    {

        [Fact]
        public void TestParquetPrimitiveDataTypes()
        {
            string pathData = GetDataPath(@"..\data\Parquet", "alltypes.parquet");
            TestCore(pathData, false, new[] { "loader=Parquet{bigIntDates=+}" });
            Done();
        }

        [Fact]
        public void TestParquetNull()
        {
            string pathData = GetDataPath(@"..\data\Parquet", "test-null.parquet");
            TestCore(pathData, false, new[] { "loader=Parquet{bigIntDates=+}" }, forceDense: true);
            Done();
        }
    }
}
