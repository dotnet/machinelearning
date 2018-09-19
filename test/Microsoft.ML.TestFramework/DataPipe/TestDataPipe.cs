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
    public sealed partial class TestDataPipe : TestDataPipeBase
    {
        private static Float[] dataFloat = new Float[] { -0.0f, 0,  1, -1,  2, -2, Single.NaN, Single.MinValue,
                Single.MaxValue, Single.Epsilon, Single.NegativeInfinity, Single.PositiveInfinity };
        private static uint[] resultsFloat = new uint[] { 21, 21, 16, 16, 31, 17, 0, 23, 24, 15, 10, 7 };

        private static VBuffer<Single> dataFloatSparse = new VBuffer<Single>(5, 3, new float[] { -0.0f, 0, 1 }, new[] { 0, 3, 4 });
        private static uint[] resultsFloatSparse = new uint[] { 21, 21, 21, 21, 16 };

        private static Double[] dataDouble = new Double[]   { -0.0, 0, 1, -1,  2, -2, Double.NaN, Double.MinValue,
                Double.MaxValue, Double.Epsilon, Double.NegativeInfinity, Double.PositiveInfinity };
        private static uint[] resultsDouble = new uint[] { 21, 21, 31, 17, 10, 15, 0, 16, 21, 15, 6, 30 };

        private static VBuffer<Double> dataDoubleSparse = new VBuffer<Double>(5, 3, new double[] { -0.0, 0, 1 }, new[] { 0, 3, 4 });
        private static uint[] resultsDoubleSparse = new uint[] { 21, 21, 21, 21, 31 };

        [Fact]
        public void TestHashTransformFloat()
        {
            TestHashTransformHelper(dataFloat, resultsFloat, NumberType.R4);
        }

        [Fact]
        public void TestHashTransformFloatVector()
        {
            var data = new[] { dataFloat };
            var results = new[] { resultsFloat };
            TestHashTransformVectorHelper(data, results, NumberType.R4);
        }

        [Fact]
        public void TestHashTransformFloatSparseVector()
        {
            var results = new[] { resultsFloatSparse };
            TestHashTransformVectorHelper(dataFloatSparse, results, NumberType.R4);
        }

        [Fact]
        public void TestHashTransformDoubleSparseVector()
        {
            var results = new[] { resultsDoubleSparse };
            TestHashTransformVectorHelper(dataDoubleSparse, results, NumberType.R8);
        }

        [Fact]
        public void TestHashTransformDouble()
        {
            TestHashTransformHelper(dataDouble, resultsDouble, NumberType.R8);
        }

        [Fact]
        public void TestHashTransformDoubleVector()
        {
            var data = new[] { dataDouble };
            var results = new[] { resultsDouble };
            TestHashTransformVectorHelper(data, results, NumberType.R8);
        }

        private void TestHashTransformHelper<T>(T[] data, uint[] results, NumberType type)
        {
            var builder = new ArrayDataViewBuilder(Env);

            builder.AddColumn("F1", type, data);
            var srcView = builder.GetDataView();

            HashTransform.Column col = new HashTransform.Column();
            col.Source = "F1";
            col.HashBits = 5;
            col.Seed = 42;
            HashTransform.Arguments args = new HashTransform.Arguments();
            args.Column = new HashTransform.Column[] { col };

            var hashTransform = HashTransform.Create(Env, args, srcView);
            using (var cursor = hashTransform.GetRowCursor(c => true))
            {
                var resultGetter = cursor.GetGetter<uint>(1);
                uint resultRow = 0;
                foreach (var r in results)
                {
                    Assert.True(cursor.MoveNext());
                    resultGetter(ref resultRow);
                    Assert.True(resultRow == r);
                }
            }
        }

        private void TestHashTransformVectorHelper<T>(T[][] data, uint[][] results, NumberType type)
        {
            var builder = new ArrayDataViewBuilder(Env);
            builder.AddColumn("F1V", type, data);
            TestHashTransformVectorHelper(builder, results);
        }

        private void TestHashTransformVectorHelper<T>(VBuffer<T> data, uint[][] results, NumberType type)
        {
            var builder = new ArrayDataViewBuilder(Env);
            builder.AddColumn("F1V", type, data);
            TestHashTransformVectorHelper(builder, results);
        }

        private void TestHashTransformVectorHelper(ArrayDataViewBuilder builder, uint[][] results)
        {
            var srcView = builder.GetDataView();
            HashTransform.Column col = new HashTransform.Column();
            col.Source = "F1V";
            col.HashBits = 5;
            col.Seed = 42;
            HashTransform.Arguments args = new HashTransform.Arguments();
            args.Column = new HashTransform.Column[] { col };

            var hashTransform = HashTransform.Create(Env, args, srcView);
            using (var cursor = hashTransform.GetRowCursor(c => true))
            {
                var resultGetter = cursor.GetGetter<VBuffer<uint>>(1);
                VBuffer<uint> resultRow = new VBuffer<uint>();
                foreach (var r in results)
                {
                    Assert.True(cursor.MoveNext());
                    resultGetter(ref resultRow);

                    Assert.True(resultRow.Length == r.Length);
                    for (int i = 0; i < r.Length; i++)
                        Assert.True(resultRow.GetItemOrDefault(i) == r[i]);
                }
            }
        }
    }
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
}
