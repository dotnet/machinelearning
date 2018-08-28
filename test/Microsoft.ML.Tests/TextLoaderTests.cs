// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.TestFramework;
using System;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
    public class TextLoaderTestPipe : TestDataPipeBase
    {
        public TextLoaderTestPipe(ITestOutputHelper output)
            : base(output)
        {

        }

        [Fact]
        public void TestTextLoaderDataTypes()
        {
            string pathData = DeleteOutputPath("SavePipe", "TextInput.txt");
            File.WriteAllLines(pathData, new string[] {
                string.Format("{0},{1},{2},{3}", sbyte.MinValue, short.MinValue, int.MinValue, long.MinValue),
                string.Format("{0},{1},{2},{3}", sbyte.MaxValue, short.MaxValue, int.MaxValue, long.MaxValue),
                ",,,"
            });

            var data = TestCore(pathData, true,
                new[] {
                "loader=Text{col=DvInt1:I1:0 col=DvInt2:I2:1 col=DvInt4:I4:2 col=DvInt8:I8:3 sep=comma}",
                }, logCurs: true);

            using (var cursor = data.GetRowCursor((a => true)))
            {
                var col1 = cursor.GetGetter<sbyte>(0);
                var col2 = cursor.GetGetter<short>(1);
                var col3 = cursor.GetGetter<int>(2);
                var col4 = cursor.GetGetter<long>(3);

                Assert.True(cursor.MoveNext());

                sbyte[] sByteTargets = new sbyte[] { sbyte.MinValue, sbyte.MaxValue, default};
                short[] shortTargets = new short[] { short.MinValue, short.MaxValue, default };
                int[] intTargets = new int[] { int.MinValue, int.MaxValue, default };
                long[] longTargets = new long[] { long.MinValue, long.MaxValue, default };

                int i = 0;
                for (; i < sByteTargets.Length; i++)
                {
                    sbyte sbyteValue = -1;
                    col1(ref sbyteValue);
                    Assert.Equal(sByteTargets[i], sbyteValue);

                    short shortValue = -1;
                    col2(ref shortValue);
                    Assert.Equal(shortTargets[i], shortValue);

                    int intValue = -1;
                    col3(ref intValue);
                    Assert.Equal(intTargets[i], intValue);

                    long longValue = -1;
                    col4(ref longValue);
                    Assert.Equal(longTargets[i], longValue);

                    if (i < sByteTargets.Length - 1)
                        Assert.True(cursor.MoveNext());
                    else
                        Assert.False(cursor.MoveNext());
                }

                Assert.Equal(i, sByteTargets.Length);
            }
        }

        [Fact]
        public void TestTextLoaderInvalidLongMin()
        {
            string pathData = DeleteOutputPath("SavePipe", "TextInput.txt");
            File.WriteAllLines(pathData, new string[] {
                "-9223372036854775809"

            });

            try
            {
                var data = TestCore(pathData, true,
                    new[] {
                    "loader=Text{col=DvInt8:I8:0 sep=comma}",
                    }, logCurs: true);
            }
            catch(Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to long.", ex.Message);
                return;
            }

            Assert.True(false, "Test failed.");
        }

        [Fact]
        public void TestTextLoaderInvalidLongMax()
        {
            string pathData = DeleteOutputPath("SavePipe", "TextInput.txt");
            File.WriteAllLines(pathData, new string[] {
                "9223372036854775808"

            });

            try
            {
                var data = TestCore(pathData, true,
                    new[] {
                    "loader=Text{col=DvInt8:I8:0 sep=comma}",
                    }, logCurs: true);
            }
            catch (Exception ex)
            {
                Assert.Equal("Value could not be parsed from text to long.", ex.Message);
                return;
            }

            Assert.True(false, "Test failed.");
        }
    }

    public class TextLoaderTests : BaseTestClass
    {
        public TextLoaderTests(ITestOutputHelper output)
            : base(output)
        {

        }
        
        [Fact]
        public void ConstructorDoesntThrow()
        {
            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<Input>());
            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<Input>(useHeader:true));
            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<Input>());
            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<Input>(useHeader: false));
            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<Input>(useHeader: false, supportSparse: false, trimWhitespace: false));
            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<Input>(useHeader: false, supportSparse: false));
            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<Input>(useHeader: false, allowQuotedStrings: false));

            Assert.NotNull(new Data.TextLoader("fakeFile.txt").CreateFrom<InputWithUnderscore>());
        }


        [Fact]
        public void CanSuccessfullyApplyATransform()
        {
            var loader = new Data.TextLoader("fakeFile.txt").CreateFrom<Input>();

            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = loader.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                Assert.NotNull(output.Data);
                Assert.NotNull(output.Data.VarName);
                Assert.Null(output.Model);
            }
        }

        [Fact]
        public void CanSuccessfullyRetrieveQuotedData()
        {
            string dataPath = GetDataPath("QuotingData.csv");
            var loader = new Data.TextLoader(dataPath).CreateFrom<QuoteInput>(useHeader: true, separator: ',', allowQuotedStrings: true, supportSparse: false);
            
            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = loader.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                experiment.Compile();
                loader.SetInput(environment, experiment);
                experiment.Run();

                IDataView data = experiment.GetOutput(output.Data);
                Assert.NotNull(data);

                using (var cursor = data.GetRowCursor((a => true)))
                {
                    var IDGetter = cursor.GetGetter<float>(0);
                    var TextGetter = cursor.GetGetter<DvText>(1);

                    Assert.True(cursor.MoveNext());

                    float ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(1, ID);

                    DvText Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("This text contains comma, within quotes.", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(2, ID);

                    Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("This text contains extra punctuations and special characters.;*<>?!@#$%^&*()_+=-{}|[]:;'", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(3, ID);

                    Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("This text has no quotes", Text.ToString());

                    Assert.False(cursor.MoveNext());
                }
            }
        }

        [Fact]
        public void CanSuccessfullyRetrieveSparseData()
        {
            string dataPath = GetDataPath("SparseData.txt");
            var loader = new Data.TextLoader(dataPath).CreateFrom<SparseInput>(useHeader: true, allowQuotedStrings: false, supportSparse: true);

            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = loader.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                experiment.Compile();
                loader.SetInput(environment, experiment);
                experiment.Run();

                IDataView data = experiment.GetOutput(output.Data);
                Assert.NotNull(data);

                using (var cursor = data.GetRowCursor((a => true)))
                {
                    var getters = new ValueGetter<float>[]{
                        cursor.GetGetter<float>(0),
                        cursor.GetGetter<float>(1),
                        cursor.GetGetter<float>(2),
                        cursor.GetGetter<float>(3),
                        cursor.GetGetter<float>(4)
                    };


                    Assert.True(cursor.MoveNext());

                    float[] targets = new float[] { 1, 2, 3, 4, 5 };
                    for (int i = 0; i < getters.Length; i++)
                    {
                        float value = 0;
                        getters[i](ref value);
                        Assert.Equal(targets[i], value);
                    }

                    Assert.True(cursor.MoveNext());

                    targets = new float[] { 0, 0, 0, 4, 5 };
                    for (int i = 0; i < getters.Length; i++)
                    {
                        float value = 0;
                        getters[i](ref value);
                        Assert.Equal(targets[i], value);
                    }

                    Assert.True(cursor.MoveNext());

                    targets = new float[] { 0, 2, 0, 0, 0 };
                    for (int i = 0; i < getters.Length; i++)
                    {
                        float value = 0;
                        getters[i](ref value);
                        Assert.Equal(targets[i], value);
                    }

                    Assert.False(cursor.MoveNext());
                }
            }

        }

        [Fact]
        public void CanSuccessfullyTrimSpaces()
        {
            string dataPath = GetDataPath("TrimData.csv");
            var loader = new Data.TextLoader(dataPath).CreateFrom<QuoteInput>(useHeader: true, separator: ',', allowQuotedStrings: false, supportSparse: false, trimWhitespace: true);

            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = loader.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                experiment.Compile();
                loader.SetInput(environment, experiment);
                experiment.Run();

                IDataView data = experiment.GetOutput(output.Data);
                Assert.NotNull(data);

                using (var cursor = data.GetRowCursor((a => true)))
                {
                    var IDGetter = cursor.GetGetter<float>(0);
                    var TextGetter = cursor.GetGetter<DvText>(1);

                    Assert.True(cursor.MoveNext());

                    float ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(1, ID);

                    DvText Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("There is a space at the end", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(2, ID);

                    Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("There is no space at the end", Text.ToString());
                    
                    Assert.False(cursor.MoveNext());
                }
            }
        }

        [Fact]
        public void ThrowsExceptionWithPropertyName()
        {
            Exception ex = Assert.Throws<InvalidOperationException>( () => new Data.TextLoader("fakefile.txt").CreateFrom<ModelWithoutColumnAttribute>() );
            Assert.StartsWith("Field or property String1 is missing ColumnAttribute", ex.Message);
        }

        public class QuoteInput
        {
            [Column("0")]
            public float ID;

            [Column("1")]
            public string Text;
        }

        public class SparseInput
        {
            [Column("0")]
            public float C1;

            [Column("1")]
            public float C2;

            [Column("2")]
            public float C3;

            [Column("3")]
            public float C4;

            [Column("4")]
            public float C5;
        }

        public class Input
        {
            [Column("0")]
            public string String1;

            [Column("1")]
            public float Number1;
        }

        public class InputWithUnderscore
        {
            [Column("0")]
            public string String_1;

            [Column("1")]
            public float Number_1;
        }

        public class ModelWithoutColumnAttribute
        {
            public string String1;
        }
    }
}
