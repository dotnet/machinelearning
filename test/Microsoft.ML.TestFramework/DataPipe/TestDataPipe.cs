// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.TextAnalytics;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using Xunit;
using Float = System.Single;

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
        public void SavePipeLabelParsers()
        {
            string pathData = GetDataPath(@"lm.sample.txt");
            string mappingPathData = GetDataPath(@"lm.labels.txt");

            TestCore(pathData, true,
                new[] {
                    "loader=Text{col=RawLabel:TXT:0 col=Names:TXT:1-2 col=Features:TXT:3-4 header+}",
                    "xf=AutoLabel{col=AutoLabel:RawLabel}",
                    "xf=Term{col=StringLabel:RawLabel terms={Wirtschaft,Gesundheit,Deutschland,Ausland,Unterhaltung,Sport,Technik & Wissen}}",
                    string.Format("xf=TermLookup{{col=FileLabel:RawLabel data={{{0}}}}}", mappingPathData),
                    "xf=ChooseColumns{col=RawLabel col=AutoLabel col=StringLabel col=FileLabel}"
                });

            mappingPathData = DeleteOutputPath("SavePipe", "Mapping.txt");
            File.WriteAllLines(mappingPathData,
                new[] {
                    "Wirtschaft\t0",
                    "Gesundheit\t0",
                    "Deutschland\t0",
                    "Ausland\t0",
                    "Unterhaltung\t0",
                    "Sport\t4294967299",
                    "Technik & Wissen\t0"
                });

            // test unbounded U8 key range
            TestCore(pathData, true,
                new[] {
                    "loader=Text{col=RawLabel:TXT:0 col=Names:TXT:1-2 col=Features:TXT:3-4 header+}",
                    string.Format("xf=TermLookup{{col=FileLabel:RawLabel data={{{0}}}}}", mappingPathData),
                    "xf=ChooseColumns{col=RawLabel col=FileLabel}"
                }, suffix: "1");

            mappingPathData = DeleteOutputPath("SavePipe", "Mapping.txt");
            File.WriteAllLines(mappingPathData,
                new[] {
                    "Wirtschaft\t100",
                    "Gesundheit\t100",
                    "Deutschland\t100",
                    "Ausland\t100",
                    "Unterhaltung\t100",
                    "Sport\t2147483758",
                    "Technik & Wissen\t100"
                });

            // test unbounded U4 key range
            TestCore(pathData, true,
                new[] {
                    "loader=Text{col=RawLabel:TXT:0 col=Names:TXT:1-2 col=Features:TXT:3-4 header+}",
                    string.Format("xf=TermLookup{{col=FileLabel:RawLabel data={{{0}}}}}", mappingPathData),
                    "xf=ChooseColumns{col=RawLabel col=FileLabel}"
                }, suffix: "2");

            mappingPathData = DeleteOutputPath("SavePipe", "Mapping.txt");
            File.WriteAllLines(mappingPathData,
                new[] {
                    "Wirtschaft\t1",
                    "Gesundheit\t0",
                    "Deutschland\t1.5",
                    "Ausland\t0.5",
                    "Unterhaltung\t1",
                    "Sport\t1",
                    "Technik & Wissen\t1"
                });

            // test numeric type
            TestCore(pathData, true,
                new[] {
                    "loader=Text{col=RawLabel:TXT:0 col=Names:TXT:1-2 col=Features:TXT:3-4 header+}",
                    string.Format("xf=TermLookup{{key=- col=FileLabel:RawLabel data={{{0}}}}}", mappingPathData),
                    "xf=ChooseColumns{col=RawLabel col=FileLabel}"
                }, suffix: "3");

            mappingPathData = DeleteOutputPath("SavePipe", "Mapping.txt");
            File.WriteAllLines(mappingPathData,
                new[] {
                    "Wirtschaft\t3.14",
                    "Gesundheit\t0.1",
                    "Deutschland\t1.5",
                    "Ausland\t0.5",
                    "Unterhaltung\t1a",
                    "Sport\t2.71",
                    "Technik & Wissen\t0.01"
                });

            // test key type with all invalid entries, and numeric type (should have missing value for every non-numeric label)
            string name = TestName + "4-out.txt";
            string pathOut = DeleteOutputPath("SavePipe", name);
            using (var writer = OpenWriter(pathOut))
            using (Env.RedirectChannelOutput(writer, writer))
            {
                TestCore(pathData, true,
                    new[] {
                        "loader=Text{col=RawLabel:TXT:0 col=Names:TXT:1-2 col=Features:TXT:3-4 header+}",
                        string.Format("xf=TermLookup{{key=- col=FileLabelNum:RawLabel data={{{0}}}}}", mappingPathData),
                        string.Format("xf=TermLookup{{col=FileLabelKey:RawLabel data={{{0}}}}}", mappingPathData),
                        "xf=ChooseColumns{col=RawLabel col=FileLabelNum col=FileLabelKey}"
                    }, suffix: "4");
                writer.WriteLine(ProgressLogLine);
                Env.PrintProgress();
            }

            CheckEqualityNormalized("SavePipe", name);

            mappingPathData = DeleteOutputPath("SavePipe", "Mapping.txt");
            File.WriteAllLines(mappingPathData,
                new[] {
                    "Wirtschaft\t10000000000",
                    "Gesundheit\t10000000001",
                    "Deutschland\t10000000002",
                    "Ausland\t10000000003",
                    "Unterhaltung\t10000000004",
                    "Sport\t10000000005",
                    "Technik & Wissen\t10000000006"
                });

            // test key type with all invalid entries, and numeric type (should have missing value for every non-numeric label)
            TestCore(pathData, true,
                new[] {
                    "loader=Text{col=RawLabel:TXT:0 col=Names:TXT:1-2 col=Features:TXT:3-4 header+}",
                    string.Format("xf=TermLookup{{col=FileLabel:RawLabel data={{{0}}}}}", mappingPathData),
                    "xf=ChooseColumns{col=RawLabel col=FileLabel}"
                }, suffix: "5");

            Done();
        }

        [Fact]
        public void SavePipeWithHeader()
        {
            string pathTerms = DeleteOutputPath("SavePipe", "Terms.txt");
            File.WriteAllLines(pathTerms, new string[] {
                "Amer-Indian-Eskimo",
                "Black",
                "Asian-Pac-Islander",
                "White",
            });

            string pathData = GetDataPath("adult.train");
            TestCore(pathData, false,
                new[] {
                    "loader=Text{header+ sep=comma col=Label:14 col=Age:0 col=Gender:TX:9 col=Mar:TX:5 col=Race:TX:8 col=Num:2,4,10-12 col=Txt:TX:~}",
                    "xf=Cat{col=Race2:Key:Race data={" + pathTerms + "} termCol=Whatever}",
                    "xf=Cat{col=Gender2:Gender terms=Male,Female}",
                    "xf=Cat{col=Mar2:Mar col={name=Race3 src=Race terms=Other,White,Black,Asian-Pac-Islander,Amer-Indian-Eskimo}}",
                });

            Done();
        }

        [Fact]
        public void SavePipeKeyToVec()
        {
            string pathTerms = DeleteOutputPath("SavePipe", "Terms.txt");
            File.WriteAllLines(pathTerms, new string[] {
                "Black",
                "White",
                "Male",
                "Female"
            });

            string pathData = GetDataPath("adult.test");
            TestCore(pathData, true,
                new[] {
                    "loader=Text{header+ sep=comma col=Mar:TX:5 col=Race:TX:8 col=Gen:TX:8~9}",
                    "xf=Concat{col=Comb:Race,Gen,Race}",
                    "xf=Cat{kind=Key col=MarKey:Mar}",
                    "xf=Cat{kind=Key col={name=CombKey src=Comb} data={" + pathTerms + "}}",
                    "xf=Convert{col=MarKeyU8:U8:MarKey col=CombKeyU1:U1:CombKey}",
                    "xf=KeyToVector{col={name=CombBagVec src=CombKey bag+} col={name=CombIndVec src=CombKey} col=MarVec:MarKey}",
                    "xf=KeyToVector{col={name=CombBagVecU1 src=CombKeyU1 bag+} col={name=CombIndVecU1 src=CombKeyU1} col=MarVecU8:MarKeyU8}",
                    "xf=ChooseColumns{col=MarKey col=CombKey col=MarVec col=MarVecU8 col=CombBagVec col=CombBagVecU1 col=CombIndVec col=CombIndVecU1 col=Mar col=Comb}",
                },

                pipe =>
                {
                    // Verify that the Vec columns match the corresponding VecXX columns. This verifies that conversion
                    // happened correctly in KeyToVector.
                    using (var c = pipe.GetRowCursor(col => true))
                    {
                        var cols = new[] { "MarVec", "MarVecU8", "CombBagVec", "CombBagVecU1", "CombIndVec", "CombIndVecU1" };
                        var getters = new ValueGetter<VBuffer<Float>>[cols.Length];
                        for (int i = 0; i < cols.Length; i++)
                        {
                            int col;
                            if (!Check(c.Schema.TryGetColumnIndex(cols[i], out col), "{0} not found!", cols[i]))
                                return;
                            getters[i] = c.GetGetter<VBuffer<Float>>(col);
                        }

                        Func<Float, Float, bool> fn = (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y);
                        var v1 = default(VBuffer<Float>);
                        var v2 = default(VBuffer<Float>);
                        while (c.MoveNext())
                        {
                            for (int i = 0; i < cols.Length; i += 2)
                            {
                                getters[i](ref v1);
                                getters[i + 1](ref v2);
                                Check(CompareVec(ref v1, ref v2, v1.Length, fn), "Mismatch");
                            }
                        }
                    }
                });

            Done();
        }

        [Fact]
        public void SavePipeConcatUnknownLength()
        {
            string pathData = DeleteOutputPath("SavePipe", "ConcatUnknownLength.txt");
            File.WriteAllLines(pathData, new string[] {
                "10,11,12,20,a b c,1,2,3",
                "13,14,15,21,d e,4,5",
                "16,17,18,22,f,6"
            });

            TestCore(pathData, true,
                new[] {
                    "loader=Text{col=Known:I4:0-2 col=Single:I4:3 col=Text:TX:4 col=Unknown:I4:~** sep=comma}",
                    // Tokenize Text, then run it through Categorical to get key values, then through KeyToVector.
                    // Then convert everything to R8 and concatenate it all.
                    "xf=WordToken{col=Tokens:Text}",
                    "xf=Cat{col=Keys:Tokens kind=Key}",
                    "xf=KeyToVector{col=Indicators:Keys bag-}",
                    "xf=Convert{col=Indicators type=R8}",
                    "xf=Convert{col=Known col=Single col=Unknown type=R8}",
                    "xf=Concat{col=All:Indicators,Known,Single,Unknown}",
                    "xf=ChooseColumns{col=All}"
                });

            Done();
        }

        [Fact]
        public void SavePipeNgram()
        {
            TestCore(null, true,
                new[] {
                    "loader=Text{col=Label:Num:0 col=Text:TX:1-9}",
                    "xf=Cat{max=5 col={name=Key src=Text kind=key}}",
                    "xf=Ngram{ngram=3 skips=1 col={name=Ngrams1 src=Key max=10}}",
                    "xf=Ngram{skips=2 col={name=Ngrams2 src=Key ngram=4 max=10:20:30} col={name=Ngrams3 src=Key ngram=3 max=10:15}}",
                    "xf=Ngram{ngram=3 col={name=Ngrams4 src=Key max=7}}",
                    "xf=Convert{col=KeyU4:U4:Key}",
                    "xf=Ngram{ngram=3 col={name=Ngrams5 src=KeyU4 max=8}}",
                    "xf=Ngram{ngram=2 col={name=Ngrams6 src=Key}}",
                    "xf=Ngram{ngram=3 col={name=Ngrams7 src=Key all=- max=5}}",
                });

            Done();
        }

        [Fact]
        public void SavePipeNgramSparse()
        {
            string pathData = DeleteOutputPath("SavePipe", "NgramSparse.txt");
            File.WriteAllLines(pathData,
                new[] {
                    "21\t10:a\t12:b\t15:c",
                    "21\t4:a\t10:b\t11:c\t17:a",
                    "21\t2:b\t10:c\t20:d",
                });

            TestCore(pathData, true,
                new[] {
                    "loader=Text{col=Text:TX:0-20}",
                    "xf=Cat{col={name=Key src=Text kind=key}}",
                    "xf=Ngram{ngram=3 skips=2 col={name=Ngrams src=Key max=100}}",
                });

            Done();
        }

        [Fact]
        public void SavePipeConcatWithAliases()
        {
            string pathData = GetDataPath("breast-cancer-withheader.txt");
            TestCore(pathData, true,
                new[]
                {
                    "loader=Text{header+ col=A:0 col=B:1-9}",
                    "xf=Concat{col={name=All source[First]=A src=A source[Rest]=B}}",
                    "xf=Concat{col={name=All2 source=A source=B source[B]=B source[Vector]=B}}",
                    "xf=DropColumns{col=A col=B}"
                });
            Done();
        }

        private static string[] _small20NewsGroupSample =
        {
            "SPORT.BASEBALL\tWhen does Fred McGriff of the Padres become a free agent?",
            "SCI.MED\tIs erythromycin effective in treating pneumonia?"
        };

        private static string[] _small20NewsGroupDict =
        {
            "sport", "baseball", "fred", "mcgriff", "padres", "free", "agent", "med", "erythromycin", "treating",
            "pneumonia"
        };

        [Fact(Skip = "")]
        public void SavePipeTermDictionary()
        {
            string dataFile = DeleteOutputPath("SavePipe", "TermDictionary-SampleText.txt");
            File.WriteAllLines(dataFile, _small20NewsGroupSample);

            string dictFile = DeleteOutputPath("SavePipe", "TermDictionary-SampleDict.txt");
            File.WriteAllLines(dictFile, _small20NewsGroupDict);

            string textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{data={0}}} norm=None charExtractor={{}} wordExtractor=Ngram{{ngram=2}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, dictFile),
                    "xf=ChooseColumns{col=Features}"
                }, suffix: "Ngram");

            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{data={0}}} norm=None charExtractor={{}} wordExtractor=NgramHash{{bits=5}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, dictFile),
                    "xf=ChooseColumns{col=Features}"
                }, suffix: "NgramHash");


            string terms = "sport,baseball,mcgriff,padres,agent,med,erythromycin,pneumonia";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0}}} norm=None charExtractor={{}} wordExtractor=Ngram{{ngram=2}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=ChooseColumns{col=Features}"
                }, suffix: "NgramTerms");

            terms = "sport,baseball,padres,med,erythromycin";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0} dropna+}} norm=None charExtractor={{}} wordExtractor=NgramHash{{ih=3 bits=4}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=ChooseColumns{col=Features}"
                }, suffix: "NgramHashTermsDropNA");

            terms = "sport,baseball,mcgriff,med,erythromycin";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0} dropna+}} norm=linf charExtractor={{}} wordExtractor=Ngram{{ngram=2}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=ChooseColumns{col=Features}"
                }, suffix: "NgramTermsDropNA");

            terms = "hello";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0} dropna+}} norm=linf charExtractor={{}} wordExtractor=Ngram{{ngram=2}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=ChooseColumns{col=T1 col=T2 col=Features}"
                }, suffix: "EmptyNgramTermsDropNA");

            Done();
        }


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

            var col = new HashTransformer.Column();
            col.Name = "F1";
            col.HashBits = 5;
            col.Seed = 42;
            var args = new HashTransformer.Arguments();
            args.Column = new HashTransformer.Column[] { col };

            var hashTransform = HashTransformer.Create(Env, args, srcView);
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
            var col = new HashTransformer.Column();
            col.Name = "F1V";
            col.HashBits = 5;
            col.Seed = 42;
            var args = new HashTransformer.Arguments();
            args.Column = new HashTransformer.Column[] { col };

            var hashTransform = HashTransformer.Create(Env, args, srcView);
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
