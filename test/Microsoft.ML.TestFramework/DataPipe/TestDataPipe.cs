// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Xunit;

namespace Microsoft.ML.RunTests
{
    public sealed partial class TestDataPipe : TestDataPipeBase
    {
        private static float[] dataFloat = new float[] { -0.0f, 0,  1, -1,  2, -2, Single.NaN, Single.MinValue,
                Single.MaxValue, Single.Epsilon, Single.NegativeInfinity, Single.PositiveInfinity };
        private static uint[] resultsFloat = new uint[] { 21, 21, 16, 16, 31, 17, 0, 23, 24, 15, 10, 7 };

        private static VBuffer<Single> dataFloatSparse = new VBuffer<Single>(5, 3, new float[] { -0.0f, 0, 1 }, new[] { 0, 3, 4 });
        private static uint[] resultsFloatSparse = new uint[] { 21, 21, 21, 21, 16 };

        private static Double[] dataDouble = new Double[]   { -0.0, 0, 1, -1,  2, -2, Double.NaN, Double.MinValue,
                Double.MaxValue, Double.Epsilon, Double.NegativeInfinity, Double.PositiveInfinity };
        private static uint[] resultsDouble = new uint[] { 21, 21, 31, 17, 10, 15, 0, 16, 21, 15, 6, 30 };

        private static VBuffer<Double> dataDoubleSparse = new VBuffer<Double>(5, 3, new double[] { -0.0, 0, 1 }, new[] { 0, 3, 4 });
        private static uint[] resultsDoubleSparse = new uint[] { 21, 21, 21, 21, 31 };

        [Fact()]
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
                    "xf=SelectColumns{keepcol=RawLabel keepcol=AutoLabel keepcol=StringLabel keepcol=FileLabel hidden=-}"
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
                    "xf=SelectColumns{keepcol=RawLabel keepcol=FileLabel hidden=-}"
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
                    "xf=SelectColumns{keepcol=RawLabel keepcol=FileLabel hidden=-}"
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
                    "xf=SelectColumns{keepcol=RawLabel keepcol=FileLabel hidden=-}"
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
            using (_env.RedirectChannelOutput(writer, writer))
            {
                TestCore(pathData, true,
                    new[] {
                            "loader=Text{col=RawLabel:TXT:0 col=Names:TXT:1-2 col=Features:TXT:3-4 header+}",
                            string.Format("xf=TermLookup{{key=- col=FileLabelNum:RawLabel data={{{0}}}}}", mappingPathData),
                            string.Format("xf=TermLookup{{col=FileLabelKey:RawLabel data={{{0}}}}}", mappingPathData),
                            "xf=SelectColumns{keepcol=RawLabel keepcol=FileLabelNum keepcol=FileLabelKey hidden=-}"
                    }, suffix: "4");
                writer.WriteLine(ProgressLogLine);
                _env.PrintProgress();
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
                    "xf=SelectColumns{keepcol=RawLabel keepcol=FileLabel hidden=-}"
                }, suffix: "5");

            Done();
        }

        [Fact]
        public void SavePipeWithHeader()
        {
            string pathTerms = DeleteOutputPath("SavePipe", "Terms.txt");
            File.WriteAllLines(pathTerms, new string[] {
                "Amer-Indian-Inuit",
                "Black",
                "Asian-Pac-Islander",
                "White",
            });

            string pathData = GetDataPath("adult.tiny.with-schema.txt");
            TestCore(pathData, false,
                new[] {
                    "loader=Text{sparse+ header+ col=Label:0 col=Age:9 col=Gender:TX:7 col=Mar:TX:3 col=Race:TX:6 col=Num:10-14 col=Txt:TX:~}",
                    "xf=Cat{col=Race2:Key:Race data={" + pathTerms + "} termCol=Whatever}",
                    "xf=Cat{col=Gender2:Gender terms=Male,Female}",
                    "xf=Cat{col=Mar2:Mar col={name=Race3 src=Race terms=Other,White,Black,Asian-Pac-Islander,Amer-Indian-Inuit}}",
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

            string pathData = GetDataPath("adult.tiny.with-schema.txt");
            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ header+ col=Mar:TX:3 col=Race:TX:6 col=Gen:TX:7~8}",
                    "xf=Concat{col=Comb:Race,Gen,Race}",
                    "xf=Cat{kind=Key col=MarKey:Mar}",
                    "xf=Cat{kind=Key col={name=CombKey src=Comb} data={" + pathTerms + "}}",
                    "xf=Convert{col=MarKeyU8:U8:MarKey col=CombKeyU1:U1:CombKey}",
                    "xf=KeyToVector{col={name=CombBagVec src=CombKey bag+} col={name=CombIndVec src=CombKey} col=MarVec:MarKey}",
                    "xf=KeyToVector{col={name=CombBagVecU1 src=CombKeyU1 bag+} col={name=CombIndVecU1 src=CombKeyU1} col=MarVecU8:MarKeyU8}",
                    "xf=SelectColumns{keepcol=MarKey keepcol=CombKey keepcol=MarVec keepcol=MarVecU8 keepcol=CombBagVec keepcol=CombBagVecU1 keepcol=CombIndVec keepcol=CombIndVecU1 keepcol=Mar keepcol=Comb}",
                },

                pipe =>
                {
                    // Verify that the Vec columns match the corresponding VecXX columns. This verifies that conversion
                    // happened correctly in KeyToVector.
                    using (var c = pipe.GetRowCursorForAllColumns())
                    {
                        var cols = new[] { "MarVec", "MarVecU8", "CombBagVec", "CombBagVecU1", "CombIndVec", "CombIndVecU1" };
                        var getters = new ValueGetter<VBuffer<float>>[cols.Length];
                        for (int i = 0; i < cols.Length; i++)
                        {
                            var col = c.Schema.GetColumnOrNull(cols[i]);
                            if (!Check(col.HasValue, "{0} not found!", cols[i]))
                                return;
                            getters[i] = c.GetGetter<VBuffer<float>>(col.Value);
                        }

                        Func<float, float, bool> fn = (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y);
                        var v1 = default(VBuffer<float>);
                        var v2 = default(VBuffer<float>);
                        while (c.MoveNext())
                        {
                            for (int i = 0; i < cols.Length; i += 2)
                            {
                                getters[i](ref v1);
                                getters[i + 1](ref v2);
                                Check(CompareVec(in v1, in v2, v1.Length, fn), "Mismatch");
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
                    "loader=Text{sparse+ col=Known:I4:0-2 col=Single:I4:3 col=Text:TX:4 col=Unknown:I4:~** sep=comma}",
                    // Tokenize Text, then run it through Categorical to get key values, then through KeyToVector.
                    // Then convert everything to R8 and concatenate it all.
                    "xf=WordToken{col=Tokens:Text}",
                    "xf=Cat{col=Keys:Tokens kind=Key}",
                    "xf=KeyToVector{col=Indicators:Keys bag-}",
                    "xf=Convert{col=Indicators type=R8}",
                    "xf=Convert{col=Known col=Single col=Unknown type=R8}",
                    "xf=Concat{col=All:Indicators,Known,Single,Unknown}",
                    "xf=SelectColumns{keepcol=All}"
                });

            Done();
        }

        [Fact]
        public void SavePipeNgram()
        {
            TestCore(null, true,
                new[] {
                    "loader=Text{quote+ sparse+ col=Label:Num:0 col=Text:TX:1-9}",
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
                    "loader=Text{quote+ sparse+ col=Text:TX:0-20}",
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
                    "xf=SelectColumns{dropcol=A dropcol=B}"
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

        [Fact(Skip = "Possible bug in HashTransform, needs some more investigation.")]
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
                    "xf=SelectColumns{keepcol=Features}"
                }, suffix: "Ngram");

            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{data={0}}} norm=None charExtractor={{}} wordExtractor=NgramHash{{bits=5}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, dictFile),
                    "xf=SelectColumns{keepcol=Features}"
                }, suffix: "NgramHash");


            string terms = "sport,baseball,mcgriff,padres,agent,med,erythromycin,pneumonia";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0}}} norm=None charExtractor={{}} wordExtractor=Ngram{{ngram=2}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=SelectColumns{keepcol=Features}"
                }, suffix: "NgramTerms");

            terms = "sport,baseball,padres,med,erythromycin";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0} dropna+}} norm=None charExtractor={{}} wordExtractor=NgramHash{{ih=3 bits=4}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=SelectColumns{keepcol=Features}"
                }, suffix: "NgramHashTermsDropNA");

            terms = "sport,baseball,mcgriff,med,erythromycin";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0} dropna+}} norm=linf charExtractor={{}} wordExtractor=Ngram{{ngram=2}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=SelectColumns{keepcol=Features}"
                }, suffix: "NgramTermsDropNA");

            terms = "hello";
            textSettings =
                "xf=Text{{col=Features:T1,T2 num- punc- dict={{terms={0} dropna+}} norm=linf charExtractor={{}} wordExtractor=Ngram{{ngram=2}}}}";
            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T1:TX:0 col=T2:TX:1}",
                    string.Format(textSettings, terms),
                    "xf=SelectColumns{keepcol=T1 keepcol=T2 keepcol=Features}"
                }, suffix: "EmptyNgramTermsDropNA");

            Done();
        }

        [Fact]
        public void SavePipeCat()
        {
            TestCore(null, false,
                new[] {
                    "loader=Text{quote+ sparse+ col=Text:TX:1-9 col=OneText:TX:1 col=Label:0}",
                    "xf=Cat{max=5 col={name=Bag src=Text kind=bag} col=One:indicator:OneText}",
                    "xf=Cat{max=7 col=Hot:Text}",
                    "xf=Cat{max=8 col=Key:kEY:Text col=KeyOne:KeY:OneText}",
                });

            Done();
        }

        [Fact()]
        public void SavePipeHash()
        {
            string pathData = DeleteOutputPath("SavePipe", "HashTransform.txt");
            File.WriteAllLines(pathData,
                new[] {
                    "1\t2\t3\t4\t3\t5",
                    "2\t3\t2\t4\t7\t10\t11",
                    "1\t1\t2\t3",
                    "10\t2:1\t7:2\t9:3"
                });

            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ col=Text:TX:0-2 col=CatU1:U1[0-2]:0-2 col=CatU2:U2[0-4]:0-2 col=CatU8:U8[]:0-2 col=OneU1:U1[]:0 col=OneU2:U2[]:1 col=OneU4:U4[]:1 col=OneU8:U8[]:2 col=Single:TX:0 col=VarU1:U1[]:3-** col=VarU2:U2[]:3-** col=VarU4:U4[]:3-** col=VarU8:U8[]:3-** col=Variable:TX:3-**}",
                    "xf=Cat{col=Cat:Key:Text col=VarCat:Key:Variable}",
                    "xf=Hash{bits=6 ordered+ col={name=Hash0 src=Text bits=4} col={name=Hash1 src=Text ord- bits=4} col={name=Hash2 src=Cat} col=Hash3:CatU8}",
                    "xf=Hash{col={name=Hash4 bits=5 src=CatU1} col={name=Hash5 src=CatU2 bits=6 ord+} col={name=Hash6 src=CatU2 bits=6} col={name=Hash7 src=CatU8 bits=6} col={name=Hash8 src=Cat bits=6}}",
                    "xf=Hash{col={name=Hash9 bits=5 src=OneU1} col={name=Hash10 bits=8 src=OneU2} col={name=Hash11 bits=3 src=OneU4} col={name=Hash12 bits=3 src=OneU8}}",
                    "xf=Hash{bits=7 ordered+ col={name=VarHash1 src=Variable} col={name=VarHash2 src=VarCat ordered-}}",
                    "xf=Hash{bits=7 ordered+ col={name=VarHash3 src=VarU1} col={name=VarHash4 src=VarU2} col={name=VarHash5 src=VarU4} col={name=VarHash6 src=VarU8}}",
                    "xf=Hash{bits=4 col={name=SingleHash src=Single ordered+}}",
                    "xf=Concat{col=VarComb:VarHash1,VarHash2,VarHash3,VarHash4,VarHash5,VarHash6}",
                    "xf=SelectColumns{keepcol=SingleHash keepcol=Hash0 keepcol=Hash1 keepcol=Hash2 keepcol=Hash3 keepcol=Hash4 keepcol=Hash5 keepcol=Hash6 keepcol=Hash7 keepcol=Hash8 keepcol=Hash9 keepcol=Hash10 keepcol=Hash11 keepcol=Hash12 keepcol=VarComb}",
                }, logCurs: true);

            Done();
        }

        [Fact]
        public void SavePipeCountSelect()
        {
            TestCore(null, false,
                new[] {
                    "loader=Text{quote+ sparse+ col=One:TX:1 col=Num:R4:2-* col=Key:U1[0-10]:1}",
                    // Create a lot of unused slots.
                    "xf=CatHash{col=OneInd:One bits=10}",
                    // One is for the non-vector case and OneInd is reduced to a small size.
                    "xf=CountFeatureSelection{col=Num col=One col=OneInd count=1}",
                    // This tests the path where a copycolumn transform is created.
                    "xf=CountFeatureSelection{col=Num col=One col=OneInd count=1}",
                    // This tests counts greater than 1
                    "xf=KeyToVector{col=Key}",
                    "xf=CountFeatureSelection{col=Key count=100}",
                });

            Done();
        }

        [Fact(Skip = "Should be enabled after NAHandle is converted to use SelectColumnsTransform instead of DropColumnsTransform")]
        public void SavePipeCountSelectWithSparse()
        {
            TestCore(null, false,
                new[] {
                    "loader=Text{col=One:TX:1 col=Num:R4:2-* col=Key:U1[1-10]:1}",
                    // Create a lot of unused slots.
                    "xf=CatHash{col=OneInd:One bits=10}",
                    // Create more unused slots and test the sparse case.
                    "xf=NAHandle{col=NumSparse:Num}",
                    // This tests that Num and NumSparse remain the same,
                    // One is for the non-vector case and OneInd is reduced to a small size.
                    "xf=CountFeatureSelection{col=Num col=NumSparse col=One col=OneInd count=1}",
                    // This tests the path where a no-op transform is created.
                    "xf=CountFeatureSelection{col=Num col=NumSparse col=One col=OneInd count=1}",
                    // This tests counts greater than 1
                    "xf=KeyToVector{col=Key}",
                    "xf=CountFeatureSelection{col=Key count=100}",
                });

            Done();
        }

        private bool VerifyMatch<TSrc, TDst>(TSrc src, TDst dst, ValueMapper<TSrc, TDst> conv, ValueMapper<TDst, TSrc> convBack)
            where TSrc : struct
            where TDst : struct
        {
            TDst v = default(TDst);
            conv(in src, ref v);
            if (EqualityComparer<TDst>.Default.Equals(dst, v))
                return true;
            TSrc vSrc = default;
            convBack(in v, ref vSrc);
            if (EqualityComparer<TDst>.Default.Equals(dst, default(TDst)) && !EqualityComparer<TSrc>.Default.Equals(src, vSrc))
                return true;
            Fail($"Values different values in VerifyMatch<{typeof(TSrc).Name}, {typeof(TDst).Name}>: converted from {typeof(TSrc).Name} to {typeof(TDst).Name}: {v}. Parsed from text: {dst}");
            return false;
        }

        [Fact]
        public void SavePipeNgramHash()
        {
            string pathData = GetDataPath("lm.sample.txt");
            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ header+ col=Label:TX:0 col=Attrs:TX:1-2 col=TextFeatures:TX:3-4 rows=100}",
                    "xf=WordToken{col={name=Tokens src=TextFeatures}}",
                    "xf=Cat{max=10 col={name=Cat src=Tokens kind=key}}",
                    "xf=Hash{col={name=Hash src=Tokens bits=10} col={name=HashBig src=Tokens bits=31}}",
                    "xf=NgramHash{col={name=NgramHashOne src=Cat bits=4 ngram=3 skips=2}}",
                    "xf=NgramHash{col={name=HashNgram1 src=Cat src=Cat bits=10 ngram=3 skips=1}}",
                    "xf=NgramHash{ngram=3 bits=8 col={name=HashNgram2 src=Hash src=Hash skips=1 ord-} col={name=HashNgram3 src=Cat src=Hash skips=2 ord- rehash+ all-}}",
                    "xf=NgramHash{bits=6 col=HashNgram4:HashBig,Hash rehash+}",
                    "xf=NgramHash{bits=3 ngram=1 col={name=HashNgram5 src=Hash src=Hash} col={name=HashNgram6 src=Hash ord-}}",
                    "xf=NgramHash{bits=6 col=HashNgram7:HashBig,Hash rehash+ all- col={name=HashNgram8 src=Hash all+ ord-}}",
                    "xf=SelectColumns{keepcol=NgramHashOne keepcol=HashNgram1 keepcol=HashNgram2 keepcol=HashNgram3 keepcol=HashNgram4 keepcol=HashNgram5 keepcol=HashNgram6 keepcol=HashNgram7 keepcol=HashNgram8 hidden=-}",
                });

            TestCore(null, true,
                new[] {
                    "loader=Text{col=CatU8:U8[0-100]:1-9 col=CatU2:U2[]:3-5}",
                    "xf=NgramHash{bits=5 col=NgramHash:CatU8 col=NgramHash2:CatU2}",
                    "xf=SelectColumns{keepcol=NgramHash keepcol=NgramHash2 hidden=-}"
                },
                suffix: "-Convert");

            Done();
        }

        [Fact]
        public void SavePipeWordTokenize()
        {
            TestCore(GetDataPath("lm.sample.txt"), false,
                new[]
                {
                    "loader=Text{col=A:TX:2 col=B:TX:3}",
                    "xf=wordToken{col={name=C source=A sep=space,-} col=D:B}",
                    "xf=concat{col=Concat:C,D}",
                    "xf=Select{dropcol=C dropcol=D}"
                });
            Done();
        }

        [Fact]
        public void SavePipeWordHash()
        {
            string pathData = GetDataPath(@"lm.sample.txt");
            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ header+ col=One:TX:4 col=Two:TX:3 rows=101}",
                    "xf=WordHashBag{bits=5",
                    "  col=F11:5:One col={name=F12 src=One ngram=4} col={name=F13 src=Two ngram=3 skips=2 bits=15}",
                    "  col=F21:Two,One col={name=F22 src=Two src=One ngram=4} col={name=F23 src=Two src=One bits=15 ngram=3 skips=2}",
                    "  col={name=F31 src=Two src=One ord-} col={name=F32 src=Two src=One ngram=4 ord-} col={name=F33 src=Two src=One ngram=3 skips=2 ord-}",
                    "}",
                    "xf=SelectColumns{keepcol=F21 keepcol=F22 keepcol=F23 keepcol=F31 keepcol=F32 keepcol=F33 keepcol=F11 keepcol=F12 keepcol=F13 hidden=-}",
                },
                (pipe) =>
                {
                    // Column F13 contains the ngram counts of column Two, and column F23 contains the ngram counts
                    // of columns Two and One. Therefore, make sure that the ngrams in column Two were hashed to the same 
                    // slots in F13 as they were in column F23. We do this by checking that for every slot, F23 is >= F13.
                    using (var c = pipe.GetRowCursorForAllColumns())
                    {
                        var col1 = c.Schema.GetColumnOrNull("F13");
                        if (!Check(col1.HasValue, "Column F13 not found!"))
                            return;
                        var col2 = c.Schema.GetColumnOrNull("F23");
                        if (!Check(col2.HasValue, "Column F23 not found!"))
                            return;

                        var get1 = c.GetGetter<VBuffer<float>>(col1.Value);
                        var get2 = c.GetGetter<VBuffer<float>>(col2.Value);
                        VBuffer<float> bag1 = default;
                        VBuffer<float> bag2 = default;
                        while (c.MoveNext())
                        {
                            get1(ref bag1);
                            get2(ref bag2);
                            if (!CompareVec(in bag1, in bag2, bag1.Length, (x1, x2) => x1 <= x2))
                            {
                                Fail("Values don't match in columns F13, F23");
                                return;
                            }
                        }
                    }
                });

            Done();
        }

        [Fact]
        public void SavePipeWordHashUnordered()
        {
            string pathData = GetDataPath(@"lm.sample.txt");
            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ header+ col=One:TX:4 col=Two:TX:3 rows=101}",
                    "xf=WordHashBag{bits=5 ord=- col=F1:One col=F2:One,One}",
                    "xf=SelectColumns{keepcol=F1 keepcol=F2}",
                },
                (pipe) =>
                {
                    // Verify that F2 = 2 * F1
                    using (var c = pipe.GetRowCursorForAllColumns())
                    {
                        var col1 = c.Schema.GetColumnOrNull("F1");
                        if (!Check(col1.HasValue, "Column F1 not found!"))
                            return;
                        var col2 = c.Schema.GetColumnOrNull("F2");
                        if (!Check(col2.HasValue, "Column F2 not found!"))
                            return;

                        var get1 = c.GetGetter<VBuffer<float>>(col1.Value);
                        var get2 = c.GetGetter<VBuffer<float>>(col2.Value);
                        VBuffer<float> bag1 = default;
                        VBuffer<float> bag2 = default;
                        while (c.MoveNext())
                        {
                            get1(ref bag1);
                            get2(ref bag2);
                            if (!CompareVec(in bag1, in bag2, bag1.Length, (x1, x2) => 2 * x1 == x2))
                            {
                                Fail("Values don't match");
                                return;
                            }
                        }
                    }
                });

            Done();
        }

        [Fact]
        public void SavePipeInvertHash()
        {
            string pathData = DeleteOutputPath("SavePipe","InvertHash-Data.txt");
            // Four columns. First "A" with words starting with "a" (for easy identification), second
            // "K" with an explicit key type, third "E" a column that has all missing values, and fourth
            // "B" with words starting with "b".
            File.WriteAllLines(pathData,
                new[]
                {
                    "annie ate an ant\t5\t4\t\tbob bakes brownies",
                    "an angry ant\t3\t3\t\tbob bowled badly",
                    "\t10\t\t\t\"\""
                });
            const string loader = "loader=Text{quote+ sparse+ col=A:TX:0 col=K:U4[11]:1-2 col=KS:U4[11]:2 col=B:TX:4 col=E:TX:3}";
            TestCore(pathData, true,
                new[] {
                    loader,
                    "xf=WordHashBag{bits=3 ord- col=F1:A,B col=F2:B,A ih=-1 ngram=2}",
                    "xf=WordHashBag{bits=3 ord- col=F3:A,B col=F4:B,A ih=1 ngram=2}",
                    "xf=SelectColumns{keepCol=A keepCol=B keepCol=F1 keepCol=F2 keepCol=F3 keepCol=F4}"
                }, suffix: "1");
            // Same, but using per column overrides, including one column without inversion.
            TestCore(pathData, true,
                new[] {
                    loader,
                    "xf=WordHashBag{bits=3 ord- col=F1:A,B col=F2:B,A ih=-1 ngram=2 col={name=F3 src=A src=B ih=1}  col={name=F4 src=B src=A ih=1} col={name=F5 src=A src=B ih=0}}",
                    "xf=SelectColumns{keepCol=A keepCol=B keepCol=F1 keepCol=F2 keepCol=F3 keepCol=F4 keepCol=F5}"
                }, suffix: "2");
            // Do to the key column.
            TestCore(pathData, true,
                new[] {
                    loader,
                    "xf=CatHash{ih=-1 col=KH:2:K col={name=KHU bits=2 src=K ordered-}}",
                    "xf=SelectColumns{keepCol=K keepCol=KH keepCol=KHU}"
                }, suffix: "3");
            // Do to the key column combining it with the text column.
            TestCore(pathData, true,
                new[] {
                    loader,
                    "xf=WordToken{col=AT:A}",
                    "xf=Hash{bits=2 ih=-1 col=AH:AT col=KH:K}",
                    "xf=NGramHash{bits=3 ih=-1 col=N3:AH,KH seed=2}",
                    "xf=NGramHash{bits=10 ih=-1 col=N10:AH,KH seed=2}",
                    "xf=SelectColumns{keepCol=A keepCol=K keepCol=KH keepCol=N3 keepCol=N10}"
                }, suffix: "4");
            // Do for scalar non-vector columns.
            TestCore(pathData, true,
                new[] {
                    loader,
                    "xf=CatHash{bits=3 ih=-1 col=AH:A col=KH:KS}",
                    "xf=SelectColumns{keepCol=A keepCol=KS keepCol=AH keepCol=KH}"
                }, suffix: "5");

            // Do with full-length grams only.
            TestCore(pathData, true,
                new[] {
                    loader,
                    "xf=WordToken{col=AT:A}",
                    "xf=Hash{col=AH:AT bits=30}",
                    "xf=NgramHash{col=AH ngram=3 bits=4 all- ih=3}",
                    "xf=SelectColumns{keepCol=AH}"
                }, suffix: "6");

            Done();
        }

        [Fact]
        public void SavePipeWordBag()
        {
            string pathData = GetDataPath(@"lm.sample.txt");
            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ header+ col=Label:TX:0 col=One:TX:4 col=Vec:TX:3,4 rows=101}",
                    "xf=AutoLabel{col=Label}",
                    "xf=WordBag{max=10",
                    "  col=F11:One col={name=F12 src=One ngram=4 max=3 max=4 max=5} col={name=F13 src=One ngram=3 skips=2}",
                    "  col=F21:Vec col={name=F22 src=Vec max=20 ngram=4}}",
                    "xf=WordBag{col={name=F23 src=Vec max=10 ngram=3 skips=2}}",
                    "xf=SelectColumns{keepCol=Label keepCol=F21 keepCol=F22 keepCol=F23 keepCol=F11 keepCol=F12 keepCol=F13}",
                });

            Done();
        }

        [Fact]
        public void SavePipeWordBagTfIdf()
        {
            string pathData = DeleteOutputPath("SavePipe", "Sample-Data.txt");
            File.WriteAllLines(pathData,
                new[]
                {
                    "Text",
                    "A B C D",
                    "E F G H",
                    "E A",
                    "E F G H K L A B",
                    "A B"
                });

            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ header=+ col=Text:TX:0}",
                    "xf=WordBag{col={name=TfIdf src=Text max=5 ngram=3 weighting=TfIdf}}",
                    "xf=SelectColumns{keepCol=TfIdf}",
                });

            Done();
        }

        [Fact]
        public void SavePipeWordBagManyToOne()
        {
            string pathData = GetDataPath(@"lm.sample.txt");
            TestCore(pathData, true,
                new[] {
                    "loader=Text{sparse+ header+ col=One:TX:4 col=Vec:TX:3,4 rows=101}",
                    "xf=WordBag{col={name=WB1 src=One max=10 ngram=3 skips=2} col={name=WB2 src=One src=One max=10 ngram=3 skips=2}}",
                    "xf=SelectColumns{keepCol=WB1 keepCol=WB2}"
                },
                (pipe) =>
                {
                    // Verify that WB2 = 2 * WB1
                    using (var c = pipe.GetRowCursorForAllColumns())
                    {
                        var b1 = default(VBuffer<float>);
                        var b2 = default(VBuffer<float>);
                        var col1 = c.Schema.GetColumnOrNull("WB1");
                        var col2 = c.Schema.GetColumnOrNull("WB2");
                        if (!col1.HasValue || !col2.HasValue)
                        {
                            Fail("Did not find expected columns");
                            return;
                        }
                        var get1 = c.GetGetter<VBuffer<float>>(col1.Value);
                        var get2 = c.GetGetter<VBuffer<float>>(col2.Value);
                        while (c.MoveNext())
                        {
                            get1(ref b1);
                            get2(ref b2);
                            if (!CompareVec(in b1, in b2, b1.Length, (x1, x2) => 2 * x1 == x2))
                            {
                                Fail("Unexpected values in row {0}", c.Position);
                                break;
                            }
                        }
                    }
                });

            Done();
        }

        [Fact]
        public void SavePipeWithKey()
        {
            var dataPath = GetDataPath("breast-cancer-withheader.txt");
            TestCore(dataPath, true,
                new[] {
                    "loader=Text{header=+",
                    "  col=Label:U1[0-1]:0",
                    "  col=Features:U2:1-*",
                    "  col=A:U1[0-5]:1",
                    "  col=B:U1[0-8]:2",
                    "  col=C:U8[6]:3",
                    "  col=D:U1[]:4",
                    "  col=E:U8[]:5",
                    "  col=F:U1[]:6",
                    "}",
                    "xf=Convert{col=Label2:U2[0-1]:Label col=Features2:Features type=Num}",
                },

                pipe =>
                {
                    var argsText = new TextLoader.Options();
                    bool tmp = CmdParser.ParseArguments(Env,
                        " header=+" +
                        " col=Label:TX:0" +
                        " col=Features:TX:1-*" +
                        " col=A:TX:1" +
                        " col=B:TX:2" +
                        " col=C:TX:3" +
                        " col=D:TX:4" +
                        " col=E:TX:5" +
                        " col=F:TX:6",
                        argsText);
                    Check(tmp, "Parsing argsText failed!");
                    IDataView view2 = TextLoader.Create(Env, argsText, new MultiFileSource(dataPath));

                    var argsConv = new TypeConvertingTransformer.Options();
                    tmp = CmdParser.ParseArguments(Env,
                        " col=Label:U1[0-1]:Label" +
                        " col=Features:U2:Features" +
                        " col=A:U1[0-5]:A" +
                        " col=B:U1[0-8]:B" +
                        " col=C:U8[6]:C" +
                        " col=D:U1[]:D" +
                        " col=E:U8[]:E" +
                        " col=F:U1[]:F",
                        argsConv);
                    Check(tmp, "Parsing argsConv failed!");
                    view2 = TypeConvertingTransformer.Create(Env, argsConv, view2);

                    argsConv = new TypeConvertingTransformer.Options();
                    tmp = CmdParser.ParseArguments(Env,
                        " col=Label2:U2:Label col=Features2:Num:Features",
                        argsConv);
                    Check(tmp, "Parsing argsConv(2) failed!");
                    view2 = TypeConvertingTransformer.Create(Env, argsConv, view2);

                    var colsChoose = new[] { "Label", "Features", "Label2", "Features2", "A", "B", "C", "D", "E", "F" };

                    IDataView view1 = ML.Transforms.SelectColumns(colsChoose).Fit(pipe).Transform(pipe);
                    view2 = ML.Transforms.SelectColumns(colsChoose).Fit(view2).Transform(view2);

                    CheckSameValues(view1, view2);
                },

                logCurs: false);

            Done();
        }

        [Fact]
        public void SavePipeDropColumns()
        {
            string pathData = GetDataPath("adult.tiny.with-schema.txt");
            TestCore(pathData, false,
                new[] {
                    "loader=Text{header+ col=One:TX:9 col=Num:R4:9-14 col=Cat:TX:0~*}",
                    "xf=MinMax{col=Num}",
                    "xf=NAHandle{col=NumSparse:Num}",
                    "xf=MinMax{col=NumSparse}",
                    "xf=SelectColumns{dropcol=NumSparse hidden=+}",
                });

            Done();
        }

        [Fact]
        public void SavePipeCustomStopwordsRemover()
        {
            string dataFile = DeleteOutputPath("SavePipe", "CustomStopwordsRemover-dataFile.txt");
            File.WriteAllLines(dataFile, new[] {
                "When does Fred McGriff of the Padres become a free agent?",
                "Is erythromycin effective in treating pneumonia?"
            });

            var stopwordsList = new[]
                {
                    "When",
                    "does",
                    "of",
                    "the",
                    "Padres",
                    "become",
                    "a",
                    "Is",
                    "effective",
                    "in"
                };
            string stopwordsFile = DeleteOutputPath("SavePipe", "CustomStopwordsRemover-stopwordsFile.txt");
            File.WriteAllLines(stopwordsFile, stopwordsList);

            Action<ILegacyDataLoader> action
                = pipe =>
                {
                    VBuffer<ReadOnlyMemory<char>>[] expected = new VBuffer<ReadOnlyMemory<char>>[2];
                    ReadOnlyMemory<char>[] values = { "Fred".AsMemory(), "McGriff".AsMemory(), "free".AsMemory(), "agent".AsMemory() };
                    expected[0] = new VBuffer<ReadOnlyMemory<char>>(values.Length, values);
                    ReadOnlyMemory<char>[] values1 = { "erythromycin".AsMemory(), "treating".AsMemory(), "pneumonia".AsMemory() };
                    expected[1] = new VBuffer<ReadOnlyMemory<char>>(values1.Length, values1);

                    using (var c = pipe.GetRowCursorForAllColumns())
                    {
                        var col = c.Schema.GetColumnOrNull("T");
                        if (!Check(col.HasValue, "Column T not found!"))
                            return;
                        var getter = c.GetGetter<VBuffer<ReadOnlyMemory<char>>>(col.Value);
                        var buffer = default(VBuffer<ReadOnlyMemory<char>>);
                        int index = 0;
                        while (c.MoveNext())
                        {
                            getter(ref buffer);
                            CompareVec(in buffer, in expected[index++], buffer.GetValues().Length, (s1, s2) => s1.Span.SequenceEqual(s2.Span));
                        }
                    }
                };

            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T:TX:0}",
                    "xf=WordToken{col=T}",
                    "xf=TextNorm{col=T case=None punc=-}",
                    string.Format("xf=CustomStopWords{{data={0} col=T}}", stopwordsFile),
                    "xf=SelectColumns{keepcol=T}"
                }, action, baselineSchema: false);

            TestCore(dataFile, true,
                new[] {
                    "loader=Text{col=T:TX:0}",
                    "xf=WordToken{col=T}",
                    "xf=TextNorm{col=T case=None punc=-}",
                    string.Format("xf=CustomStopWords{{stopwords={0} col=T}}", string.Join(",", stopwordsList)),
                    "xf=SelectColumns{keepcol=T}"
                }, action, baselineSchema: false);

            Done();
        }

        [Fact]
        public void SavePipeTokenizerAndStopWords()
        {
            string dataFile = DeleteOutputPath("SavePipe", "Multi-Languages.txt");
            File.WriteAllLines(dataFile, new[] {
                "1 \"Oh, no,\" she's saying, \"our $400 blender can't handle something this hard!\"	English",
                "2 Vous êtes au volant d'une voiture et vous roulez à grande vitesse	French",
                "3 Lange nichts voneinander gehört! Es freut mich, dich kennen zu lernen	German",
                "4 Goedemorgen, Waar kom je vandaan? Ik kom uit Nederlands	Dutch",
                "5 Ciao, Come va? Bene grazie. E tu? Quanto tempo!	Italian",
                "六 初めまして 良い一日を ごきげんよう！ さようなら	Japanese",
                "6 ¡Hola! ¿Cómo te llamas? Mi nombre es ABELE	Spanish"
            });

            TestCore(dataFile, true,
                new[] {
                    "Loader=Text{col=Source:TXT:0 col=Lang:TXT:1 sep=tab}",
                    "xf=Token{col=SourceTokens:Source}",
                    "xf=StopWords{langscol=Lang col=Output:SourceTokens}"
                }, roundTripText: false);

            Done();
        }

        [Fact]
        public void SavePipeDropNAs()
        {
            string pathData = DeleteOutputPath("SavePipe", "DropNAs.txt");
            File.WriteAllLines(pathData,
                new[]
                {
                    "2,0,|,Hello World!",
                    "3,4,|,",
                    "0,nan,|,Bye all",
                    "7,8,|,Good bye",
                    "?,nan,|,this is a"
                });

            TestCore(pathData, false,
                new[]
                {
                    "loader=Text{header- sep=, col=Num:R4:0-1 col=Sep:TX:2 col=Text:TX:3}",
                    "xf=NADrop{col=NumNAsDropped:Num}",
                    "xf=Token{col=Text}",
                    "xf=Term{col=Text2:Text terms=Hello,all,Good,Bye}",
                    "xf=NADrop{col=TextNAsDropped:Text2}",
                    "xf=Copy{col=Sep2:Sep col=Sep3:Sep}",
                    "xf=Select{keepcol=Num keepcol=Sep keepcol=NumNAsDropped keepcol=Sep2 keepcol=Text keepcol=Sep3 keepcol=TextNAsDropped}"
                }, baselineSchema: false, roundTripText: false);

            Done();
        }

        [TestCategory("DataPipeSerialization")]
        [Fact]
        public void SavePipeTrainAndScoreFccFastTree()
        {
            RunMTAThread(() => TestCore(null, false,
                new[]
                {
                    "loader=Text{sparse+}",
                    "xf=TrainScore{tr=FT scorer=fcc{top=4 bottom=2 str+}}",
                    "xf=Copy{col=ContributionsStr:FeatureContributions}",
                    "xf=TrainScore{tr=FT scorer=fcc{top=3 bottom=3}}"
                }, digitsOfPrecision: 6));

            Done();
        }

        [TestCategory("DataPipeSerialization")]
        [Fact]
        public void SavePipeTrainAndScoreFccTransformStr()
        {
            TestCore(null, false,
                new[]
                {
                    "loader=Text{sparse+} xf=TrainScore{tr=AP{shuf-} scorer=fcc{str+}}"
                }, digitsOfPrecision: 4);

            Done();
        }

        [Fact]
        public void SavePipeLda()
        {
            string pathData = DeleteOutputPath("SavePipe", "Lda.txt");
            File.WriteAllLines(pathData, new string[] {
                "1\t0\t0",
                "0\t1\t0",
                "0\t0\t"
            });
            TestCore(pathData, false,
                new[] {
                    "loader=Text{col=F1V:Num:0-2}",
                    "xf=Lda{col={name=Result src=F1V numtopic=3 alphasum=3 ns=3 reset=+ t=1} summary=+}",
                }, forceDense: true);
            Done();
        }

        [Fact]
        public void TestHashTransformFloat()
        {
            TestHashTransformHelper(dataFloat, resultsFloat, NumberDataViewType.Single);
        }

        [Fact]
        public void TestHashTransformFloatVector()
        {
            var data = new[] { dataFloat };
            var results = new[] { resultsFloat };
            TestHashTransformVectorHelper(data, results, NumberDataViewType.Single);
        }

        [Fact]
        public void TestHashTransformFloatSparseVector()
        {
            var results = new[] { resultsFloatSparse };
            TestHashTransformVectorHelper(dataFloatSparse, results, NumberDataViewType.Single);
        }

        [Fact]
        public void TestHashTransformDoubleSparseVector()
        {
            var results = new[] { resultsDoubleSparse };
            TestHashTransformVectorHelper(dataDoubleSparse, results, NumberDataViewType.Double);
        }

        [Fact]
        public void TestHashTransformDouble()
        {
            TestHashTransformHelper(dataDouble, resultsDouble, NumberDataViewType.Double);
        }

        [Fact]
        public void TestHashTransformDoubleVector()
        {
            var data = new[] { dataDouble };
            var results = new[] { resultsDouble };
            TestHashTransformVectorHelper(data, results, NumberDataViewType.Double);
        }

        private void TestHashTransformHelper<T>(T[] data, uint[] results, NumberDataViewType type)
        {
            var builder = new ArrayDataViewBuilder(Env);

            builder.AddColumn("F1", type, data);
            var srcView = builder.GetDataView();

            var hashTransform = new HashingTransformer(Env, new HashingEstimator.ColumnOptions("F1", "F1", 5, 42)).Transform(srcView);
            using (var cursor = hashTransform.GetRowCursorForAllColumns())
            {
                var resultGetter = cursor.GetGetter<uint>(cursor.Schema[1]);
                uint resultRow = 0;
                foreach (var r in results)
                {
                    Assert.True(cursor.MoveNext());
                    resultGetter(ref resultRow);
                    Assert.True(resultRow == r);
                }
            }
        }

        private void TestHashTransformVectorHelper<T>(T[][] data, uint[][] results, NumberDataViewType type)
        {
            var builder = new ArrayDataViewBuilder(Env);
            builder.AddColumn("F1V", type, data);
            TestHashTransformVectorHelper(builder, results);
        }

        private void TestHashTransformVectorHelper<T>(VBuffer<T> data, uint[][] results, NumberDataViewType type)
        {
            var builder = new ArrayDataViewBuilder(Env);
            builder.AddColumn("F1V", type, data);
            TestHashTransformVectorHelper(builder, results);
        }

        private void TestHashTransformVectorHelper(ArrayDataViewBuilder builder, uint[][] results)
        {
            var srcView = builder.GetDataView();
            var hashTransform = new HashingTransformer(Env, new HashingEstimator.ColumnOptions("F1V", "F1V", 5, 42)).Transform(srcView);
            using (var cursor = hashTransform.GetRowCursorForAllColumns())
            {
                var resultGetter = cursor.GetGetter<VBuffer<uint>>(cursor.Schema[1]);
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

        [Fact]
        public void ArrayDataViewBuilder()
        {
            ArrayDataViewBuilder builder = new ArrayDataViewBuilder(Env);
            const int rows = 100;
            Random rgen = new Random(0);
            float[] values = new float[rows];
            for (int i = 0; i < values.Length; ++i)
                values[i] = (float)(2 * rgen.NextDouble() - 1);
            builder.AddColumn("Foo", NumberDataViewType.Single, values);

            int[][] barValues = new int[rows][];
            const int barSlots = 4;
            for (int i = 0; i < rows; ++i)
            {
                barValues[i] = new int[barSlots];
                for (int j = 0; j < barSlots; ++j)
                    barValues[i][j] = rgen.Next(-100, 100);
            }
            builder.AddColumn("Bar", NumberDataViewType.Int32, barValues);
            bool[] bizValues = new bool[rows];
            for (int i = 0; i < rows; ++i)
                bizValues[i] = (rgen.Next(2) == 1);
            builder.AddColumn("Biz", BooleanDataViewType.Instance, bizValues);

            IDataView view = builder.GetDataView();

            Assert.Equal(3, view.Schema.Count);
            // REVIEW: Generalize schema test.
            Assert.Equal("Foo", view.Schema[0].Name);
            Assert.Equal("Bar", view.Schema[1].Name);
            Assert.Equal("Biz", view.Schema[2].Name);
            int temp;
            Assert.True(view.Schema.TryGetColumnIndex("Foo", out temp));
            Assert.Equal(0, temp);
            Assert.True(view.Schema.TryGetColumnIndex("Bar", out temp));
            Assert.Equal(1, temp);
            Assert.True(view.Schema.TryGetColumnIndex("Biz", out temp));
            Assert.Equal(2, temp);

            // Check the number of rows.
            Assert.True(view.GetRowCount().HasValue);
            Assert.Equal((long)rows, view.GetRowCount().Value);

            using (DataViewRowCursor cursor = view.GetRowCursorForAllColumns())
            {
                var del = cursor.GetGetter<float>(cursor.Schema[0]);
                var del2 = cursor.GetGetter<VBuffer<int>>(cursor.Schema[1]);
                var del3 = cursor.GetGetter<bool>(cursor.Schema[2]);
                float value = 0;
                VBuffer<int> value2 = default(VBuffer<int>);
                bool value3 = default(bool);
                int row = 0;
                while (cursor.MoveNext())
                {
                    // First "Foo" column.
                    del(ref value);
                    Assert.Equal(values[row], value);

                    // Second "Bar" column.
                    del2(ref value2);
                    Assert.Equal(barSlots, value2.Length);
                    Assert.True(value2.IsDense);
                    for (int s = 0; s < barSlots; ++s)
                        Assert.Equal(barValues[row][s], value2.GetValues()[s]);

                    // Third "Biz" column.
                    del3(ref value3);
                    Assert.Equal(bizValues[row], value3);

                    // Non-column cursor data.
                    Assert.Equal((long)row, cursor.Position);
                    Assert.True(row < rows, "row cursor cursor returned more rows than expected");

                    ++row;
                }
                Assert.Equal(rows, row);
            }

            SaveLoadText(view, Env);
            SaveLoad(view, Env);

            SaveLoadText(view, Env, suffix: "NoSchema", roundTrip: false, outputSchema: false, outputHeader: true);
            SaveLoadText(view, Env, suffix: "NoHeader", roundTrip: false, outputSchema: true, outputHeader: false);
            SaveLoadText(view, Env, suffix: "NoSchemaNoHeader", roundTrip: false, outputSchema: false, outputHeader: false);

            SaveLoadTransposed(view, Env);
            SaveLoadTransposed(view, Env, suffix: "2ndSave");

            Done();
        }

        [Fact]
        public void ArrayDataViewBuilderNoRows()
        {
            ArrayDataViewBuilder builder = new ArrayDataViewBuilder(Env);
            builder.AddColumn("Foo", NumberDataViewType.Int32, new int[0]);
            builder.AddColumn("Bar", NumberDataViewType.UInt16, new ushort[0]);

            IDataView view = builder.GetDataView();

            SaveLoadText(view, Env);
            SaveLoad(view, Env);

            Done();
        }

        [Fact]
        public void ArrayDataViewBuilderNoRowsNoCols()
        {
            ArrayDataViewBuilder builder = new ArrayDataViewBuilder(Env);
            IDataView view = builder.GetDataView(0);

            // Text saving by design does not work with no columns.
            bool caught;
            try
            {
                SaveLoadText(view, Env);
                caught = false;
            }
            catch (ArgumentOutOfRangeException exception)
            {
                caught = exception.IsMarked();
            }
            Check(caught, "text save/load should have thrown on no columns, but did not");
            SaveLoad(view, Env);

            Done();
        }

        [Fact]
        public void ArrayDataViewBuilderNoCols()
        {
            ArrayDataViewBuilder builder = new ArrayDataViewBuilder(Env);
            IDataView view = builder.GetDataView(100);

            // Text saving by design does not work with no columns.
            bool caught;
            try
            {
                SaveLoadText(view, Env);
                caught = false;
            }
            catch (ArgumentOutOfRangeException exception)
            {
                caught = exception.IsMarked();
            }
            Check(caught, "text save/load should have thrown on no columns, but did not");
            SaveLoad(view, Env);

            Done();
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
                new[] {  (float)1.0,  (float)0.0,  (float)0.0 },
                new[] {  (float)0.0,  (float)1.0,  (float)0.0 },
                new[] {  (float)0.0,  (float)0.0,  (float)1.0 },
            };

            builder.AddColumn("F1V", NumberDataViewType.Single, data);
            var srcView = builder.GetDataView();

            var opt = new LatentDirichletAllocationEstimator.ColumnOptions(name: "F1V", numberOfTopics: 3,
                numberOfSummaryTermsPerTopic: 3, alphaSum: 3, numberOfThreads: 1, resetRandomGenerator: true);
            var est = ML.Transforms.Text.LatentDirichletAllocation(opt);
            var ldaTransformer = est.Fit(srcView);
            var transformedData = ldaTransformer.Transform(srcView);

            using (var cursor = transformedData.GetRowCursorForAllColumns())
            {
                var resultGetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[1]);
                VBuffer<float> resultFirstRow = new VBuffer<float>();
                VBuffer<float> resultSecondRow = new VBuffer<float>();
                VBuffer<float> resultThirdRow = new VBuffer<float>();

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
        public void TestLdaTransformerEmptyDocumentException()
        {
            var builder = new ArrayDataViewBuilder(Env);
            string colName = "Zeros";
            var data = new[]
            {
                new[] {  (float)0.0,  (float)0.0,  (float)0.0 },
                new[] {  (float)0.0,  (float)0.0,  (float)0.0 },
                new[] {  (float)0.0,  (float)0.0,  (float)0.0 },
            };

            builder.AddColumn(colName, NumberDataViewType.Single, data);

            var srcView = builder.GetDataView();
            try
            {
                var lda = ML.Transforms.Text.LatentDirichletAllocation("Zeros").Fit(srcView).Transform(srcView);
            }
            catch (InvalidOperationException ex)
            {
                Assert.Equal(ex.Message, string.Format("The specified documents are all empty in column '{0}'.", colName));
                return;
            }

            Assert.True(false, "The LDA transform does not throw expected error on empty documents.");
        }
    }
}
