// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.RunTests
{
    public class TestDataset
    {
        public string name;
        public string trainFilename;
        public string testFilename;
        public string validFilename;
        public string labelFilename;
        public char fileSeparator;
        public bool fileHasHeader;
        public bool allowQuoting;

        // REVIEW: Replace these with appropriate SubComponents!
        public string settings;
        public string testSettings;
        public string extraSettings;
        // REVIEW: Remove the three above setting strings once conversion work is complete.
        public string loaderSettings;
        public string[] mamlExtraSettings;
        public Func<TextLoader.Column[]> GetLoaderColumns;

        public TestDataset Clone()
        {
            var ret = new TestDataset
            {
                name = name,
                trainFilename = trainFilename,
                testFilename = testFilename,
                validFilename = validFilename,
                labelFilename = labelFilename,
                settings = settings,
                testSettings = testSettings,
                extraSettings = extraSettings,
                loaderSettings = loaderSettings,
                mamlExtraSettings = mamlExtraSettings
            };
            return ret;
        }
    }

    public static class TestDatasets
    {
        public static TestDataset breastCancer = new TestDataset
        {
            name = "breast-cancer",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            // REVIEW: For the purpose of the TL->MAML test translation work, this indicates
            // that the TestDataset instance was reviewed and no specific MAML settings are necessary, or
            // should be added by people doing other translation work as there are presumably tests already
            // depending on the empty settings as written.
            loaderSettings = ""
        };

        public static TestDataset breastCancerBoolLabel = new TestDataset
        {
            name = "breast-cancer",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            loaderSettings = "loader=Text{col=Label:BL:0 col=Features:~}"
        };

        public static TestDataset breastCancerGroupId = new TestDataset
        {
            name = "breast-cancer-group",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            loaderSettings = "loader=Text{col=Label:0 col=GroupId:U4[0-10]:1 col=Features:1-*}"
        };

        public static TestDataset breastCancerConst = new TestDataset
        {
            name = "breast-cancer",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            extraSettings = "cacheinst- inst=Text{label=0 attr=1-9}",
        };

        /// <summary>
        /// Should perform identically with breastCancer above.
        /// </summary>
        public static TestDataset breastCancerPipe = new TestDataset
        {
            name = "breast-cancer",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            // Using "col=Features:1-5,6,7-9" improves code coverage. Same with "col=Attr:TX:6".
            loaderSettings = "loader=Text{sparse- col=Attr:TX:6 col=Label:0 col=Features:1-5,6,7-9}",
            mamlExtraSettings = new[] { "cache-" },
            extraSettings = "/cacheinst- /inst Pipe{loader=Text{sparse- col=Attr:TX:6 col=Label:0 col=Features:1-5,6,7-9} lab=Label feat=Features}"
        };

        /// <summary>
        /// Fixes missing values.
        /// </summary>
        public static TestDataset breastCancerPipeMissing = new TestDataset
        {
            name = "breast-cancer-missing",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            // Note that More and More_Cleansed are not really needed (duplicate info), but improve code coverage.
            loaderSettings = "loader=Text{col=Label:0 col=Good:1-5,7-9 col=Mixed:6 col=More:4-6}",
            mamlExtraSettings = new[] { "cache-", "xf=NAHandle{col=Fixed:Mixed col=More}", "xf=Concat{col=Features:Good,Fixed,More}" },
        };

        /// <summary>
        /// Filters missing values.
        /// </summary>
        public static TestDataset breastCancerPipeMissingFilter = new TestDataset
        {
            name = "breast-cancer-missing-filter",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            // Note that More and More_Cleansed are not really needed (duplicate info), but improve code coverage.
            loaderSettings = "loader=Text{col=Label:0 col=Good:1-5,7-9 col=Mixed:6 col=More:4-6}",
            mamlExtraSettings = new[] { "cache-", "xf=MissingFilter{col=Mixed col=More}", "xf=Concat{col=Features:Good,Mixed,More}" },
        };

        public static TestDataset breastCancerOneClass = new TestDataset
        {
            name = "breast-cancer-one-class",
            trainFilename = "breast-cancer.oneclass.txt",
            testFilename = "breast-cancer.txt",
            loaderSettings = ""
        };

        public static TestDataset breastCancerSparseBinaryFeatures = new TestDataset
        {
            name = "breast-cancer-sparse",
            trainFilename = "breast-cancer.txt",
            testFilename = "breast-cancer.txt",
            loaderSettings = "xf=expr{col=Features expr=x:float(x>4?1:0)}"
        };

        // The data set contains images of hand-written digits.
        // The input is given in the form of matrix id 8x8 where
        // each element is an integer in the range 0..16
        public static TestDataset Digits = new TestDataset
        {
            name = "Digits",
            trainFilename = @"external/digits.csv",
        };

        public static TestDataset vw = new TestDataset
        {
            name = "vw",
            trainFilename = "vw.dat",
            testFilename = "vw.dat"
        };

        public static TestDataset housing = new TestDataset
        {
            name = "housing",
            trainFilename = "housing.txt",
            testFilename = "housing.txt",
            fileSeparator = '\t',
            fileHasHeader = true,
            loaderSettings = "loader=Text{col=Label:0 col=Features:~ header=+}",
            GetLoaderColumns = () =>
            {
                return new[] {
                    new TextLoader.Column("MedianHomeValue", DataKind.Single, 0),
                    new TextLoader.Column("CrimesPerCapita", DataKind.Single, 1),
                    new TextLoader.Column("PercentResidental", DataKind.Single, 2),
                    new TextLoader.Column("PercentNonRetail", DataKind.Single, 3),
                    new TextLoader.Column("CharlesRiver", DataKind.Single, 4),
                    new TextLoader.Column("NitricOxides", DataKind.Single, 5),
                    new TextLoader.Column("RoomsPerDwelling", DataKind.Single, 6),
                    new TextLoader.Column("PercentPre40s", DataKind.Single, 7),
                    new TextLoader.Column("EmploymentDistance", DataKind.Single, 8),
                    new TextLoader.Column("HighwayDistance", DataKind.Single, 9),
                    new TextLoader.Column("TaxRate", DataKind.Single, 10),
                    new TextLoader.Column("TeacherRatio", DataKind.Single, 11),
                };
            }
        };

        public static TestDataset generatedRegressionDatasetmacro = new TestDataset
        {
            name = "generatedRegressionDataset",
            trainFilename = "generated_regression_dataset.csv",
            testFilename = "generated_regression_dataset.csv",
            loaderSettings = "col=Label:R4:11 col=Features:R4:0-10 sep=; header+"
        };

        public static TestDataset WikiDetox = new TestDataset
        {
            name = "WikiDetox",
            trainFilename = "external/WikiDetoxAnnotated160kRows.tsv",
            testFilename = "external/WikiDetoxAnnotated160kRows.tsv"
        };

        public static TestDataset MSLRWeb = new TestDataset
        {
            name = "MSLRWeb",
            trainFilename = "external/MSLRWeb10KTrain720kRows.tsv",
            validFilename = "external/MSLRWeb10KValidate240kRows.tsv",
            testFilename = "external/MSLRWeb10KTest240kRows.tsv"
        };

        public static TestDataset Sentiment = new TestDataset
        {
            name = "sentiment",
            trainFilename = "wikipedia-detox-250-line-data.tsv",
            testFilename = "wikipedia-detox-250-line-test.tsv",
            fileHasHeader = true,
            fileSeparator = '\t',
            allowQuoting = true,
            GetLoaderColumns = () =>
             {
                 return new[]
                 {
                    new TextLoader.Column("Label", DataKind.Boolean, 0),
                    new TextLoader.Column("SentimentText", DataKind.String, 1)
                 };
             }
        };

        public static TestDataset generatedRegressionDataset = new TestDataset
        {
            name = "generatedRegressionDataset",
            trainFilename = "generated_regression_dataset.csv",
            testFilename = "generated_regression_dataset.csv",
            loaderSettings = "loader=Text{col=Label:R4:11 col=Features:R4:0-10 sep=; header+}"
        };

        public static TestDataset msm = new TestDataset
        {
            // REVIEW: Why is the MSM train set smaller than the test set? Reverse these!
            name = "MSM-sparse-sample",
            trainFilename = "MSM-sparse-sample-train.txt",
            testFilename = "MSM-sparse-sample-test.txt",
            loaderSettings = "loader=Text{col=Name:TX:0 col=Label:Num:1 col=Features:Num:~}",
            mamlExtraSettings = new[] { "xf=Expr{col=Name expr={x=>right(x, 1)}}" },
        };

        public static TestDataset msmNamesHeader = new TestDataset
        {
            name = "MSM-names",
            trainFilename = @"..\SmartMatch\Instances-Relevance.txt",
            testFilename = @"..\SmartMatch\Instances-Relevance.txt",
            settings = "header+;name:0,1"
        };

        public static TestDataset msmNamesHeaderIps = new TestDataset
        {
            name = "MSM-names",
            trainFilename = @"..\SmartMatch\Instances-Relevance.txt",
            testFilename = @"..\SmartMatch\Instances-Relevance.txt",
            settings = "header+;name:0,1"
        };

        public static TestDataset extract1 = new TestDataset
        {
            name = "Extract1",
            trainFilename = "Extract1.txt",
            testFilename = "Extract1.txt"
        };

        public static TestDataset breastCancerBing = new TestDataset
        {
            name = "breast-cancer-bing",
            trainFilename = "breast-cancer-bing.txt",
            testFilename = "breast-cancer-bing.txt",
            extraSettings = "/inst ExtractInstances{}"
        };

        public static TestDataset adult = new TestDataset
        {
            name = "Census",
            trainFilename = "adult.tiny.with-schema.txt",
            testFilename = "adult.tiny.with-schema.txt",
            fileHasHeader = true,
            fileSeparator = '\t',
            loaderSettings = "loader=Text{header+ col=Label:0 col=Num:9-14 col=Cat:TX:1-8}",
            mamlExtraSettings = new[] { "xf=Cat{col=Cat}", "xf=Concat{col=Features:Num,Cat}" },
            extraSettings = @"/inst Text{header+ sep=, label=14 handler=Categorical{cols=5-9,1,13,3}}",
        };

        public static TestDataset adultOnlyCat = new TestDataset
        {
            name = "Census-Cat-Only",
            trainFilename = "adult.tiny.with-schema.txt",
            testFilename = "adult.tiny.with-schema.txt",
            loaderSettings = "loader=Text{header+ col=Label:0 col=Cat:TX:1-8}",
            mamlExtraSettings = new[] { "xf=Cat{col=Cat}", "xf=Concat{col=Features:Cat}" },
            extraSettings = @"/inst Text{header+ sep=, label=14 handler=Categorical{cols=5-9,1,13,3}}",
        };

        public static TestDataset adultHash = new TestDataset
        {
            name = "CensusHash",
            trainFilename = "adult.tiny.with-schema.txt",
            testFilename = "adult.tiny.with-schema.txt",
            loaderSettings = "loader=Text{header+ col=Label:0 col=Num:9-14 col=Cat:TX:1-8}",
            mamlExtraSettings = new[] { "xf=CatHash{col=Cat bits=5}", "xf=Concat{col=Features:Num,Cat}" },
            extraSettings = @"/inst Text{header+ sep=, label=14 handler=CatHash{cols=1,3,5-9,13 bits=5}}"
        };

        public static TestDataset adultHashWithDataPipe = new TestDataset
        {
            name = "CensusHashWithPipe",
            trainFilename = "adult.tiny.with-schema.txt",
            testFilename = "adult.tiny.with-schema.txt",
            loaderSettings = "loader=Text{header+ col=Cat:TX:1-8 col=Label:0 col=Num:~}",
            mamlExtraSettings = new[] { "xf=CatHash{col=Hash:5:Cat}", "xf=Concat{col=Features:Num,Hash}" }
        };

        public static TestDataset adultText = new TestDataset
        {
            name = "CensusText",
            trainFilename = "adult.tiny.with-schema.txt",
            testFilename = "adult.tiny.with-schema.txt",
            loaderSettings = "loader=Text{header+ col=Label:0 col=Word:TX:1-8 col=Num:~}",
            mamlExtraSettings = new[] { "xf=WordBag{col=Word}", "xf=Concat{col=Features:Num,Word}" },
            extraSettings = @"/inst Text{header+ sep=, label=14 handler=WordBag{cols=1,3,5-9,13}}"
        };

        public static TestDataset adultTextHash = new TestDataset
        {
            name = "CensusTextHash",
            trainFilename = "adult.tiny.with-schema.txt",
            testFilename = "adult.tiny.with-schema.txt",
            loaderSettings = "loader=Text{header+ col=Label:0 col=Word:TX:1-8 col=Num:~}",
            mamlExtraSettings = new[] { "xf=WordHashBag{col=Word bits=8}", "xf=Concat{col=Features:Num,Word}" },
            extraSettings = @"/inst Text{header+ sep=, label=14 handler=WordHashBag{cols=1,3,5-9,13 sep=, bits=8}}"
        };

        public static TestDataset adultRanking = new TestDataset
        {
            name = "adultRanking",
            trainFilename = "adult.tiny.with-schema.txt",
            loaderSettings = "loader=Text{header+ sep=tab, col=Label:R4:0 col=Workclass:TX:1 col=Categories:TX:2-8 col=NumericFeatures:R4:9-14}",
        };

        public static TestDataset displayPoisson = new TestDataset
        {
            name = "DisplayPoisson",
            trainFilename = @"..\synthetic\Poisson-display-train.txt",
            testFilename = @"..\synthetic\Poisson-display-test.txt",
            settings = "header+;cat:1,2,3;label:5;max:2000;attr:6,7",
            testSettings = "header+;cat:1,2,3;label:6;attr:4,7,8"
        };

        public static TestDataset displayPoissonWithInstanceParser = new TestDataset
        {
            name = "DisplayPoisson",
            trainFilename = @"..\synthetic\Poisson-display-train.txt",
            testFilename = @"..\synthetic\Poisson-display-test.txt",
            settings = "header+;cat:1,2,3;label:5;max:2000;attr:6,7",
            testSettings = "header+;cat:1,2,3;label:6;attr:4,7,8"
        };

        public static TestDataset childrenPoisson = new TestDataset
        {
            name = "ChildrenPoisson",
            trainFilename = @"..\children\children.txt",
            testFilename = @"..\children\children.txt",
            loaderSettings = "loader=Text{header+ sep=space col=Cat1:TX:1 col=Cat2:TX:2 col=Cat3:TX:3 col=Label:4 col=Ignore:TX:0,5-7 col=Features:8-*}",
            mamlExtraSettings = new[] { "xf=Cat{col=Cat1 col=Cat2 col=Cat3}", "xf=Concat{col=Features:Features,Cat1,Cat2,Cat3}" },
            // settings = "header+;sep:space;cat:1,2,3;label:4;attr:0,5,6,7",
        };

        public static TestDataset autosSample = new TestDataset
        {
            name = "AutosSample",
            trainFilename = @"auto-sample.txt",
            testFilename = @"auto-sample.txt",
            loaderSettings = "loader=Text{col=Label:0 col=Cat3:TX:3 col=Cat4:TX:4 col=Cat5:TX:5 col=Cat6:TX:6 col=Cat7:TX:7 col=Cat8:TX:8 col=Cat9:TX:9 col=Cat15:TX:15 col=Cat16:TX:16 col=Cat18:TX:18 col=Features:~}",
            mamlExtraSettings = new[] {
                "xf=Cat{col=Cat3 col=Cat4 col=Cat5 col=Cat6 col=Cat7 col=Cat8 col=Cat9 col=Cat15 col=Cat16 col=Cat18}",
                "xf=Concat{col=Features:Features,Cat3,Cat4,Cat5,Cat6,Cat7,Cat8,Cat9,Cat15,Cat16,Cat18}" },
            // extraSettings = "inst=Text{cat=3,4,5,6,7,8,9,15,16,18 label=0 maxBad=100}"
        };

        public static TestDataset reutersMaxDim = new TestDataset()
        {
            name = "reuters",
            trainFilename = @"RCV1\reuters-toy-test.txt",
            testFilename = @"RCV1\reuters-toy-test.txt",
            loaderSettings = "loader=Text{size=10000 col=Label:0 col=Features:1-*}",
            settings = "maxdim:10000"
        };

        public static TestDataset irisLoader = new TestDataset()
        {
            name = "iris",
            trainFilename = @"iris.txt",
            testFilename = @"iris.txt",
            loaderSettings = "loader=Text{col=Label:TX:0 col=Features:1-*}",
            mamlExtraSettings = new[] { "xf=Term{col=Label}" },
        };

        public static TestDataset irisData = new TestDataset()
        {
            name = "iris",
            trainFilename = @"iris.data",
            loaderSettings = "loader=Text{col=Label:TX:4 col=Features:0-3}",
            GetLoaderColumns = () =>
            {
                return new[]
                {
                    new TextLoader.Column("SepalLength", DataKind.Single, 0),
                    new TextLoader.Column("SepalWidth", DataKind.Single, 1),
                    new TextLoader.Column("PetalLength", DataKind.Single, 2),
                    new TextLoader.Column("PetalWidth",DataKind.Single, 3),
                    new TextLoader.Column("Label", DataKind.String, 4)
                };
            }
        };

        public static TestDataset irisLabelName = new TestDataset()
        {
            name = "iris-label-name",
            trainFilename = @"iris-label-name.txt",
            testFilename = @"iris-label-name.txt",
            loaderSettings = "loader=Text{header+ col=Label:TX:0 col=Features:1-*}",
            mamlExtraSettings = new[] { "xf=Term{col=Label}" },
        };

        public static TestDataset irisTreeFeaturized = new TestDataset()
        {
            name = "iris-tree-featurized",
            trainFilename = @"iris.txt",
            testFilename = @"iris.txt",
            loaderSettings = "loader=Text{col=Label:U4[0-2]:0 col=Features:1-*}",
            mamlExtraSettings = new[] { "xf=TreeFeat{lps=0 trainer=ftr{iter=3}} xf=copy{col=Features:Leaves}" },
        };

        public static TestDataset irisTreeFeaturizedPermuted = new TestDataset()
        {
            name = "iris-tree-featurized-permuted",
            trainFilename = @"iris.txt",
            testFilename = @"iris.txt",
            loaderSettings = "loader=Text{col=Label:U4[0-2]:0 col=Features:1-*}",
            mamlExtraSettings = new[] { "xf=TreeFeat{lps=2 trainer=ftr{iter=3}} xf=copy{col=Features:Leaves}" },
        };

        public static TestDataset irisLoaderU404 = new TestDataset()
        {
            name = "iris",
            trainFilename = @"iris.txt",
            testFilename = @"iris.txt",
            loaderSettings = "loader=Text{col=Label:U4[0-2]:0 col=Features:1-4}",
        };

        public static TestDataset iris = new TestDataset()
        {
            name = "iris",
            trainFilename = @"iris.txt",
            testFilename = @"iris.txt",
            fileHasHeader = true,
            fileSeparator = '\t',
            mamlExtraSettings = new[] { "xf=Term{col=Label}" }
        };

        public static TestDataset irisMissing = new TestDataset()
        {
            name = "irisMissing",
            trainFilename = @"iris.txt",
            testFilename = @"iris.txt",
            // Create missing labels in iris by generating a number then replacing 5% with an NA label.
            mamlExtraSettings = new[] { "xf=generateNumber{col=A} xf=expr{col=Label:Label,A expr={(lab,a):a<0.05?na(lab):lab}}" }
        };

        public static TestDataset LM = new TestDataset()
        {
            name = "LM",
            trainFilename = @"..\LM\Local.source_features.de-de.txt",
            testFilename = @"..\LM\Local.validate_features.de-de.txt",
            labelFilename = @"..\LM\Mapping.de-de.txt",
            settings = @"name:1,2;text:3,4;header+"
        };

        public static TestDataset LMWordHashBag = new TestDataset()
        {
            name = "LM",
            trainFilename = @"..\LM\Local.source_features.de-de.txt",
            testFilename = @"..\LM\Local.validate_features.de-de.txt",
            labelFilename = @"..\LM\Mapping.de-de.txt",
            extraSettings = @"/inst Text{header+ attr=1,2 handler=WordHashBag{cols=3,4}}"
        };

        public static TestDataset LMCharGrams = new TestDataset()
        {
            name = "LMCharGrams",
            trainFilename = @"..\LM\Local.source_features.de-de.txt",
            testFilename = @"..\LM\Local.validate_features.de-de.txt",
            labelFilename = @"..\LM\Mapping.de-de.txt",
            extraSettings = @"/inst Text{header+ attr=1,2 handler=CharGram{cols=3,4 len=3}}"
        };

        public static TestDataset LMBigrams = new TestDataset()
        {
            name = "LMBigrams",
            trainFilename = @"..\LM\Local.source_features.de-de.txt",
            testFilename = @"..\LM\Local.validate_features.de-de.txt",
            labelFilename = @"..\LM\Mapping.de-de.txt",
            extraSettings = @"/inst Text{header+ attr=1,2 handler=WordBag{cols=3,4 ngram=2}}"
        };

        public static TestDataset LMNgrams = new TestDataset()
        {
            name = "LMNgrams",
            trainFilename = @"..\LM\Local.source_features.de-de.txt",
            testFilename = @"..\LM\Local.validate_features.de-de.txt",
            labelFilename = @"..\LM\Mapping.de-de.txt",
            extraSettings = @"/inst Text{header+ attr=1,2 handler=WordBag{cols=3,4 ngram=5 max=200000}}"
        };

        public static TestDataset LMSkipNgrams = new TestDataset()
        {
            name = "LMSkipNgrams",
            trainFilename = @"..\LM\Local.source_features.de-de.txt",
            testFilename = @"..\LM\Local.validate_features.de-de.txt",
            labelFilename = @"..\LM\Mapping.de-de.txt",
            extraSettings = @"/inst Text{header+ attr=1,2 handler=WordBag{cols=3,4 ngram=4 skips=1 max=200000}}"
        };

        public static TestDataset LMNgramsHashing = new TestDataset()
        {
            name = "LMNgramsHashing",
            trainFilename = @"..\LM\Local.source_features.de-de.txt",
            testFilename = @"..\LM\Local.validate_features.de-de.txt",
            labelFilename = @"..\LM\Mapping.de-de.txt",
            extraSettings = @"/inst Text{header+ attr=1,2 handler=WordHashBag{cols=3,4 ngram=10}}"
        };

        public static TestDataset rankingText = new TestDataset()
        {
            name = "ranking",
            trainFilename = @"ranking-sample.txt",
            testFilename = @"ranking-sample.txt",
            labelFilename = @"ranking-sample.txt.labels",
            loaderSettings = "loader=Text{header+ col=Label:TX:0 col=GroupId:U4[0-*]:1 col=Name:TX:1-2 col=Features:3-*}",
            extraSettings = @"/inst Text{header+ name=1-2 groupKey=n0}"
        };

        public static TestDataset rankingExtract = new TestDataset()
        {
            name = "ranking",
            trainFilename = @"ranking-sample.txt",
            testFilename = @"ranking-sample.txt",
            labelFilename = @"ranking-sample.txt.labels",
            extraSettings = @"/inst ExtractInstances{header+ name=1-2 groupKey=n0}"
        };

        public static TestDataset breastCancerWeighted = new TestDataset
        {
            name = "breast-cancer-weighted",
            trainFilename = @"ArtificiallyWeighted\breast-cancer-weights-quarter.txt",
            testFilename = @"ArtificiallyWeighted\breast-cancer-weights-quarter.txt",
            loaderSettings = "loader=Text{col=Weight:0 col=Label:1 col=Features:~}",
            settings = "weight:0;label:1"
        };

        public static TestDataset breastCancerDifferentlyWeighted = new TestDataset
        {
            name = "breast-cancer-weighted",
            trainFilename = @"ArtificiallyWeighted\breast-cancer-weights-quarter.txt",
            testFilename = @"ArtificiallyWeighted\breast-cancer-weights-quarter.txt",
            loaderSettings = "loader=Text{col=Label:Num:1 col=Weight:Num:4 col=Features:Num:~}",
        };

        public static TestDataset housingWeightedRep = new TestDataset
        {
            name = "housing-weighted",
            trainFilename = @"ArtificiallyWeighted\housing-weights-quarter-rep.txt",
            testFilename = @"ArtificiallyWeighted\housing-weights-quarter-rep.txt",
            loaderSettings = "loader=Text{col=Weight:0 col=Label:1 col=Features:~}",
        };

        public static TestDataset housingDifferentlyWeightedRep = new TestDataset
        {
            name = "housing-weighted",
            trainFilename = @"ArtificiallyWeighted\housing-weights-quarter-rep.txt",
            testFilename = @"ArtificiallyWeighted\housing-weights-quarter-rep.txt",
            loaderSettings = "loader=Text{col=Label:1 col=Weight:10 col=Features:~}",
            settings = "weight:10;label:1"
        };

        public static TestDataset rankingWeighted = new TestDataset()
        {
            name = "ranking-weighted",
            trainFilename = @"ArtificiallyWeighted\ranking-sample-weights-one.txt",
            testFilename = @"ArtificiallyWeighted\ranking-sample-weights-one.txt",
            labelFilename = @"ranking-sample.txt.labels",
            loaderSettings = "loader=Text{header+ col=Weight:0 col=Label:TX:1 col=Name:TX:2-3 col=GroupId:U4[0-*]:2 col=Features:~}",
        };

        public static TestDataset adultSparseWithCategory = new TestDataset()
        {
            name = "adult-sparseWithCat",
            trainFilename = @"adult.SparseWithCat.txt",
            testFilename = @"adult.SparseWithCat.txt",
            settings = @"cat:0"
        };

        public static TestDataset adultSparseWithCategoryHash = new TestDataset()
        {
            name = "adult-sparseWithCat",
            trainFilename = @"adult.SparseWithCat.txt",
            testFilename = @"adult.SparseWithCat.txt",
            settings = "",
            extraSettings = @"/inst TextInstances { handler=CatHashHandler{cols=0 bits=6} }"
        };

        public static TestDataset adultSparseWithCatAsAtt = new TestDataset()
        {
            name = "adult-sparseWithCat",
            trainFilename = @"adult.SparseWithCat.txt",
            testFilename = @"adult.SparseWithCat.txt",
            settings = "",
            extraSettings = @"/inst TextInstances { attr=0 threads-}"
        };

        public static TestDataset pClick = new TestDataset()
        {
            name = "pClick",
            trainFilename = @"..\pclick\pclick-train.txt",
            testFilename = @"..\pclick\pclick-test.txt",
            loaderSettings = "loader=Text{header+ col=Label:0 col=Features:1-84 rows=3000}",
            settings = @"header+;attr:85,86;max:3000"
        };

        public static TestDataset mnist28 = new TestDataset()
        {
            name = "mnist28",
            trainFilename = @"Train-28x28.txt",
            testFilename = @"Test-28x28.txt"
        };

        public static TestDataset mnistTiny28 = new TestDataset()
        {
            name = "mnistTiny28",
            trainFilename = @"Train-Tiny-28x28.txt",
            testFilename = @"Test-Tiny-28x28.txt"
        };

        public static TestDataset sampleBingRegression = new TestDataset()
        {
            name = "sampleBingRegression",
            trainFilename = @"..\Bing\SampleInputExtraction.txt",
            testFilename = @"..\Bing\SampleInputExtraction.txt",
            labelFilename = @"..\Bing\labelmap.txt",
            settings = @"header:+;attr:1-4;groupkey:a0"
        };

        public static TestDataset sampleBingBin = new TestDataset()
        {
            name = "sampleBingBin",
            trainFilename = @"..\Bing\SampleInputExtraction.bin",
            testFilename = @"..\Bing\SampleInputExtraction.bin",
            labelFilename = @"..\Bing\labelmap.txt",
            loaderSettings = "loader=Text{header+ col=Label:0 col=Features:1-84 rows=3000}",
        };

        public static TestDataset mnistOneClass = new TestDataset()
        {
            name = "mnistOneClass",
            trainFilename = @"MNIST.Train.0-class.tiny.txt",
            testFilename = @"MNIST.Test.tiny.txt",
            fileHasHeader = false,
            fileSeparator = '\t',
            settings = ""
        };

        public static TestDataset WebClicksSample = new TestDataset()
        {
            name = "webClicksSample",
            trainFilename = @"..\AdSelection\webClicksSample.small.txt",
            testFilename = @"..\AdSelection\webClicksSample.small.txt",
            settings = ""
        };

        public static TestDataset AppFailure = new TestDataset()
        {
            name = "AppFailure",
            trainFilename = @"AppFailure.csv",
            settings = "/inst Text{sep=, name=0 label=2}"
        };

        public static TestDataset azureCounterUnlabeled = new TestDataset()
        {
            name = "azureCounterUnlabeled",
            trainFilename = @"azure-train-unlabeled.txt",
            testFilename = @"azure-test-unlabeled.txt",
            loaderSettings = "loader=Text{sep=space col=Name:TX:0 col=Features:~}",
            mamlExtraSettings = new[] { "xf=Expr{col=Label:Name expr={x : na(float(0))}}" },
            settings = "/inst Text{name=0 sep=space nolabel=+}"
        };

        public static TestDataset MQ2008 = new TestDataset
        {
            name = "MQ2008",
            trainFilename = @"MQ2008\Train.idv.small.txt",
            testFilename = @"MQ2008\Test.idv.small.txt",
            loaderSettings = "loader=Text"
        };

        public static TestDataset SequenceDataset = new TestDataset()
        {
            name = "sequenceDataset",
            trainFilename = @"..\V3\Data\OCR\train.tsv",
            testFilename = @"..\V3\Data\OCR\train.tsv",
            loaderSettings = "loader=Text{col=Label:U1[0-25]:1 col=GroupId:U4[1-*]:3 col=Features:Num:4-*}"
        };

        public static TestDataset trivialMatrixFactorization = new TestDataset()
        {
            name = "trivialMatrixFactorization",
            trainFilename = @"trivial-train.tsv",
            testFilename = @"trivial-test.tsv",
            fileHasHeader = true,
            fileSeparator = '\t',
            loaderSettings = "loader=Text{col=Label:R4:0 col=User:U4[0-19]:1 col=Item:U4[0-39]:2 header+}"
        };
    }
}
