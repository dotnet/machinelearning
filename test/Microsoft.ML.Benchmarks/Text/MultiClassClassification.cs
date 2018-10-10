// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.IO;

namespace Microsoft.ML.Benchmarks
{
    [Config(typeof(TrainConfig))]
    public class MultiClassClassificationTrain
    {
        private string _dataPath_Wiki;

        [GlobalSetup]
        public void SetupTrainingSpeedTests()
        {
            _dataPath_Wiki = Path.GetFullPath(TestDatasets.WikiDetox.trainFilename);

            if (!File.Exists(_dataPath_Wiki))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _dataPath_Wiki));           
        }

        [Benchmark]
        public void CV_Multiclass_WikiDetox_BigramsAndTrichar_OVAAveragedPerceptron()
        {
            string cmd = @"CV k=5 data=" + _dataPath_Wiki + 
                        " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+}" +
                        " xf=Convert{col=logged_in type=R4}" +
                        " xf=CategoricalTransform{col=ns}" +
                        " xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}}" +
                        " xf=Concat{col=Features:FeaturesText,logged_in,ns}" +
                        " tr=OVA{p=AveragedPerceptron{iter=10}}";

            using (var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, CategoricalTransform, AveragedPerceptronTrainer>())
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void CV_Multiclass_WikiDetox_BigramsAndTrichar_LightGBMMulticlass()
        {
            string cmd = @"CV k=5 data=" + _dataPath_Wiki +
                    " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+}" +
                    " xf=Convert{col=logged_in type=R4}" +
                    " xf=CategoricalTransform{col=ns}" +
                    " xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}}" +
                    " xf=Concat{col=Features:FeaturesText,logged_in,ns}" +
                    " tr=LightGBMMulticlass{iter=10}";

            using (var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, CategoricalTransform, LightGbmMulticlassTrainer>())
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void CV_Multiclass_WikiDetox_WordEmbeddings_OVAAveragedPerceptron()
        {
            string cmd = @"CV k=5  data=" + _dataPath_Wiki +
                " tr=OVA{p=AveragedPerceptron{iter=10}}" +
                " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+}" +
                " xf=Convert{col=logged_in type=R4}" +
                " xf=CategoricalTransform{col=ns}" +
                " xf=TextTransform{col=FeaturesText:comment tokens=+ wordExtractor=NGramExtractorTransform{ngram=2}}" +
                " xf=WordEmbeddingsTransform{col=FeaturesWordEmbedding:FeaturesText_TransformedText model=FastTextWikipedia300D}" +
                " xf=Concat{col=Features:FeaturesText,FeaturesWordEmbedding,logged_in,ns}";

            using (var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, CategoricalTransform, AveragedPerceptronTrainer>())
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void CV_Multiclass_WikiDetox_WordEmbeddings_SDCAMC()
        {
            string cmd = @"CV k=5 data=" + _dataPath_Wiki +
                " tr=SDCAMC" +
                " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+}" +
                " xf=Convert{col=logged_in type=R4}" +
                " xf=CategoricalTransform{col=ns}" +
                " xf=TextTransform{col=FeaturesText:comment tokens=+ wordExtractor={} charExtractor={}}" +
                " xf=WordEmbeddingsTransform{col=FeaturesWordEmbedding:FeaturesText_TransformedText model=FastTextWikipedia300D}" +
                " xf=Concat{col=Features:FeaturesWordEmbedding,logged_in,ns}";

            using (var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, CategoricalTransform, SdcaMultiClassTrainer>())
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }
    }

    public class MultiClassClassificationTest
    {
        private string _dataPath_Wiki;
        private string _modelPath_Wiki;

        [GlobalSetup]
        public void SetupScoringSpeedTests()
        {
            _dataPath_Wiki = Path.GetFullPath(TestDatasets.WikiDetox.trainFilename);

            if (!File.Exists(_dataPath_Wiki))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _dataPath_Wiki));

            _modelPath_Wiki = Path.Combine(Directory.GetCurrentDirectory(), @"WikiModel.zip");

            string cmd = @"CV k=5 data=" + _dataPath_Wiki +
                " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} xf=Convert{col=logged_in type=R4}" +
                " xf=CategoricalTransform{col=ns}" +
                " xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}}" +
                " xf=Concat{col=Features:FeaturesText,logged_in,ns}" +
                " tr=OVA{p=AveragedPerceptron{iter=10}}" +
                " out={" + _modelPath_Wiki + "}";

            using (var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, CategoricalTransform, AveragedPerceptronTrainer>())
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void Test_Multiclass_WikiDetox_BigramsAndTrichar_OVAAveragedPerceptron()
        {
            // This benchmark is profiling bulk scoring speed and not training speed. 
            string modelpath = Path.Combine(Directory.GetCurrentDirectory(), @"WikiModel.fold000.zip");
            string cmd = @"Test data=" + _dataPath_Wiki + " in=" + modelpath;

            using (var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, CategoricalTransform, AveragedPerceptronTrainer>())
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }
    }
}
