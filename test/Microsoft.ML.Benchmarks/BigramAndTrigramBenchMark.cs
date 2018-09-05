// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.IO;
using System.Text;

namespace Microsoft.ML.Benchmarks
{
    internal class EmptyWriter : TextWriter
    {
        private static EmptyWriter _instance = null;

        private EmptyWriter()
        {
        }

        public static EmptyWriter Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new EmptyWriter();
                }
                return _instance;
            }
        }
        public override Encoding Encoding => null;
    }

    public class BigramAndTrigramBenchmark
    {
        private static string s_dataPath_Wiki;
        private static string s_modelPath_Wiki;

        [GlobalSetup(Targets = new string[] { nameof(Preceptron_CV), nameof(LightGBM_CV), nameof(WordEmbedding_CV_AP), nameof(WordEmbedding_CV_SDCAMC) })]
        public void Setup_Preceptron_LightGBM()
        {
            s_dataPath_Wiki = TestDatasets.wikiDetox.trainFilename;

            if (!File.Exists(s_dataPath_Wiki))
            {
                throw new FileNotFoundException(s_dataPath_Wiki);
            }
        }

        [GlobalSetup(Target = nameof(wikiDetox))]
        public void Setup_wikiDetox()
        {
            Setup_Preceptron_LightGBM();
            s_modelPath_Wiki = Path.Combine(Directory.GetCurrentDirectory(), @"wikiModel.zip");
            string cmd = @"CV k=5 data=" + s_dataPath_Wiki + " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} xf=Convert{col=logged_in type=R4} xf=CategoricalTransform{col=ns} xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}} xf=Concat{col=Features:FeaturesText,logged_in,ns} tr=OVA{p=AveragedPerceptron{iter=10}} out={" + s_modelPath_Wiki + "}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void Preceptron_CV()
        {
            string cmd = @"CV k=5 data=" + s_dataPath_Wiki + " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} xf=Convert{col=logged_in type=R4} xf=CategoricalTransform{col=ns} xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}} xf=Concat{col=Features:FeaturesText,logged_in,ns} tr=OVA{p=AveragedPerceptron{iter=10}}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void LightGBM_CV()
        {
            string cmd = @"CV k=5 data=" + s_dataPath_Wiki + " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} xf=Convert{col=logged_in type=R4} xf=CategoricalTransform{col=ns} xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}} xf=Concat{col=Features:FeaturesText,logged_in,ns} tr=LightGBMMulticlass{}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void WordEmbedding_CV_AP()
        {
            string cmd = @"CV tr=OVA{p=AveragedPerceptron{iter=10}} k=5 loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} data=" + s_dataPath_Wiki + " xf=Convert{col=logged_in type=R4} xf=CategoricalTransform{col=ns} xf=TextTransform{col=FeaturesText:comment tokens=+ wordExtractor=NGramExtractorTransform{ngram=2}} xf=WordEmbeddingsTransform{col=FeaturesWordEmbedding:FeaturesText_TransformedText model=FastTextWikipedia300D} xf=Concat{col=Features:FeaturesText,FeaturesWordEmbedding,logged_in,ns}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void WordEmbedding_CV_SDCAMC()
        {
            string cmd = @"CV tr=SDCAMC k=5 loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} data=" + s_dataPath_Wiki + " xf=Convert{col=logged_in type=R4} xf=CategoricalTransform{col=ns} xf=TextTransform{col=FeaturesText:comment tokens=+ wordExtractor={} charExtractor={}} xf=WordEmbeddingsTransform{col=FeaturesWordEmbedding:FeaturesText_TransformedText model=FastTextWikipedia300D} xf=Concat{col=Features:FeaturesWordEmbedding,logged_in,ns}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void wikiDetox()
        {
            string modelpath = Path.Combine(Directory.GetCurrentDirectory(), @"wikiModel.fold000.zip");
            string cmd = @"Test data=" + s_dataPath_Wiki + " in=" + modelpath;
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }
    }
}
