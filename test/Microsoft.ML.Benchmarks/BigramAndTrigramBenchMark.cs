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
    // Adding this class to not print anything to the console.
    // This is required for the current version of BenchmarkDotNet
    internal class EmptyWriter : TextWriter
    {
        internal static readonly EmptyWriter Instance = new EmptyWriter();
        public override Encoding Encoding => null;
    }

    public class BigramAndTrigramBenchmark
    {
        private static string s_dataPath_Wiki;
        private static string s_modelPath_Wiki;

        [GlobalSetup(Targets = new string[] { nameof(Preceptron_CV), nameof(LightGBM_CV) })]
        public void Setup_Preceptron_LightGBM()
        {
            s_dataPath_Wiki = Path.GetFullPath(TestDatasets.WikiDetox.trainFilename);

            if (!File.Exists(s_dataPath_Wiki))
            {
                throw new FileNotFoundException(s_dataPath_Wiki);
            }
        }

        [GlobalSetup(Target = nameof(WikiDetox))]
        public void Setup_WikiDetox()
        {
            Setup_Preceptron_LightGBM();
            s_modelPath_Wiki = Path.Combine(Directory.GetCurrentDirectory(), @"WikiModel.zip");
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
        public void WikiDetox()
        {
            string modelpath = Path.Combine(Directory.GetCurrentDirectory(), @"WikiModel.fold000.zip");
            string cmd = @"Test data=" + s_dataPath_Wiki + " in=" + modelpath;
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }
    }
}
