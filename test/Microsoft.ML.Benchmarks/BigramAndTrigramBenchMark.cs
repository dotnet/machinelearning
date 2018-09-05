// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.TestFramework;
using System.IO;

namespace Microsoft.ML.Benchmarks
{
    public class BigramAndTrigramBenchmark
    {        
        private static string s_dataPath_Wiki;
        private static string s_output_Wiki;

        [Benchmark]
        public void Preceptron_CV()
        {
            string outPath = Path.Combine(s_output_Wiki, @"0.model.zip");
            string cmd = @"CV k=5 data=" + s_dataPath_Wiki + " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} xf=Convert{col=logged_in type=R4} xf=CategoricalTransform{col=ns} xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}} xf=Concat{col=Features:FeaturesText,logged_in,ns} tr=OVA{p=AveragedPerceptron{iter=10}} out={" + outPath + "}";
            Maml.MainAll(cmd);
        }

        [Benchmark]
        public void LightGBM_CV()
        {
            string outPath = Path.Combine(s_output_Wiki, @"1.model.zip");
            string cmd = @"CV k=5 data=" + s_dataPath_Wiki + " loader=TextLoader{quote=- sparse=- col=Label:R4:0 col=rev_id:TX:1 col=comment:TX:2 col=logged_in:BL:4 col=ns:TX:5 col=sample:TX:6 col=split:TX:7 col=year:R4:3 header=+} xf=Convert{col=logged_in type=R4} xf=CategoricalTransform{col=ns} xf=TextTransform{col=FeaturesText:comment wordExtractor=NGramExtractorTransform{ngram=2}} xf=Concat{col=Features:FeaturesText,logged_in,ns} tr=LightGBMMulticlass{} out={" + outPath + "}";
            Maml.MainAll(cmd);
        }

        [Benchmark]
        public void wikiDetox_Test()
        {
            string modelPath = Path.Combine(s_output_Wiki, @"0.model.fold000.zip");
            string cmd = @"Test data=" + s_dataPath_Wiki + " in=" + modelPath;
            Maml.MainAll(cmd);
        }

        [GlobalSetup]
        public void Setup()
        {
            s_dataPath_Wiki = Program.GetInvariantCultureDataPath("wikiDetoxAnnotated160kRows.tsv");

            if (!File.Exists(s_dataPath_Wiki))
            {
                throw new FileNotFoundException(s_dataPath_Wiki);
            }

            var currentAssemblyLocation = new FileInfo(typeof(BaseTestClass).Assembly.Location);
            string outDir = Path.Combine(currentAssemblyLocation.Directory.FullName, "TestOutput");

            s_output_Wiki = Path.Combine(outDir, @"BenchmarkDefForMLNET\WikiDetox\00-baseline,_Bigram+Trichar\");
            Directory.CreateDirectory(s_output_Wiki);
        }
    }
}
