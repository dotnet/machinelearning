using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Microsoft.ML.Scenarios
{

    public partial class ScenariosTests
    {
        [Fact]
        public void PredictNewsCluster()
        {
            string modelFilePath = GetOutputPath("PredictNewsCluster.zip");
            string dataPath = GetDataPath("20newsgroups.txt");

            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<NewsData>(useHeader: false));
            pipeline.Add(new CategoricalOneHotVectorizer("Label"));
            pipeline.Add(new ColumnConcatenator("AllText", "Subject", "Content"));
            
            pipeline.Add(new TextFeaturizer("Features", "AllText")
            {
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
                OutputTokens = true,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextTransformTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 1, AllLengths = true }
            });
            pipeline.Add(new KMeansPlusPlusClusterer() { K = 20, MaxIterations=10 });

            var  model = pipeline.Train<NewsData, NewsPrediction>();

            var result = model.Predict(new NewsData() { Subject = "Re: Return of the Know Nothing Party", Content = @"In article <C5JLq3.2BL@wetware.com>, drieux@wetware.com                               writes... >In article 23791@organpipe.uug.arizona.edu, ece_0028@bigdog.engr.arizona.edu (David Anderson) writes: >>In article <C56HDM.945@wetware.com> drieux@wetware.com (drieux, just drieux) writes: >>>But I guess one needs to know a little about the bible, >>>christianity and american history..... >> >>Mt. St. Helens didn't spew such crap.  How do you manage, >>drieux, day in & day out, to keep it up?? >  >So which are you advocating? >That You know Nothing About American History,  >Or that You Know Nothing About the Bible? >  >Is this a Restoration of the ""Know Nothing"" Party? >  Go easy on him drieux. It is the right of every American to know nothing about anything.   >ciao >drieux  >""All Hands to the Big Sea of COMedy! > All Hands to the Big Sea of COMedy!"" >  -Last Call of the Wild of the Humour Lemmings >   ------------------------------------------------------------------------------ ""Who said anything about panicking ? "" snapped Authur.           Garrett Johnson ""This is still just culture shock.You wait till I've       Garrett@Ingres.com settled into the situation and found my bearings. THEN I'll start panicking!"" - Douglas Adams   ------------------------------------------------------------------------------" });
        }

        public class NewsData
        {
            [Column(ordinal: "0")]
            public string Id;

            [Column(ordinal: "1", name:"Label")]
            public string Topic;

            [Column(ordinal: "2")]
            public string Subject;

            [Column(ordinal: "3")]
            public string Content;
        }

        public class NewsPrediction
        {
            [ColumnName("Score")]
            public float[] Topic;
        }

        
    }
}
