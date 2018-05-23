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
        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/203")]
        public void PredictNewsCluster()
        {
            string dataPath = GetDataPath(@"external/20newsgroups.txt");

            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(dataPath).CreateFrom<NewsData>(useHeader: false));
            pipeline.Add(new CategoricalOneHotVectorizer("Label"));
            pipeline.Add(new ColumnConcatenator("AllText", "Subject", "Content"));
            pipeline.Add(new TextFeaturizer("Features", "AllText")
            {
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextTransformTextNormKind.None,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 1, AllLengths = true }
            });

            pipeline.Add(new KMeansPlusPlusClusterer() { K = 20 });
            var model = pipeline.Train<NewsData, ClusteringPrediction>();
            var gunResult = model.Predict(new NewsData() { Subject = "Let's disscuss gun control", Content = @"The United States has 88.8 guns per 100 people, or about 270,000,000 guns, which is the highest total and per capita number in the world. 22% of Americans own one or more guns (35% of men and 12% of women). America's pervasive gun culture stems in part from its colonial history, revolutionary roots, frontier expansion, and the Second Amendment, which states: ""A well regulated militia,
                being necessary to the security of a free State,
                the right of the people to keep and bear Arms,
                shall not be infringed.""

Proponents of more gun control laws state that the Second Amendment was intended for militias; that gun violence would be reduced; that gun restrictions have always existed; and that a majority of Americans, including gun owners, support new gun restrictions. " });
            var puppiesResult = model.Predict(new NewsData()
            {
                Subject = "Studies Reveal Five Ways Dogs Show Us Their Love",
                Content = @"Let's face it: We all adore our dogs as if they were family and we tend to shower our dogs with affection in numerous ways. Perhaps you may buy your dog a favorite toy or stop by the dog bakery to order some great tasting doggy cookies, or perhaps you just love patting your dog in the evening in the way he most loves. But how do our dogs tell us they love us too?

Until the day your dog can talk, you'll never likely hear him pronounce ""I love you,"" and in the meantime, don't expect him to purchase you a Hallmark card or some balloons with those renowned romantic words printed on top. Also, don’t expect a box of chocolates or a bouquet of flowers from your dog when Valentine's day is around the corner. Sometimes it might feel like we're living an uneven relationship, but just because dogs don't communicate their love the way we do, doesn't mean they don't love us!"
            });
        }

        public class NewsData
        {
            [Column(ordinal: "0")]
            public string Id;

            [Column(ordinal: "1", name: "Label")]
            public string Topic;

            [Column(ordinal: "2")]
            public string Subject;

            [Column(ordinal: "3")]
            public string Content;
        }

        public class ClusteringPrediction
        {
            [ColumnName("PredictedLabel")]
            public uint SelectedClusterId;
            [ColumnName("Score")]
            public float[] Distance;
        }

        public class ClusterData
        {
            [ColumnName("Features")]
            [VectorType(2)]
            public float[] Points;
        }

        [Fact]
        public void PredictClusters()
        {
            int n = 1000;
            int k = 5;
            var rand = new Random();
            var clusters = new ClusterData[k];
            var data = new ClusterData[n];
            for (int i = 0; i < k; i++)
            {
                //pick clusters as points on circle with angle to axis X equal to 360*i/k
                clusters[i] = new ClusterData { Points = new float[2] { (float)Math.Cos(Math.PI * i *2 /  k), (float)Math.Sin(Math.PI * i *2 / k) } };
            }
            // create data points by randomly picking cluster and shifting point slightly away from it.
            for (int i = 0; i < n; i++)
            {
                var index = rand.Next(0, k);
                var shift = (rand.NextDouble() - 0.5) / k;
                data[i] = new ClusterData
                {
                    Points = new float[2]
                    {
                        (float)(clusters[index].Points[0] + shift),
                        (float)(clusters[index].Points[1] + shift)
                    }
                };
            }
            var pipeline = new LearningPipeline();
            pipeline.Add(CollectionDataSource.Create(data));
            pipeline.Add(new KMeansPlusPlusClusterer() { K = k });
            var model = pipeline.Train<ClusterData, ClusteringPrediction>();
            //validate no initial centers of clusters belong to same cluster.
            var labels = new HashSet<uint>();
            for (int i = 0; i < k; i++)
            {
                var scores = model.Predict(clusters[i]);
                Assert.True(!labels.Contains(scores.SelectedClusterId));
                labels.Add(scores.SelectedClusterId);
            }
        }
    }
}
