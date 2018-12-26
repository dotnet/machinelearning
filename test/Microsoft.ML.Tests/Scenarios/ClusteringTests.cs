﻿using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Legacy.Transforms;
using Xunit;

namespace Microsoft.ML.Scenarios
{
#pragma warning disable 612, 618
    public partial class ScenariosTests
    {
        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/203")]
        public void PredictNewsCluster()
        {
            string dataPath = GetDataPath(@"external/20newsgroups.txt");

            var pipeline = new Legacy.LearningPipeline(seed: 1, conc: 1);
            pipeline.Add(new Legacy.Data.TextLoader(dataPath).CreateFrom<NewsData>(useHeader: false, allowQuotedStrings: true, supportSparse: false));
            pipeline.Add(new ColumnConcatenator("AllText", "Subject", "Content"));
            pipeline.Add(new TextFeaturizer("Features", "AllText")
            {
                KeepPunctuations = false,
                UsePredefinedStopWordRemover = true,
                VectorNormalizer = TextFeaturizingEstimatorTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 1, AllLengths = true }
            });

            pipeline.Add(new Legacy.Trainers.KMeansPlusPlusClusterer() { K = 20 });
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
            [LoadColumn(0)]
            public string Id;

            [LoadColumn(1) , ColumnName("Label")]
            public string Topic;

            [LoadColumn(2)]
            public string Subject;

            [LoadColumn(3)]
            public string Content;
        }

        public class ClusteringPrediction
        {
            [ColumnName("PredictedLabel")]
            public uint SelectedClusterId;
            [ColumnName("Score")]
            public float[] Distance;
        }

        public class ClusteringData
        {
            [ColumnName("Features")]
            [VectorType(2)]
            public float[] Points;
        }

        [Fact]
        public void PredictClusters()
        {
            int n = 1000;
            int k = 4;
            var rand = new Random(1);
            var clusters = new ClusteringData[k];
            var data = new ClusteringData[n];
            for (int i = 0; i < k; i++)
            {
                //pick clusters as points on circle with angle to axis X equal to 360*i/k
                clusters[i] = new ClusteringData { Points = new float[2] { (float)Math.Cos(Math.PI * i * 2 / k), (float)Math.Sin(Math.PI * i * 2 / k) } };
            }
            // create data points by randomly picking cluster and shifting point slightly away from it.
            for (int i = 0; i < n; i++)
            {
                var index = rand.Next(0, k);
                var shift = (rand.NextDouble() - 0.5) / 10;
                data[i] = new ClusteringData
                {
                    Points = new float[2]
                    {
                        (float)(clusters[index].Points[0] + shift),
                        (float)(clusters[index].Points[1] + shift)
                    }
                };
            }
            var pipeline = new Legacy.LearningPipeline(seed: 1, conc: 1);
            pipeline.Add(Legacy.Data.CollectionDataSource.Create(data));
            pipeline.Add(new Legacy.Trainers.KMeansPlusPlusClusterer() { K = k });
            var model = pipeline.Train<ClusteringData, ClusteringPrediction>();
            //validate that initial points we pick up as centers of cluster during data generation belong to different clusters.
            var labels = new HashSet<uint>();
            for (int i = 0; i < k; i++)
            {
                var scores = model.Predict(clusters[i]);
                Assert.True(!labels.Contains(scores.SelectedClusterId));
                labels.Add(scores.SelectedClusterId);
            }

            var evaluator = new Legacy.Models.ClusterEvaluator();
            var testData = Legacy.Data.CollectionDataSource.Create(clusters);
            var metrics = evaluator.Evaluate(model, testData);

            //Label is not specified, so NMI would be equal to NaN
            Assert.Equal(metrics.Nmi, double.NaN);
            //Calculate dbi is false by default so Dbi would be 0
            Assert.Equal(metrics.Dbi, (double)0.0);
            Assert.Equal(metrics.AvgMinScore, (double)0.0, 5);
        }
    }
#pragma warning restore 612, 618
}
