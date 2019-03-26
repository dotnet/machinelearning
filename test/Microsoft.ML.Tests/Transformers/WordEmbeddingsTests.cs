// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class WordEmbeddingsTests : TestDataPipeBase
    {
        public WordEmbeddingsTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        public void TestWordEmbeddings()
        {
            var dataPath = GetDataPath(TestDatasets.Sentiment.trainFilename);
            var data = new TextLoader(ML,
                   new TextLoader.Options()
                   {
                       Separator = "\t",
                       HasHeader = true,
                       Columns = new[]
                       {
                            new TextLoader.Column("Label", DataKind.Boolean, 0),
                            new TextLoader.Column("SentimentText", DataKind.String, 1)
                       }
                   }).Load(GetDataPath(dataPath));

            var est = ML.Transforms.Text.NormalizeText("NormalizedText", "SentimentText", keepDiacritics: false, keepPunctuations: false)
                  .Append(ML.Transforms.Text.TokenizeIntoWords("Words", "NormalizedText"))
                  .Append(ML.Transforms.Text.RemoveDefaultStopWords("CleanWords", "Words"));
            var words = est.Fit(data).Transform(data);

            var pipe = ML.Transforms.Text.ApplyWordEmbedding("WordEmbeddings", "CleanWords", modelKind: WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding);

            TestEstimatorCore(pipe, words, invalidInput: data);

            var outputPath = GetOutputPath("Text", "wordEmbeddings.tsv");
            var savedData = ML.Data.TakeRows(pipe.Fit(words).Transform(words), 4);
            savedData = ML.Transforms.SelectColumns("WordEmbeddings").Fit(savedData).Transform(savedData);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);
            CheckEquality("Text", "wordEmbeddings.tsv");
            Done();
        }

        [Fact]
        public void TestCustomWordEmbeddings()
        {
            var dataPath = GetDataPath(TestDatasets.Sentiment.trainFilename);
            var data = new TextLoader(ML,
                   new TextLoader.Options()
                   {
                       Separator = "\t",
                       HasHeader = true,
                       Columns = new[]
                       {
                            new TextLoader.Column("Label", DataKind.Boolean, 0),
                            new TextLoader.Column("SentimentText", DataKind.String, 1)
                       }
                   }).Load(GetDataPath(dataPath));

            var est = ML.Transforms.Text.NormalizeText("NormalizedText", "SentimentText", keepDiacritics: false, keepPunctuations: false)
                  .Append(ML.Transforms.Text.TokenizeIntoWords("Words", "NormalizedText"))
                  .Append(ML.Transforms.Text.RemoveDefaultStopWords("CleanWords", "Words"));
            var words = est.Fit(data).Transform(data);
            var pathToCustomModel = DeleteOutputPath("custommodel.txt");
            using (StreamWriter file = new StreamWriter(pathToCustomModel))
            {
                file.WriteLine("This is custom file for 4 words with 5 dimentional vector. First line in this file is ignored");
                file.WriteLine("stop" + " " + string.Join(" ", 1.5f, 2.5f, 3.5f, 4.5f, 5.5f));
                file.WriteLine("bursts" + " " + string.Join(" ", -0.9f, -3f, 7.3f, 1.0f, 12f));
                file.WriteLine("you" + " " + string.Join(" ", -1f, -2f, -4f, -6f, -1f));
                file.WriteLine("dude" + " " + string.Join(" ", 100f, 0f, 0f, 0f, 0f));
            }
            var pipe = ML.Transforms.Text.ApplyWordEmbedding("WordEmbeddings", pathToCustomModel, "CleanWords");

            TestEstimatorCore(pipe, words, invalidInput: data);

            var outputPath = GetOutputPath("Text", "customWordEmbeddings.tsv");
            var savedData = ML.Data.TakeRows(pipe.Fit(words).Transform(words), 10);
            savedData = ML.Transforms.SelectColumns("WordEmbeddings", "CleanWords").Fit(savedData).Transform(savedData);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);
            CheckEquality("Text", "customWordEmbeddings.tsv");
            Done();
        }
    }
}
