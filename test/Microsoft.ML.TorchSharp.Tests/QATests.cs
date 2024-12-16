// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.TorchSharp.Tests
{
    public class QATests : TestDataPipeBase
    {
        public QATests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestSingleSentenceData
        {
            public string Context;
            public string Question;
            public string TrainingAnswer;
            public int AnswerIndex;
        }

        [Fact(Skip = "Skip in CI build, uses too much memory.")]
        public void TestSimpleQA()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {
                        Context = "Following the disbandment of Destiny's Child in June 2005, she released her second solo album, B'Day (2006), which contained hits \"Déjà Vu\", \"Irreplaceable\", and \"Beautiful Liar\". Beyoncé also ventured into acting, with a Golden Globe-nominated performance in Dreamgirls (2006), and starring roles in The Pink Panther (2006) and Obsessed (2009). Her marriage to rapper Jay Z and portrayal of Etta James in Cadillac Records (2008) influenced her third album, I Am... Sasha Fierce (2008), which saw the birth of her alter-ego Sasha Fierce and earned a record-setting six Grammy Awards in 2010, including Song of the Year for \"Single Ladies (Put a Ring on It)\". Beyoncé took a hiatus from music in 2010 and took over management of her career; her fourth album 4 (2011) was subsequently mellower in tone, exploring 1970s funk, 1980s pop, and 1990s soul. Her critically acclaimed fifth studio album, Beyoncé (2013), was distinguished from previous releases by its experimental production and exploration of darker themes.",
                        Question = "After her second solo album, what other entertainment venture did Beyonce explore?",
                        TrainingAnswer = "acting",
                        AnswerIndex = 207
                    },
                }));
            var chain = new EstimatorChain<ITransformer>();
            var estimator = chain.Append(ML.MulticlassClassification.Trainers.QuestionAnswer(maxEpochs: 1));

            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.Equal(6, estimatorSchema.Count);
            Assert.Equal("Answer", estimatorSchema[4].Name);
            Assert.Equal("Score", estimatorSchema[5].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[4].ItemType);
            Assert.Equal(NumberDataViewType.Single, estimatorSchema[5].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(6, transformerSchema.Count);
            Assert.Equal("Answer", transformerSchema[4].Name);
            Assert.Equal("Score", transformerSchema[5].Name);
            Assert.Equal(new VectorDataViewType(TextDataViewType.Instance), transformerSchema[4].Type);
            Assert.Equal(new VectorDataViewType(NumberDataViewType.Single), transformerSchema[5].Type);

            TestEstimatorCore(estimator, dataView, shouldDispose: true);
            transformer.Dispose();
        }

        [Fact(Skip = "Needs to be on a comp with GPU or will take a LONG time.")]
        public void TestQALargeFileGpu()
        {
            ML.GpuDeviceId = 0;
            ML.FallbackToCpu = false;

            var trainFile = GetDataPath("squad-train.tsv");

            IDataView dataView = TextLoader.Create(ML, new TextLoader.Options()
            {
                Columns = new[]
                {
                new TextLoader.Column("Context", DataKind.String,0),
                new TextLoader.Column("Question", DataKind.String,1),
                new TextLoader.Column("TrainingAnswer", DataKind.String,2),
                new TextLoader.Column("AnswerIndex", DataKind.Int32,3)
                },
                HasHeader = true,
                Separators = new[] { '\t' },
                MaxRows = 2000 // Dataset has 75k rows. Only load 1k for quicker training,
            }, new MultiFileSource(trainFile));

            var estimator = ML.MulticlassClassification.Trainers.QuestionAnswer(maxEpochs: 30);
            var model = estimator.Fit(dataView);
            var transformedData = model.Transform(dataView);

            using (var cursor = transformedData.GetRowCursor(transformedData.Schema["Answer"], transformedData.Schema["Score"], transformedData.Schema["TrainingAnswer"], transformedData.Schema["Context"], transformedData.Schema["Question"]))
            {
                var answerGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(transformedData.Schema["Answer"]);
                var contextGetter = cursor.GetGetter<ReadOnlyMemory<char>>(transformedData.Schema["Context"]);
                var questionGetter = cursor.GetGetter<ReadOnlyMemory<char>>(transformedData.Schema["Question"]);
                var trainingAnswerGetter = cursor.GetGetter<ReadOnlyMemory<char>>(transformedData.Schema["TrainingAnswer"]);
                var scoreGetter = cursor.GetGetter<VBuffer<float>>(transformedData.Schema["Score"]);

                VBuffer<ReadOnlyMemory<char>> answer = default;
                ReadOnlyMemory<char> trainingAnswer = default;
                ReadOnlyMemory<char> context = default;
                ReadOnlyMemory<char> question = default;
                VBuffer<float> score = default;
                int correct = 0;
                int incorrect = 0;

                while (cursor.MoveNext())
                {
                    answerGetter(ref answer);
                    trainingAnswerGetter(ref trainingAnswer);
                    contextGetter(ref context);
                    questionGetter(ref question);
                    scoreGetter(ref score);
                    if (trainingAnswer.ToString().Contains(answer.GetValues()[0].ToString()) || answer.GetValues()[0].ToString().Contains(trainingAnswer.ToString()))
                        correct++;
                    else
                        incorrect++;
                }

                Assert.True(correct > incorrect);
            }
            model.Dispose();
        }
    }
}
