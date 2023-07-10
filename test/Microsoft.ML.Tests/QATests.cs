// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Apache.Arrow;
using ICSharpCode.SharpZipLib.Tar;
using Microsoft.Data.Analysis;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch.utils;

namespace Microsoft.ML.Tests
{
    [Collection("NoParallelization")]
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

        [Fact]
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
            var prev = estimator.Fit(dataView).Transform(dataView).Preview();
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

            TestEstimatorCore(estimator, dataView);
        }
    }
}
