// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
    public class MemoryCollectionTests : BaseTestClass
    {
        public MemoryCollectionTests(ITestOutputHelper output)
            : base(output)
        {

        }

        [Fact]
        public void CheckConstructor()
        {
            Assert.NotNull(MemoryCollection.Create(new List<Input>() { new Input { Number1 = 1, String1 = "1" } }));
            Assert.NotNull(MemoryCollection.Create(new Input[1] { new Input { Number1 = 1, String1 = "1" } }));
            Assert.NotNull(MemoryCollection.Create(new Input[1] { new Input { Number1 = 1, String1 = "1" } }.AsEnumerable()));

            bool thrown = false;
            try
            {
                MemoryCollection.Create(new List<Input>());
            }
            catch
            {
                thrown = true;
            }
            Assert.True(thrown);

            thrown = false;
            try
            {
                MemoryCollection.Create(new Input[0]);
            }
            catch
            {
                thrown = true;
            }
            Assert.True(thrown);
        }

        [Fact]
        public void CanSuccessfullyApplyATransform()
        {
            var collection = MemoryCollection.Create(new List<Input>() { new Input { Number1 = 1, String1 = "1" } });
            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = (ILearningPipelineDataStep)collection.ApplyStep(null, experiment);

                Assert.NotNull(output.Data);
                Assert.NotNull(output.Data.VarName);
                Assert.Null(output.Model);
            }
        }

        [Fact]
        public void CanSuccessfullyEnumerated()
        {
            var collection = MemoryCollection.Create(new List<Input>() {
                new Input { Number1 = 1, String1 = "1" },
                new Input { Number1 = 2, String1 = "2" },
                new Input { Number1 = 3, String1 = "3" }
            });

            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = collection.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                experiment.Compile();
                collection.SetInput(environment, experiment);
                experiment.Run();

                IDataView data = experiment.GetOutput(output.Data);
                Assert.NotNull(data);

                using (var cursor = data.GetRowCursor((a => true)))
                {
                    var IDGetter = cursor.GetGetter<float>(0);
                    var TextGetter = cursor.GetGetter<DvText>(1);

                    Assert.True(cursor.MoveNext());

                    float ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(1, ID);

                    DvText Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("1", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(2, ID);

                    Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("2", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(3, ID);

                    Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("3", Text.ToString());

                    Assert.False(cursor.MoveNext());
                }
            }
        }

        [Fact]
        public void CanTrain()
        {
            var pipeline = new LearningPipeline();
            var data = new List<IrisData>() {
                new IrisData { SepalLength = 1f, SepalWidth = 1f ,PetalLength=0.3f, PetalWidth=5.1f, Label=1},
                new IrisData { SepalLength = 1f, SepalWidth = 1f ,PetalLength=0.3f, PetalWidth=5.1f, Label=1},
                new IrisData { SepalLength = 1.2f, SepalWidth = 0.5f ,PetalLength=0.3f, PetalWidth=5.1f, Label=0}
            };
            var collection = MemoryCollection.Create(data);

            pipeline.Add(collection);
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();

            IrisPrediction prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            pipeline = new LearningPipeline();
            collection = MemoryCollection.Create(data.AsEnumerable());
            pipeline.Add(collection);
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            model = pipeline.Train<IrisData, IrisPrediction>();

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

        }

        public class Input
        {
            [Column("0")]
            public float Number1;

            [Column("1")]
            public string String1;
        }

        public class IrisData
        {
            [Column("0")]
            public float Label;

            [Column("1")]
            public float SepalLength;

            [Column("2")]
            public float SepalWidth;

            [Column("3")]
            public float PetalLength;

            [Column("4")]
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }

    }
}
