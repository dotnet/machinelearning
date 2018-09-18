// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.RunTests;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TreeEstimators : TestDataPipeBase
    {

        public TreeEstimators(ITestOutputHelper output) : base(output)
        {
        }


        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeBinaryEstimator()
        {
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var reader = new TextLoader(env,
                    new TextLoader.Arguments()
                    {
                        Separator = "\t",
                        HasHeader = true,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("SentimentText", DataKind.Text, 1)
                        }
                    });

                var data = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.Sentiment.trainFilename)));

                // Pipeline.
                var pipeline = new TextTransform(env, "SentimentText", "Features")
                  .Append(new FastTreeBinaryClassificationTrainer(env, "Label", "Features", advancedSettings: s => {
                      s.NumTrees = 10;
                      s.NumThreads = 1;
                      s.NumLeaves = 5;
                  }));

                TestEstimatorCore(pipeline, data);
            }
        }

        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRankerEstimator()
        {
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var reader = new TextLoader(env, new TextLoader.Arguments
                {
                    HasHeader = true,
                    Separator ="\t",
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.R4, 0),
                        new TextLoader.Column("Workclass", DataKind.Text, 1),
                        new TextLoader.Column("NumericFeatures", DataKind.R4, new [] { new TextLoader.Range(9, 14) })
                    }
                });
                var data = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.adultRanking.trainFilename)));

                // Pipeline.
                var pipeline = new TermEstimator(env, new[]{
                                    new TermTransform.ColumnInfo("Workclass", "Group"),
                                    new TermTransform.ColumnInfo("Label", "Label0") })
                    .Append(new FastTreeRankingTrainer(env, "Label0", "NumericFeatures", "Group", 
                                advancedSettings: s => { s.NumTrees = 10; }));

                TestEstimatorCore(pipeline, data);
            }
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRegressorEstimator()
        {
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // "loader=Text{col=Label:R4:11 col=Features:R4:0-10 sep=; header+}"
                var reader = new TextLoader(env,
                    new TextLoader.Arguments()
                    {
                        Separator = ";",
                        HasHeader = true,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 11),
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(0, 10) } )
                        }
                    });

                var data = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.winequalitymacro.trainFilename)));

                // Pipeline.
                var pipeline = new FastTreeRegressionTrainer(env, "Label", "Features", advancedSettings: s => {
                      s.NumTrees = 10;
                      s.NumThreads = 1;
                      s.NumLeaves = 5;
                  });

                TestEstimatorCore(pipeline, data);
            }
        }
    }
}
