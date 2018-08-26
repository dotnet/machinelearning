// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;

namespace Microsoft.ML.Benchmarks
{
    public class KMeansAndLogisticRegressionBench
    {
        private readonly string _dataPath = Program.GetInvariantCultureDataPath("adult.train");

        [Benchmark]
        public IPredictor TrainKMeansAndLR()
        {
            using (var env = new TlcEnvironment(seed: 1))
            {
                // Pipeline
                var loader = new TextLoader(env,
                    new TextLoader.Arguments()
                    {
                        HasHeader = true,
                        Separator = ",",
                        Column = new[] {
                            new TextLoader.Column()
                            {
                                Name = "Label",
                                Source = new [] { new TextLoader.Range() { Min = 14, Max = 14} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "CatFeatures",
                                Source = new [] {
                                    new TextLoader.Range() { Min = 1, Max = 1 },
                                    new TextLoader.Range() { Min = 3, Max = 3 },
                                    new TextLoader.Range() { Min = 5, Max = 9 },
                                    new TextLoader.Range() { Min = 13, Max = 13 }
                                },
                                Type = DataKind.TX
                            },
                            new TextLoader.Column()
                            {
                                Name = "NumFeatures",
                                Source = new [] {
                                    new TextLoader.Range() { Min = 0, Max = 0 },
                                    new TextLoader.Range() { Min = 2, Max = 2 },
                                    new TextLoader.Range() { Min = 4, Max = 4 },
                                    new TextLoader.Range() { Min = 10, Max = 12 }
                                },
                                Type = DataKind.R4
                            }
                        }
                    }, new MultiFileSource(_dataPath));

                IDataTransform trans = CategoricalTransform.Create(env, new CategoricalTransform.Arguments
                {
                    Column = new[]
                    {
                        new CategoricalTransform.Column { Name = "CatFeatures", Source = "CatFeatures" }
                    }
                }, loader);

                trans = NormalizeTransform.CreateMinMaxNormalizer(env, trans, "NumFeatures");
                trans = new ConcatTransform(env, trans, "Features", "NumFeatures", "CatFeatures");
                trans = TrainAndScoreTransform.Create(env, new TrainAndScoreTransform.Arguments
                {
                    Trainer = new SubComponent<ITrainer, SignatureTrainer>("KMeans", "k=100"),
                    FeatureColumn = "Features"
                }, trans);
                trans = new ConcatTransform(env, trans, "Features", "Features", "Score");

                // Train
                var trainer = new LogisticRegression(env, new LogisticRegression.Arguments() { EnforceNonNegativity = true, OptTol = 1e-3f });
                var trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                return trainer.Train(trainRoles);
            }
        }
    }
}
