// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers.KMeans;

namespace Microsoft.ML.Benchmarks
{
    public class KMeansAndLogisticRegressionBench
    {
        private readonly string _dataPath = Program.GetInvariantCultureDataPath("adult.train");

        [Benchmark]
        public ParameterMixingCalibratedPredictor TrainKMeansAndLR()
        {
            using (var env = new ConsoleEnvironment(seed: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env,
                    new TextLoader.Arguments()
                    {
                        HasHeader = true,
                        Separator = ",",
                        Column = new[] {
                            new TextLoader.Column("Label", DataKind.R4, 14),
                            new TextLoader.Column("CatFeatures", DataKind.TX,
                                new [] {
                                    new TextLoader.Range() { Min = 1, Max = 1 },
                                    new TextLoader.Range() { Min = 3, Max = 3 },
                                    new TextLoader.Range() { Min = 5, Max = 9 },
                                    new TextLoader.Range() { Min = 13, Max = 13 }
                                }),
                            new TextLoader.Column("NumFeatures", DataKind.R4,
                                new [] {
                                    new TextLoader.Range() { Min = 0, Max = 0 },
                                    new TextLoader.Range() { Min = 2, Max = 2 },
                                    new TextLoader.Range() { Min = 4, Max = 4 },
                                    new TextLoader.Range() { Min = 10, Max = 12 }
                                })
                        }
                    }, new MultiFileSource(_dataPath));

                IDataView trans = new CategoricalEstimator(env, "CatFeatures").Fit(loader).Transform(loader);

                trans = NormalizeTransform.CreateMinMaxNormalizer(env, trans, "NumFeatures");
                trans = new ConcatTransform(env, "Features", "NumFeatures", "CatFeatures").Transform(trans);
                trans = TrainAndScoreTransform.Create(env, new TrainAndScoreTransform.Arguments
                {
                    Trainer = ComponentFactoryUtils.CreateFromFunction(host =>
                        new KMeansPlusPlusTrainer(host, "Features", advancedSettings: s=> 
                        {
                            s.K = 100;
                        })),
                    FeatureColumn = "Features"
                }, trans);
                trans = new ConcatTransform(env, "Features", "Features", "Score").Transform(trans);

                // Train
                var trainer = new LogisticRegression(env, "Features", "Label", advancedSettings: args => { args.EnforceNonNegativity = true; args.OptTol = 1e-3f; });
                var trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                return trainer.Train(trainRoles);
            }
        }
    }
}